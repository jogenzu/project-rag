# app.py
from fastapi import FastAPI,Request, HTTPException, UploadFile, File, Query, Depends, Cookie
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
import json
import uuid
from datetime import datetime, timedelta
import asyncio
import sqlite3
from typing import Dict, Optional
from contextlib import asynccontextmanager
import urllib.parse
import file_utils
from dotenv import load_dotenv
import logging.config
import bcrypt
from jose import JWTError, jwt

logging.config.fileConfig('/app/logging.ini')

# 判断当前目录下.env 文件是否存在
if not os.path.exists(".env"):
    raise FileNotFoundError("请在当前目录下创建.env文件")


print("如果是首次使用，请打开.env 进行大模型key配置 ")

# 加载.env文件
load_dotenv()


# 创建应用启动上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前执行
    init()
    init_db()  # 初始化数据库
    init_admin_user()  # 初始化默认管理员账号
    load_documents()
    yield
    # 关闭时执行
    save_documents()
    # 可以在这里添加清理代码

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan, debug=True)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/chat.html")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 添加CORS中间件允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 SQLite 数据库
def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 创建聊天会话表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        summary TEXT,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')

    # 创建消息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_id INTEGER,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')

    # 数据库迁移：为旧数据库添加user_id列
    try:
        # 检查messages表是否有user_id列
        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'user_id' not in columns:
            # 添加user_id列
            cursor.execute("ALTER TABLE messages ADD COLUMN user_id INTEGER")
            print("数据库迁移：已为messages表添加user_id列")

        # 检查chat_sessions表是否有user_id列
        cursor.execute("PRAGMA table_info(chat_sessions)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'user_id' not in columns:
            # 添加user_id列
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN user_id INTEGER")
            print("数据库迁移：已为chat_sessions表添加user_id列")
    except Exception as e:
        print(f"数据库迁移警告: {str(e)}")

    conn.commit()
    conn.close()
    print("数据库初始化完成!\n")

# 初始化默认管理员账号
def init_admin_user():
    # 默认管理员配置（硬编码在数据库中）
    admin_username = "admin"
    admin_password = "admin123"
    
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    # 检查管理员是否存在
    cursor.execute("SELECT id FROM users WHERE username = ?", (admin_username,))
    exists = cursor.fetchone()
    
    if not exists:
        # 加密密码
        password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # 创建管理员账号
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (admin_username, password_hash, 'admin')
        )
        conn.commit()
        print(f"默认管理员账号已创建 - 用户名: {admin_username}, 密码: {admin_password}")
    else:
        print(f"管理员账号已存在 - 用户名: {admin_username}")
    
    conn.close()

# JWT 工具函数
def create_access_token(data: dict):
    """创建 JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=int(os.getenv("JWT_EXPIRE_HOURS", 24)))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("JWT_SECRET_KEY"), algorithm=os.getenv("JWT_ALGORITHM", "HS256"))
    return encoded_jwt

def decode_access_token(token: str):
    """解码 JWT Token"""
    try:
        secret_key = os.getenv("JWT_SECRET_KEY")
        algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        print(f"[DEBUG] Decoding token with secret_key: {secret_key[:10] if secret_key else 'None'}..., algorithm: {algorithm}")
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except JWTError as e:
        print(f"[DEBUG] JWT decode error: {e}")
        return None

def get_current_user(auth_token: Optional[str] = Cookie(None)):
    """获取当前登录用户"""
    print(f"[DEBUG] get_current_user called, auth_token: {auth_token}")
    if auth_token is None:
        print("[DEBUG] auth_token is None, returning None")
        return None
    token = auth_token
    
    payload = decode_access_token(token)
    print(f"[DEBUG] decoded payload: {payload}")
    if payload is None:
        print("[DEBUG] payload is None, returning None")
        return None
    
    user_id = payload.get("sub")
    print(f"[DEBUG] user_id from payload: {user_id} (type: {type(user_id)})")
    if user_id is None:
        print("[DEBUG] user_id is None, returning None")
        return None
    
    # 将user_id转换为整数
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        print(f"[DEBUG] Failed to convert user_id to int: {user_id}")
        return None
    
    conn = sqlite3.connect('chat_history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    print(f"[DEBUG] user from db: {user}")
    if user:
        return dict(user)
    return None

# Pydantic 模型
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChangePassword(BaseModel):
    username: str
    new_password: str

# 全局变量
model = None
index = None
documents = []
document_to_chunks = {}
chunks_to_document = {}
all_chunks = []
client = None

# 文档和会话存储
uploaded_documents: Dict[str, Dict] = {}  # {id: {name, content, path}}
chat_sessions: Dict[str, Dict] = {}  # {id: {summary, updated_at, messages}}

# 保存和加载文档数据
def save_documents():
    # 创建一个可序列化的版本（不包含文件内容以减少文件大小），因为有文件上传下载功能，所以这里需要实时更新保存
    serializable_docs = {}
    for doc_id, doc_data in uploaded_documents.items():
        serializable_docs[doc_id] = {
            "name": doc_data["name"],
            "path": doc_data["path"]
        }
    
    with open("/app/docs/documents_index.json", "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)


def load_documents():
    global uploaded_documents  # 全局变量，为了在rebuild_index() 中使用
    
    index_path = "/app/docs/documents_index.json"
    if not os.path.exists(index_path):
        return
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            serialized_docs = json.load(f)
        
        # 加载文档元数据和内容
        for doc_id, doc_data in serialized_docs.items():
            path = doc_data.get("path")
            if path and os.path.exists(path):
                # 加载实际内容
                if path.endswith(".pdf"):
                    content_text, error  = file_utils.load_pdf_file(path)
                elif path.endswith(".txt"):
                    content_text, error = file_utils.load_text_file(path)
                elif path.endswith(".docx"):
                    content_text, error = file_utils.load_docx_file(path)
                else:
                    continue
                
                # 是个字典，主要把content_text 加载进去。docs目录下的所有文档都存存进字典了
                uploaded_documents[doc_id] = {
                    "name": doc_data["name"],
                    "path": path,
                    "content": content_text
                }
        
        # 重建索引
        rebuild_index()
        print("文档加载完成！\n")
    except Exception as e:
        print(f"加载文档索引失败: {str(e)}")

# 初始化函数
def init():
    global model, index, client
    
    # 初始化AI客户端
    client = OpenAI(
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL")
    )

    print("\nAI客户端初始化完成!\n")
    
    # 加载嵌入模型
    local_model_path = 'local_m3e_model'
    if os.path.exists(local_model_path):
        model = SentenceTransformer(local_model_path)
    else:
        model = SentenceTransformer('moka-ai/m3e-base')
        model.save(local_model_path)
    print("模型加载完成!\n")


# 文本分块函数
def chunk_document(text, max_chars=500, overlap=20):
    """
    将中文文本按指定最大字符数分割成块，支持重叠。
    
    参数：
        text (str): 输入的中文文本
        max_chars (int): 每个块的最大字符数，默认为500
        overlap (int): 相邻块之间的重叠字符数，默认为5
    
    返回：
        list: 分割后的文本块列表
    """
    if not text:
        return []
    
    chunks = []
    text_length = len(text)
    start = 0
    
    while start < text_length:
        # 计算当前块的结束位置
        end = min(start + max_chars, text_length)
        # 确保不截断中文字符
        chunk = text[start:end]
        chunks.append(chunk)
        # 更新起始位置，考虑重叠
        start += max_chars - overlap
    
    return chunks

# 重新构建索引
def rebuild_index():
    print("----开始重建文档索引...：")
    global index, document_to_chunks, chunks_to_document, all_chunks
    
    # 重置数据
    document_to_chunks = {}
    chunks_to_document = {}
    all_chunks = []
    
    # 处理上传的文档 ，对长文档进行分块
    for doc_id, doc_data in uploaded_documents.items():
        content = doc_data.get("content", "")
        chunks = chunk_document(content)
        # 存储映射关系
        document_to_chunks[doc_id] = []
        for chunk in chunks:
            chunk_id = len(all_chunks)
            all_chunks.append(chunk)
            document_to_chunks[doc_id].append(chunk_id)
            chunks_to_document[chunk_id] = doc_id
    
    # 如果没有文档，不创建索引
    if not all_chunks:
        index = None
        return
    
    #print(f"all_chunks: {all_chunks}")
    print(f"打印最后一个chunk: {all_chunks[-1]}")
    print(f"chunks_count: {len(all_chunks)}")
        
    # 生成嵌入
    chunk_embeddings = get_embeddings(all_chunks)
    
    # 初始化FAISS索引
    dimension = chunk_embeddings.shape[1]  # 768 for m3e-base
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    
    # 保存索引
    faiss.write_index(index, "m3e_faiss_index.bin")
    
    # 保存映射关系
    mapping_data = {
        'doc_to_chunks': document_to_chunks,
        'chunks_to_doc': chunks_to_document,
        'all_chunks': all_chunks
    }
    np.save("chunks_mapping.npy", mapping_data)
    
    print("\n----索引创建并保存成功！\n")

# 获取嵌入向量
def get_embeddings(texts):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)

# 检索函数
def retrieve_docs(query, k=3):
    if index is None or not all_chunks:
        return [], []
        
    query_embedding = get_embeddings([query])
    distances, chunk_indices = index.search(query_embedding, k)
    
    # 获取包含这些chunks的原始文档
    retrieved_doc_ids = set()
    retrieved_chunks = []
    
    for chunk_idx in chunk_indices[0]:
        if chunk_idx >= 0 and chunk_idx < len(all_chunks):
            doc_id = chunks_to_document.get(int(chunk_idx))
            if doc_id is not None:
                retrieved_doc_ids.add(doc_id)
                retrieved_chunks.append((doc_id, all_chunks[int(chunk_idx)]))
    
    # 获取原始文档详情
    retrieved_docs = []
    for doc_id in retrieved_doc_ids:
        if doc_id in uploaded_documents:
            retrieved_docs.append(f"文档: {uploaded_documents[doc_id]['name']}")
    
    return retrieved_docs, retrieved_chunks



# ========== 认证相关 API ==========

@app.post("/api/auth/register")
async def register(user: UserRegister):
    """用户注册"""
    # 检查用户名是否已存在
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (user.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 加密密码
    password_hash = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # 创建用户
    cursor.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        (user.username, password_hash, 'user')
    )
    conn.commit()
    conn.close()
    
    return {"message": "注册成功", "username": user.username}

@app.post("/api/auth/login")
async def login(user: UserLogin):
    """用户登录"""
    conn = sqlite3.connect('chat_history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 查询用户
    cursor.execute("SELECT * FROM users WHERE username = ?", (user.username,))
    db_user = cursor.fetchone()
    conn.close()
    
    if not db_user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    # 验证密码
    if not bcrypt.checkpw(user.password.encode('utf-8'), db_user['password_hash'].encode('utf-8')):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    # 生成 Token - 将user_id转换为字符串
    token = create_access_token({"sub": str(db_user['id']), "username": db_user['username'], "role": db_user['role']})
    
    # 设置 Cookie
    response = JSONResponse(content={"message": "登录成功", "token": token, "user": {"id": db_user['id'], "username": db_user['username'], "role": db_user['role']}})
    response.set_cookie(
        key="auth_token",
        value=token,
        max_age=3600 * int(os.getenv("JWT_EXPIRE_HOURS", 24)),
        httponly=True,
        samesite="lax"
    )
    return response

@app.post("/api/auth/logout")
async def logout():
    """用户登出"""
    response = JSONResponse(content={"message": "登出成功"})
    response.delete_cookie("auth_token")
    return response

@app.get("/api/auth/me")
async def get_current_user_info(current_user: Optional[Dict] = Depends(get_current_user)):
    """获取当前用户信息"""
    print(f"[DEBUG] /api/auth/me called, current_user: {current_user}")
    if current_user is None:
        return {"user": None}
    return {"user": {"id": current_user['id'], "username": current_user['username'], "role": current_user['role']}}

@app.post("/api/auth/change-password")
async def change_password(password_data: ChangePassword):
    """修改密码（无需身份验证，通过用户名直接修改）"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    # 检查用户是否存在
    cursor.execute("SELECT id FROM users WHERE username = ?", (password_data.username,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 加密新密码
    password_hash = bcrypt.hashpw(password_data.new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # 更新密码
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (password_hash, password_data.username)
    )
    conn.commit()
    conn.close()
    
    return {"message": "密码修改成功"}

@app.get("/api/auth/verify")
async def verify_token(current_user: Optional[Dict] = Depends(get_current_user)):
    """验证 Token 有效性"""
    if current_user is None:
        return {"valid": False}
    return {"valid": True, "user": {"id": current_user['id'], "username": current_user['username'], "role": current_user['role']}}




# ========== 权限装饰器 ==========
def require_role(allowed_roles: list):
    """权限装饰器：要求用户具有指定角色之一"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 从请求中获取当前用户
            request = kwargs.get('request') or args[0] if args else None
            if request:
                current_user = get_current_user(request.cookies.get('auth_token'))
            else:
                current_user = None
            
            # 检查用户权限
            if current_user is None or current_user['role'] not in allowed_roles:
                raise HTTPException(status_code=403, detail="权限不足")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ========== 文档管理 API ==========
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), current_user: Optional[Dict] = Depends(get_current_user)):
    # 检查管理员权限
    if current_user is None or current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="需要管理员权限")

    if not file.filename.endswith((".txt", ".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="仅支持.txt或.pdf或.docx文件!")
    
    # 确保docs目录存在
    os.makedirs("docs", exist_ok=True) # 创建docs 目录,如果不存在的话，exist_ok=True表示如果目录已存在则不报错
    
    # 保存文件到docs目录
    file_path = os.path.join("docs", file.filename)
    
    # 检查文件名是否重复，如果重复则添加时间戳
    if os.path.exists(file_path):
        filename, extension = os.path.splitext(file.filename) # 分离文件名和扩展名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join("docs", f"{filename}_{timestamp}{extension}")
        file_name = f"{filename}_{timestamp}{extension}"
    else:
        file_name = file.filename
        
    # 读取文件内容
    content = await file.read()
    
    # 保存文件到磁盘
    with open(file_path, "wb") as f:
        f.write(content)
    
    #依据扩展名加载内容 
    if file.filename.endswith(".pdf"):
        content_text, error = file_utils.load_pdf_file(file_path)
    elif file.filename.endswith(".txt"):
        content_text, error = file_utils.load_text_file(file_path)
    elif file.filename.endswith(".docx"):
        content_text, error = file_utils.load_docx_file(file_path)
     
    doc_id = str(uuid.uuid4())
    uploaded_documents[doc_id] = {
        "name": file_name,
        "content": content_text,
        "path": file_path
    }
    
    # 重建索引
    rebuild_index()
    
    # 保存文档索引
    save_documents()
    
    return {"id": doc_id, "name": file_name}

@app.get("/api/documents")
async def list_documents(current_user: Optional[Dict] = Depends(get_current_user)):
    # 检查管理员权限
    if current_user is None or current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return [{"id": k, "name": v["name"]} for k, v in uploaded_documents.items()]

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    # 检查管理员权限
    if current_user is None or current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    if doc_id not in uploaded_documents:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    # 删除文件
    file_path = uploaded_documents[doc_id].get("path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"删除文件时出错: {str(e)}")
    
    # 从内存中删除记录（这是删除字典的某个键值的方法）
    del uploaded_documents[doc_id]  # del 是 Python 的内置语句​（不是函数），用于删除对象的引用,这
                                    #里从字典 uploaded_documents 中删除键为 doc_id 的键值对
    
    # 重建索引
    rebuild_index()
    
    # 保存文档索引
    save_documents()
    
    return {"message": "删除成功"}


@app.post("/api/stream")
async def stream_post(request: Request, current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        # 解析请求体中的 JSON 数据
        req_data = await request.json()
        query = req_data.get("query")
        session_id = req_data.get("session_id")  # 获取会话ID
        web_search = req_data.get("web_search", False)  # 获取联网搜索选项
        return await process_stream_request(query, session_id, web_search, current_user)
    except Exception as e:
        error_msg = str(e)
        print(f"聊天接口错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/stream")
async def stream_get(query: str = Query(None), session_id: str = Query(None), web_search: bool = Query(False), current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")
        return await process_stream_request(query, session_id, web_search, current_user)
    except Exception as e:
        error_msg = str(e)
        print(f"聊天接口错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)



 
# 执行网络搜索
async def perform_web_search(query: str):
    try:
        import requests
        
        
        # 使用Google搜索
        query = urllib.parse.quote(query)
        #search_url = f"https://www.google.com/search?q={encoded_query}"
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&start=0"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        data = response.json()  # Parse JSON response
      
        print(f"Google search response: {data}")
        
        if response.status_code != 200:
            return f"搜索失败，状态码: {response.status_code}"
 
        return str(data)
            
    except Exception as e:
        return f"执行网络搜索时出错: {str(e)}"

async def process_stream_request(query: str, session_id: str = None, web_search: bool = False, current_user: Optional[Dict] = None):
    print(f"query: {query}, session_id: {session_id}, web_search: {web_search}, current_user: {current_user}")
    # 检查session是否存在
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
    has_session = cursor.fetchone()
    
    # 获取用户ID
    user_id = None
    if current_user and 'id' in current_user:
        user_id = current_user['id']
    
    conn.close()
    
    if not has_session:
        session_id = str(uuid.uuid4())
    
    # 构建上下文
    context_parts = []
    
    # 如果启用了网络搜索，添加网络搜索结果
    if web_search:
        web_results = await perform_web_search(query)
        print(f"web_results: {web_results}")
        context_parts.append(web_results)
    
    # 检索相关文档
    retrieved_docs, retrieved_chunks = retrieve_docs(query)
    
    # 添加文档检索结果
    context_parts.append("相关文档:\n" + "\n".join(retrieved_docs))
    
    if retrieved_chunks:
        chunk_context = "\n\n文档内容片段:\n"
        for i, (doc_id, chunk) in enumerate(retrieved_chunks):
            doc_name = "未知文档"
            if doc_id in uploaded_documents:
                doc_name = uploaded_documents[doc_id]["name"]
            chunk_context += f"[文档{i+1}: {doc_name}] {chunk}\n"
        context_parts.append(chunk_context)
    else:
        context_parts.append("\n\n没有找到相关的文档内容。")
    
    # 合并上下文
    context = "\n".join(context_parts)
    
    prompt = f"上下文信息:\n{context}\n\n问题: {query}\n请基于上下文信息回答问题，如果上下文中没有相关信息，请回答咱们的资源库中没有相关信息，不要编造答案。"
    print("****",prompt)
    
    # 用于保存完整响应
    full_response = ""
    
    # 创建stream响应    
    async def generate():
        nonlocal full_response
        
        system_message = "你是一个专业的问答助手。"
        
        #if web_search:
        #    system_message += "你拥有联网搜索能力，可以搜索互联网以提供最新的信息。"
        #else:
        #    system_message += "请仅基于提供的上下文信息回答问题，不要添加任何未在上下文中提及的信息。"
        #    
        #system_message += "如果没有相关信息，请直接告知用户咱们的资料库中没有相关信息,无法回答该问题。"
        
        stream = client.chat.completions.create(
            model= os.getenv("MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                #print(f"$$$$$$$$ {content}")
                yield f"data: {json.dumps({'content': content, 'session_id': session_id})}\n\n"
                await asyncio.sleep(0.01)  # 添加小延迟确保流式输出
            
                
            if chunk.choices[0].finish_reason is not None:
                yield f"data: {json.dumps({'content': '', 'session_id': session_id, 'done': True})}\n\n"
                break
        print("----------",full_response)
        # 响应完成后，将完整会话保存到数据库
        if has_session:
            await add_message_to_session(session_id, query, full_response, user_id)
        else:
            await create_new_chat_session(session_id, query, full_response, user_id)
            
    return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )

# 创建新的聊天会话
async def create_new_chat_session(session_id, query, response, user_id=None):
    # 创建会话摘要
    summary = query[:30] + "..." if len(query) > 30 else query
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 插入会话记录
        cursor.execute(
            "INSERT INTO chat_sessions (id, summary, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, summary, user_id, current_time, current_time)
        )
        
        # 插入用户消息
        cursor.execute(
            "INSERT INTO messages (session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, "user", query, current_time)
        )
        
        # 插入机器人响应
        cursor.execute(
            "INSERT INTO messages (session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, "bot", response, current_time)
        )
        
        conn.commit()
        conn.close()
        
        print(f"创建新会话 {session_id} 成功")
        return True
    except Exception as e:
        print(f"创建新会话失败: {str(e)}")
        return False

# 向现有会话添加消息
async def add_message_to_session(session_id, query, response, user_id=None):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 插入用户消息
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", query, current_time)
        )
        
        # 插入机器人响应
        cursor.execute(
            "INSERT INTO messages (session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, "bot", response, current_time)
        )
        
        # 更新会话时间戳，使其保持最新
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (current_time, session_id)
        )
        
        conn.commit()
        conn.close()
        
        print(f"向会话 {session_id} 添加消息成功")
        return True
    except Exception as e:
        print(f"向会话添加消息失败: {str(e)}")
        return False

# 会话历史记录 API
@app.get("/api/chat/history")
async def get_chat_history(current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        conn = sqlite3.connect('chat_history.db')
        conn.row_factory = sqlite3.Row  # 启用行工厂，使结果可以通过列名访问
        cursor = conn.cursor()
        
        # 根据用户角色返回不同的历史记录
        if current_user is None:
            # 游客：只能看到公共聊天记录（user_id 为 NULL 的）
            cursor.execute("""
                SELECT DISTINCT cs.id, cs.summary, cs.updated_at 
                FROM chat_sessions cs
                JOIN messages m ON cs.id = m.session_id
                WHERE m.user_id IS NULL
                ORDER BY cs.updated_at DESC LIMIT 10
            """)
        elif current_user['role'] == 'admin':
            # 管理员：可以看到所有会话
            cursor.execute("SELECT id, summary, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT 10")
        else:
            # 普通用户：只能看到自己的会话
            cursor.execute("""
                SELECT id, summary, updated_at 
                FROM chat_sessions 
                WHERE user_id = ?
                ORDER BY updated_at DESC LIMIT 10
            """, (current_user['id'],))
        
        rows = cursor.fetchall()
        
        # 将行转换为字典
        sessions = [dict(row) for row in rows]
        
        conn.close()
        return sessions
        
    except Exception as e:
        print(f"获取聊天历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取聊天历史失败: {str(e)}")

@app.get("/api/chat/session/{session_id}")
async def get_session(session_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        conn = sqlite3.connect('chat_history.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 查询会话是否存在
        cursor.execute("SELECT id, user_id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 权限检查
        if current_user is None:
            # 游客：只能查看公共会话（user_id 为 NULL）
            if session['user_id'] is not None:
                conn.close()
                raise HTTPException(status_code=403, detail="权限不足")
        elif current_user['role'] != 'admin':
            # 普通用户：只能查看自己的会话
            if session['user_id'] != current_user['id']:
                conn.close()
                raise HTTPException(status_code=403, detail="权限不足")
        
        # 获取会话中的所有消息
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id asc",
            (session_id,)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return {"messages": messages}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取会话详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话详情失败: {str(e)}")

# 删除会话
@app.delete("/api/chat/session/{session_id}")
async def delete_session(session_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 查询会话信息
        cursor.execute("SELECT id, user_id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 权限检查
        if current_user is None:
            # 游客：不能删除会话
            conn.close()
            raise HTTPException(status_code=403, detail="权限不足")
        elif current_user['role'] != 'admin':
            # 普通用户：只能删除自己的会话
            if session['user_id'] != current_user['id']:
                conn.close()
                raise HTTPException(status_code=403, detail="权限不足")
        
        # 首先删除会话关联的所有消息
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        # 然后删除会话本身
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        conn.commit()
        conn.close()
        
        return {"message": "会话已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"删除会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

# 修改会话的summary
@app.post("/api/chat/session/{session_id}/summary")
async def update_session_summary(session_id: str, request: Request, current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        #从post获取summary
        req_data = await request.json()
        summary = req_data.get("summary")
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 查询会话信息
        cursor.execute("SELECT id, user_id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 权限检查
        if current_user is None:
            # 游客：不能修改会话
            conn.close()
            raise HTTPException(status_code=403, detail="权限不足")
        elif current_user['role'] != 'admin':
            # 普通用户：只能修改自己的会话
            if session['user_id'] != current_user['id']:
                conn.close()
                raise HTTPException(status_code=403, detail="权限不足")
 
        # 更新会话
        cursor.execute("UPDATE chat_sessions SET summary = ? WHERE id = ?", (summary, session_id))
        conn.commit()
        conn.close()
        
        return {"message": "会话已修改"}
   
    except HTTPException:
        raise
    except Exception as e:
        print(f"修改会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"修改会话失败: {str(e)}")

# 导出会话为markdown格式下载
@app.get("/api/chat/export/{session_id}")
async def export_session(session_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    try:
        conn = sqlite3.connect('chat_history.db')
        conn.row_factory = sqlite3.Row  # Set row factory to enable dictionary access
        cursor = conn.cursor()
        
        # 查询会话是否存在
        cursor.execute("SELECT id, summary, user_id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 权限检查
        if current_user is None:
            # 游客：只能导出公共会话（user_id 为 NULL）
            if session['user_id'] is not None:
                conn.close()
                raise HTTPException(status_code=403, detail="权限不足")
        elif current_user['role'] != 'admin':
            # 普通用户：只能导出自己的会话
            if session['user_id'] != current_user['id']:
                conn.close()
                raise HTTPException(status_code=403, detail="权限不足")
        
        # 获取会话中的所有消息
        cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id asc", (session_id,))
        messages = cursor.fetchall()
        
        # 构建markdown内容
        markdown_content = f"# 会话历史记录\n\n"
        markdown_content += f"## 会话ID: {session_id}\n\n"
        markdown_content += f"## 会话总结: {session['summary']}\n\n"
        
        for message in messages:
            role = message['role']
            content = message['content']
            markdown_content += f"### {role}\n\n{content}\n\n"
        
        conn.close()
        
        return StreamingResponse(
            iter([markdown_content]),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=session_{session_id}.md"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"导出会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出会话失败: {str(e)}")

# 删除单条消息（仅管理员）
@app.delete("/api/chat/message/{message_id}")
async def delete_message(message_id: int, current_user: Optional[Dict] = Depends(get_current_user)):
    # 检查管理员权限
    if current_user is None or current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 删除消息
        cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="消息不存在")
        
        conn.commit()
        conn.close()
        
        return {"message": "消息已删除"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"删除消息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除消息失败: {str(e)}")


# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config="/app/logging.ini")
