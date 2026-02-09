# Project-RAG 项目说明文档

## 项目概述

Project-RAG 是一个基于检索增强生成（Retrieval-Augmented Generation, RAG）技术的智能问答系统。该系统结合了本地向量检索和大语言模型（LLM），能够基于上传的文档进行精准的问答对话。

## 技术栈

- **后端框架**: FastAPI
- **前端框架**: Vue.js 3
- **嵌入模型**: M3E (中文向量模型)
- **向量数据库**: FAISS
- **大语言模型**: 支持多种 LLM（OpenAI、通义千问、GLM等）
- **文档处理**: PyPDF2、python-docx、pandas
- **数据库**: SQLite（聊天历史存储）
- **部署**: Docker + Docker Compose

## 主要功能

### 1. 文档管理功能
- **文档上传**: 支持 PDF、TXT、DOCX 格式文件上传
- **文档解析**: 自动提取文档内容并清理格式
- **文档存储**: 文档内容保存到服务器，支持持久化
- **文档索引**: 自动构建向量索引，支持语义检索
- **文档删除**: 可删除已上传的文档，自动重建索引

### 2. 智能问答功能
- **基于文档的问答**: 基于上传的文档内容回答用户问题
- **流式响应**: 支持流式输出，提升用户体验
- **上下文检索**: 自动检索相关文档片段，构建上下文
- **联网搜索**: 支持 Google 联网搜索（需配置 API）
- **语音输入**: 支持语音输入识别（浏览器原生）
- **语音输出**: 支持 TTS 语音播报，可自动播放或手动播放

### 3. 聊天会话管理
- **会话创建**: 自动创建新会话，生成唯一会话ID
- **会话历史**: 保存所有聊天记录到 SQLite 数据库
- **会话列表**: 显示最近10条历史会话（按时间倒序）
- **会话加载**: 点击历史会话可加载完整对话内容
- **会话导出**: 支持将会话导出为 Markdown 文件
- **会话删除**: 可删除不需要的历史会话
- **会话重命名**: 支持修改会话标题

### 4. 向量检索功能
- **文本分块**: 智能文档分块（默认500字符，重叠20字符）
- **向量嵌入**: 使用 M3E 模型生成文本向量
- **相似度检索**: 基于 FAISS 的快速相似度搜索
- **索引持久化**: 索引文件保存到本地，加速启动
- **索引重建**: 文档变更时自动重建索引

## 文件结构说明

```
project-rag/
├── app/                          # 应用主目录
│   ├── app.py                    # FastAPI 后端主程序
│   ├── file_utils.py             # 文件处理工具（PDF、TXT、DOCX等）
│   ├── Dockerfile                # Docker 镜像构建文件
│   ├── requirements.txt          # Python 依赖列表
│   ├── .env                      # 环境变量配置（模型API等）
│   ├── .dockerignore             # Docker 忽略文件配置
│   ├── logging.ini               # 日志配置文件
│   ├── chat_history.db           # SQLite 数据库（聊天历史）
│   ├── chunks_mapping.npy        # 文档块映射关系
│   ├── m3e_faiss_index.bin       # FAISS 向量索引文件
│   ├── static/                   # 静态资源目录
│   │   ├── chat.html             # 聊天界面（主页面）
│   │   ├── chat2.html            # 聊天界面（备用页面，未启用）
│   │   └── documents.html        # 文档管理页面
│   ├── docs/                     # 上传的文档存储目录
│   │   ├── documents_index.json  # 文档索引元数据
│   │   └── ...                   # 各种格式文档（PDF、TXT等）
│   ├── local_m3e_model/          # 本地 M3E 嵌入模型
│   │   ├── config.json           # 模型配置
│   │   ├── model.safetensors     # 模型权重
│   │   ├── tokenizer.json        # 分词器
│   │   └── ...                   # 其他模型文件
│   └── logs/                     # 日志目录
├── docker-compose.yml            # Docker Compose 配置
└── .gitignore                    # Git 忽略文件配置
```

## 核心文件说明

### app/app.py - 后端主程序
FastAPI 应用主文件，包含所有后端接口和业务逻辑：

**主要功能模块：**
- 应用初始化和生命周期管理
- 数据库初始化（SQLite）
- 文档上传、删除、列表管理
- 文档向量索引构建和检索
- 聊天流式响应（SSE）
- 联网搜索（Google Custom Search API）
- 会话管理（创建、加载、删除、导出）

**核心 API 接口：**
- `POST /api/upload` - 上传文档
- `GET /api/documents` - 获取文档列表
- `DELETE /api/documents/{doc_id}` - 删除文档
- `POST /api/stream` - 聊天流式响应
- `GET /api/chat/history` - 获取聊天历史
- `GET /api/chat/session/{session_id}` - 获取会话详情
- `DELETE /api/chat/session/{session_id}` - 删除会话
- `POST /api/chat/session/{session_id}/summary` - 修改会话标题
- `GET /api/chat/export/{session_id}` - 导出会话为 Markdown
- `GET /health` - 健康检查

### app/file_utils.py - 文件处理工具
提供多种文件格式的解析功能：

**支持文件格式：**
- `.txt` - 文本文件（UTF-8、GBK 编码）
- `.pdf` - PDF 文档（使用 PyPDF2）
- `.docx` - Word 文档（使用 python-docx）
- `.md` - Markdown 文档
- `.xlsx/.xls` - Excel 表格（使用 pandas）

**主要函数：**
- `load_text_file()` - 加载文本文件
- `load_pdf_file()` - 加载 PDF 文件
- `load_docx_file()` - 加载 Word 文档
- `load_markdown_file()` - 加载 Markdown 文件
- `load_excel_file()` - 加载 Excel 文件
- `load_documents_from_directory()` - 批量加载目录下文档
- `clean_text()` - 清理文本格式

### app/static/chat.html - 聊天界面
Vue.js 3 单页应用，提供聊天交互界面：

**主要功能：**
- 消息输入和发送
- 流式响应显示（SSE）
- 历史会话列表
- 会话加载、删除、导出、重命名
- 语音输入识别（Web Speech API）
- 语音播放（TTS）
- Markdown 渲染（使用 marked.js）
- 代码高亮（使用 highlight.js）
- 自动语音播放开关
- 侧边栏折叠/展开

**UI 特性：**
- 响应式设计，支持移动端
- 美观的聊天气泡样式
- 实时滚动到最新消息
- 会话时间智能显示（今天/昨天/星期）

### app/static/documents.html - 文档管理页面
Vue.js 3 单页应用，提供文档管理界面：

**主要功能：**
- 文档上传（支持 PDF、TXT）
- 文档列表展示
- 文档删除
- 导航到聊天页面

### app/Dockerfile - Docker 镜像构建
定义容器镜像构建规则：
- 基于 Python 3.10-slim
- 安装依赖（使用清华源加速）
- 复制应用代码
- 配置 Uvicorn 启动命令

### docker-compose.yml - Docker 编排配置
Docker Compose 配置文件：
- 服务端口映射：8000:8000
- 挂载卷：./app:/app（开发模式热重载）
- 环境变量：
  - `PYTHONUNBUFFERED=1` - Python 输出不缓冲
  - `TZ=Asia/Shanghai` - 时区设置为上海
- 兼容 Windows 和 Linux

### app/requirements.txt - Python 依赖
项目依赖包列表：
- `fastapi` - Web 框架
- `uvicorn` - ASGI 服务器
- `sentence-transformers` - 嵌入模型
- `faiss-cpu` - 向量检索
- `numpy` - 数值计算
- `openai` - LLM API 客户端
- `PyPDF2` - PDF 解析
- `python-docx` - Word 解析
- `pandas` - Excel 解析
- `beautifulsoup4` - HTML 解析
- `markdown` - Markdown 处理
- `python-dotenv` - 环境变量加载

### app/.env - 环境变量配置
模型 API 配置文件：
- `MODEL_NAME` - 模型名称（如 qwen-plus、gpt-4o、glm-4-plus）
- `MODEL_BASE_URL` - API 基础 URL
- `MODEL_API_KEY` - API 密钥

**支持的模型：**
- 通义千问（阿里云）
- OpenAI GPT
- 智谱 GLM
- 其他兼容 OpenAI API 的模型

## 工作流程

### 1. 文档上传流程
1. 用户在 `documents.html` 上传文档
2. 后端接收文件并保存到 `docs/` 目录
3. 根据文件类型解析文档内容
4. 将文档信息保存到内存和 `documents_index.json`
5. 自动重建向量索引

### 2. 向量索引构建流程
1. 加载所有已上传文档内容
2. 对文档进行分块（500字符/块，重叠20字符）
3. 使用 M3E 模型生成文本向量
4. 构建 FAISS 索引
5. 保存索引文件（`m3e_faiss_index.bin`）
6. 保存映射关系（`chunks_mapping.npy`）

### 3. 问答流程
1. 用户在 `chat.html` 输入问题
2. 后端使用问题向量检索相关文档块
3. 构建上下文（包含检索到的文档片段）
4. 调用 LLM API（流式响应）
5. 使用 SSE 实时推送响应内容
6. 保存会话到 SQLite 数据库

### 4. 会话管理流程
1. 新会话：生成 UUID，保存问题和回答
2. 继续会话：使用现有 session_id，追加新消息
3. 会话列表：从数据库查询最近10条会话
4. 导出会话：将消息格式化为 Markdown 下载

## 部署说明

### 本地开发
```bash
cd app
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 部署
```bash
# 构建镜像
docker build -t project-rag-app:latest ./app

# 启动服务
docker-compose up -d
```

### 环境配置
1. 复制 `.env` 文件并配置模型 API
2. 确保 `docs/` 目录存在
3. 首次启动会自动下载 M3E 模型

## 注意事项

1. **模型配置**: 必须在 `.env` 中配置有效的模型 API
2. **文档格式**: 仅支持 PDF、TXT、DOCX 格式
3. **联网搜索**: 需配置 Google Custom Search API
4. **语音功能**: 依赖浏览器 Web Speech API
5. **Docker 兼容**: 已针对 Windows 和 Linux 优化
6. **索引重建**: 大量文档时首次启动可能较慢
7. **数据库位置**: SQLite 数据库存储在应用目录
8. **时区设置**: 默认使用上海时区

## 技术亮点

1. **本地向量模型**: 使用 M3E 中文嵌入模型，无需联网
2. **流式响应**: SSE 实时推送，提升用户体验
3. **持久化存储**: 文档、索引、聊天记录全部持久化
4. **跨平台兼容**: Docker 部署支持 Windows 和 Linux
5. **多格式支持**: 支持多种文档格式解析
6. **语音交互**: 支持语音输入和输出
7. **会话管理**: 完整的 CRUD 操作和导出功能
8. **Markdown 渲染**: 美观的富文本显示和代码高亮

## 扩展建议

1. 支持更多文档格式（HTML、CSV、JSON等）
2. 添加用户认证和权限管理
3. 支持多租户（不同用户隔离数据）
4. 添加向量数据库（如 Milvus、Weaviate）
5. 支持嵌入模型在线更新
6. 添加对话评价和反馈机制
7. 支持多语言
8. 添加文档预览功能
