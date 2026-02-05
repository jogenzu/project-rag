# 使用官方 Python 镜像
#FROM python:3.10-slim
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 更新 pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

# 复制并安装依赖
COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt 

#RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
#    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 复制整个 app 目录（包含子目录和静态文件）
COPY . .

#
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000",\
    "--log-config","/app/logging.ini"]
