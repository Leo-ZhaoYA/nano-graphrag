# 使用官方的基础镜像
FROM python:3.11.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录的内容到 /app
COPY . /app

# 更新系统并安装编译工具
RUN apt-get update && apt-get install -y g++ libstdc++6 nodejs npm vim nano curl procps net-tools htop iputils-ping sudo telnet lsof grep wget  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# 更新 pip
RUN pip install --upgrade pip

# 安装依赖
RUN pip install --no-cache-dir -r docker_requirements.txt

# 安装 Python 和 Node.js 包
# 注意这里npm install不能-g，会找不到jsonrepair
RUN cd /app && \
    pip install --no-input pythonmonkey && \
    npm install --yes jsonrepair

# 定义环境变量
ENV FLASK_APP=server.py

# 使端口可供此容器外的环境使用
EXPOSE 8004

# 在容器启动时运行 Python 应用程序
CMD ["python", "-u", "server.py"]