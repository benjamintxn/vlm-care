# Dockerfile

FROM python:3.11-slim

# 1) Install system packages for building Detectron2
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libomp-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Set working directory & copy requirements
WORKDIR /app
COPY requirements.txt .

# 3) Install other Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 4) Env vars for gRPC / Apple Silicon build
ENV GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 \
    GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 \
    CC=clang \
    CXX=clang++ \
    ARCHFLAGS="-arch arm64"

# 5) Clone & install Detectron2 from source
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2_src \
 && pip install --no-cache-dir -e /detectron2_src \
 && rm -rf /detectron2_src

# 6) Copy the rest of your code
COPY . .

CMD ["bash"]
