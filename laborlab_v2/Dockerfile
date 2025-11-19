# GPU 모드용 베이스 이미지
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS gpu-base

# Python 3.11 설치 (deadsnakes PPA 사용)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pip 설치 및 python3.11을 기본 python으로 설정
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1

# CPU/Mac 모드용 베이스 이미지
FROM python:3.11-slim AS cpu-base

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# GPU 모드 최종 스테이지
FROM gpu-base AS final-gpu
ARG DEVICE_MODE=gpu

WORKDIR /app
COPY . /app/

# CUDA 지원 PyTorch 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r /app/laborlab_v2/requirements.txt && \
    pip install --no-cache-dir -e /app

ENV PYTHONPATH=/app
ENV TERMINAL_OUTPUT_DIR=/app/laborlab_v2/log
ENV DEVICE_MODE=gpu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app/laborlab_v2

# CPU/Mac 모드 최종 스테이지
FROM cpu-base AS final-cpu
ARG DEVICE_MODE=cpu

WORKDIR /app
COPY . /app/

# 일반 PyTorch 설치 (CPU 버전)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/laborlab_v2/requirements.txt && \
    pip install --no-cache-dir -e /app

ENV PYTHONPATH=/app
ENV TERMINAL_OUTPUT_DIR=/app/laborlab_v2/log
ENV DEVICE_MODE=${DEVICE_MODE}

WORKDIR /app/laborlab_v2
