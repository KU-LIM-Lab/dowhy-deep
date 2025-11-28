# CUDA 12.4 기반 이미지 사용 (Ubuntu 22.04)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Python 3.11 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# python3.11을 기본 python 및 python3로 설정 (우선순위 높게)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10 && \
    update-alternatives --set python /usr/bin/python3.11 && \
    update-alternatives --set python3 /usr/bin/python3.11

# python3.11에 pip 설치 (get-pip.py 사용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# pip를 python3.11의 pip로 설정
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 10 2>/dev/null || \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip || true

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 루트로 복사
COPY . /app/

# Python 의존성 설치 (로컬 설치, https 요청 없음)
# requirements.txt는 로컬에 이미 설치된 패키지들을 사용
# python과 pip가 모두 python3.11을 가리키므로 직접 사용 가능
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/laborlab_2/requirements.txt && \
    pip install --no-cache-dir -e /app

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV TERMINAL_OUTPUT_DIR=/app/laborlab_2/log

# 작업 디렉토리 설정
WORKDIR /app/laborlab_2

# 기본 명령어: main.py 실행
CMD ["python", "-m", "src.main", "--config", "config.json"]

