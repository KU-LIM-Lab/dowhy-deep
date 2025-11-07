# Python 3.11 기반 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 루트로 복사
COPY . /app/

# Python 의존성 설치 (순서 중요)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/laborlab/requirements.txt && \
    pip install --no-cache-dir -e /app

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV TERMINAL_OUTPUT_DIR=/app/laborlab/log

# 작업 디렉토리 설정
WORKDIR /app/laborlab

# 기본 명령어는 docker-compose.yml에서 오버라이드
# 직접 실행 시: docker-compose run --rm causal-analysis python run_batch_experiments.py
# 또는: docker-compose exec causal-analysis python run_batch_experiments.py
CMD ["sleep", "infinity"]

