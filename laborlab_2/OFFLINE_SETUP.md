# 폐쇠망(오프라인) 환경 구축 가이드

## ⚠️ 현재 상태 분석

**현재 설정은 오프라인 환경에서 정상적으로 작동하지 않습니다.**

다음 부분들이 인터넷 연결을 필요로 합니다:
1. `Dockerfile`의 `apt-get update` (시스템 패키지 다운로드)
2. `Dockerfile`의 `pip install` (Python 패키지 다운로드)
3. `docker-compose.yml`의 `ollama/ollama:latest` 이미지 다운로드
4. 베이스 이미지 `nvidia/cuda:12.4.0-runtime-ubuntu22.04` 다운로드

## 📦 온라인 환경에서 미리 다운로드해야 할 파일들

### 1. Docker 이미지

#### 1.1 베이스 이미지 다운로드
```bash
# CUDA 베이스 이미지 다운로드
docker pull nvidia/cuda:12.4.0-runtime-ubuntu22.04
docker save nvidia/cuda:12.4.0-runtime-ubuntu22.04 -o nvidia-cuda-12.4.0-runtime-ubuntu22.04.tar

# 이미지 크기: 약 2-3GB
```

#### 1.2 Ollama 이미지 다운로드
```bash
# Ollama 이미지 다운로드
docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o ollama-latest.tar

# 이미지 크기: 약 1-2GB
```

#### 1.3 빌드된 애플리케이션 이미지 저장 (선택사항)
```bash
# 온라인 환경에서 빌드 후 저장
docker-compose build
docker save laborlab-2-analysis:latest -o laborlab-2-image.tar

# 또는 docker-compose로 빌드된 이미지 확인
docker images | grep laborlab
```

### 2. Python 패키지 (Wheel 파일)

#### 2.1 패키지 다운로드 스크립트
온라인 환경에서 다음 스크립트를 실행하여 모든 패키지를 다운로드:

```bash
# requirements.txt의 모든 패키지 다운로드
pip download -r requirements.txt -d ./offline_packages --platform linux_x86_64 --only-binary :all:

# 또는 소스 배포본도 포함하려면
pip download -r requirements.txt -d ./offline_packages

# TabPFN 특별 처리 (GitHub에서 직접 다운로드 필요)
# TabPFN은 requirements.txt에 주석 처리되어 있으므로 별도 처리 필요
```

#### 2.2 필요한 Python 패키지 목록
`requirements.txt`에 명시된 패키지들:
- numpy>=2.0.0
- pandas>=2.0.0
- scikit-learn>=1.0.0
- scipy>=1.10.0
- statsmodels>=0.14.0
- networkx>=3.3.0
- sympy>=1.10.1
- joblib>=1.1.0
- tqdm>=4.64.0
- causal-learn>=0.1.3.0
- econml>=0.16.0
- numba>=0.59.0
- torch>=2.0.0 (PyTorch - 매우 큼, 약 1-2GB)
- tabpfn>=0.1.0
- matplotlib>=3.5.3
- pydot>=1.4.2
- python-dateutil>=2.8.0
- openpyxl>=3.1.0
- openai>=1.0.0
- ollama>=0.1.0

**예상 총 크기: 약 5-10GB** (PyTorch 포함)

#### 2.3 TabPFN 특별 처리
TabPFN은 GitHub에서 직접 설치해야 할 수 있습니다:
```bash
# TabPFN 소스 코드 다운로드
git clone https://github.com/PriorLabs/TabPFN.git
cd TabPFN
git checkout 86bad3f492d72d849c583d57f0ddda8ea3216ed0
cd ..
tar -czf TabPFN-source.tar.gz TabPFN/
```

### 3. APT 패키지 (Ubuntu 22.04)

#### 3.1 필요한 시스템 패키지 목록
Dockerfile에서 설치하는 패키지들:
- python3.11
- python3.11-dev
- python3-pip
- gcc
- g++
- make
- git
- curl

#### 3.2 APT 패키지 다운로드
```bash
# Ubuntu 22.04 환경에서 실행
mkdir -p ./offline_apt_packages
cd ./offline_apt_packages

# 패키지 다운로드 (의존성 포함)
apt-get download python3.11 python3.11-dev python3-pip gcc g++ make git curl

# 모든 의존성 다운로드
apt-get install --download-only python3.11 python3.11-dev python3-pip gcc g++ make git curl

# 또는 apt-offline 사용 (더 효율적)
apt-offline set offline_packages.sig --install-packages python3.11 python3.11-dev python3-pip gcc g++ make git curl
apt-offline get offline_packages.sig --bundle offline_packages.zip
```

### 4. Ollama 모델 파일 (선택사항)

LLM 기능을 사용하는 경우:
```bash
# Ollama 모델 다운로드 (온라인 환경에서)
# 예: llama2, mistral 등
# 이는 컨테이너 실행 후 ollama pull 명령으로 다운로드 가능
# 또는 ./ollama_models 디렉토리에 미리 다운로드
```

## 🚀 오프라인 환경에서의 설정 방법

### 방법 1: Docker 이미지 미리 빌드 (권장)

#### 1.1 온라인 환경에서
```bash
# 1. 모든 파일 준비
mkdir -p offline_resources
cd offline_resources

# 2. Docker 이미지 저장
docker pull nvidia/cuda:12.4.0-runtime-ubuntu22.04
docker save nvidia/cuda:12.4.0-runtime-ubuntu22.04 -o nvidia-cuda.tar

docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o ollama.tar

# 3. Python 패키지 다운로드
pip download -r ../laborlab_2/requirements.txt -d ./python_packages --platform linux_x86_64

# 4. APT 패키지 다운로드 (Ubuntu 22.04)
apt-get download python3.11 python3.11-dev python3-pip gcc g++ make git curl
# 또는 apt-offline 사용

# 5. 전체를 압축
tar -czf offline_resources.tar.gz .
```

#### 1.2 오프라인 환경에서
```bash
# 1. 리소스 압축 해제
tar -xzf offline_resources.tar.gz

# 2. Docker 이미지 로드
docker load < nvidia-cuda.tar
docker load < ollama.tar

# 3. Dockerfile 수정 필요 (아래 참조)
# 4. docker-compose build 실행
```

### 방법 2: Dockerfile 수정 (오프라인 대응)

Dockerfile을 다음과 같이 수정해야 합니다:

```dockerfile
# CUDA 12.4 기반 이미지 사용 (Ubuntu 22.04)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# 로컬 APT 패키지 복사 및 설치
COPY offline_apt_packages/*.deb /tmp/apt_packages/
RUN dpkg -i /tmp/apt_packages/*.deb || true && \
    apt-get update --allow-insecure-repositories && \
    apt-get install -f -y --allow-unauthenticated && \
    rm -rf /var/lib/apt/lists/*

# python3.11을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 루트로 복사
COPY . /app/

# 로컬 Python 패키지 복사
COPY offline_packages /tmp/pip_packages

# Python 의존성 설치 (로컬 패키지 사용)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --find-links /tmp/pip_packages --no-index -r /app/laborlab_2/requirements.txt && \
    pip install --no-cache-dir -e /app

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV TERMINAL_OUTPUT_DIR=/app/laborlab_2/log

# 작업 디렉토리 설정
WORKDIR /app/laborlab_2

# 기본 명령어: main.py 실행
CMD ["python", "-m", "src.main", "--config", "config.json"]
```

### 방법 3: docker-compose.yml 수정

```yaml
  ollama:
    container_name: ollama
    # image 대신 build 사용하거나, 미리 로드된 이미지 사용
    image: ollama/ollama:latest  # 이미 docker load로 로드된 이미지 사용
    # 또는
    # build:
    #   context: ./ollama_build
    #   dockerfile: Dockerfile.ollama
```

## 📋 체크리스트

### 온라인 환경에서 준비할 항목:

- [ ] `nvidia/cuda:12.4.0-runtime-ubuntu22.04` Docker 이미지 (tar 파일)
- [ ] `ollama/ollama:latest` Docker 이미지 (tar 파일)
- [ ] Python 패키지 wheel 파일들 (requirements.txt 기반)
- [ ] APT 패키지 deb 파일들 (Ubuntu 22.04)
- [ ] TabPFN 소스 코드 (필요한 경우)
- [ ] Ollama 모델 파일들 (필요한 경우)

### 오프라인 환경에서 수행할 작업:

- [ ] Docker 이미지 로드 (`docker load`)
- [ ] Dockerfile 수정 (로컬 패키지 경로 지정)
- [ ] docker-compose.yml 확인
- [ ] 빌드 테스트 (`docker-compose build`)
- [ ] 실행 테스트 (`docker-compose up`)

## 🔍 검증 방법

오프라인 환경에서 다음 명령으로 검증:

```bash
# 1. 네트워크 연결 차단 확인
ping 8.8.8.8  # 실패해야 함

# 2. Docker 이미지 로드
docker load < nvidia-cuda.tar
docker load < ollama.tar

# 3. 이미지 확인
docker images

# 4. 빌드 테스트
docker-compose build --no-cache

# 5. 실행 테스트
docker-compose up
```

## ⚠️ 주의사항

1. **플랫폼 호환성**: Python wheel 파일은 Linux x86_64용으로 다운로드해야 합니다.
2. **의존성 해결**: 일부 패키지는 복잡한 의존성을 가지므로 모든 의존성을 포함해야 합니다.
3. **CUDA 버전**: 호스트 시스템의 CUDA 버전과 Docker 이미지의 CUDA 버전이 호환되어야 합니다.
4. **디스크 공간**: 전체 리소스는 약 15-20GB 정도 필요할 수 있습니다.

