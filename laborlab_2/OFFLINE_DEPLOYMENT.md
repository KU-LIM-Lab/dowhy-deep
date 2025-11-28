# 폐쇠망 배포 가이드

## 📋 개요

이 가이드는 온라인 환경에서 필요한 리소스를 준비하고, 폐쇠망 환경으로 전송하여 실행하는 방법을 설명합니다.

## 🔄 단계별 가이드

### 1단계: 온라인 환경에서 준비 (현재 컴퓨터)

#### 1.1 프로젝트 파일 패키징

```bash
# laborlab_2 디렉토리로 이동
cd laborlab_2

# 프로젝트 전체를 압축 (데이터 제외하고 소스 코드만)
tar -czf laborlab_2_source.tar.gz \
    --exclude='data/seis_data' \
    --exclude='data/checkpoint' \
    --exclude='log' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    src/ config.json requirements.txt Dockerfile Dockerfile.offline docker-compose.yml README.md scripts/

# 또는 전체 프로젝트 (데이터 포함)
tar -czf laborlab_2_full.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='log' \
    --exclude='data/checkpoint' \
    .
```

#### 1.2 Python 패키지 다운로드 (선택사항 - Docker 사용 시)

```bash
# 수동으로 패키지 다운로드
mkdir -p offline_packages
pip download -r requirements.txt -d offline_packages --platform linux_x86_64
```

#### 1.3 Docker 이미지 준비 (Docker 사용 시)

```bash
# Docker 이미지 다운로드 및 저장
docker pull nvidia/cuda:12.4.0-runtime-ubuntu22.04
docker save nvidia/cuda:12.4.0-runtime-ubuntu22.04 -o nvidia-cuda.tar

docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o ollama.tar
```

### 2단계: 폐쇠망 환경으로 전송할 파일

#### 필수 파일 목록

1. **프로젝트 소스 코드**
   - `laborlab_2_source.tar.gz` 또는 `laborlab_2_full.tar.gz`
   - 또는 전체 `laborlab_2/` 폴더

2. **데이터 파일** (별도 전송 가능)
   - `data/seis_data/` - 모든 JSON 및 CSV 파일
   - `data/graph_data/` - 모든 .dot 파일
   - `data/variable_mapping.json`
   - `data/job_subcategories_*.csv`

3. **DoWhy 라이브러리** (프로젝트 루트에 필요)
   - `../dowhy/` 폴더 전체

4. **Docker 리소스** (Docker 사용 시)
   - `nvidia-cuda.tar`
   - `ollama.tar`
   - `offline_packages/` (Python wheel 파일들)

### 3단계: 폐쇠망 환경에서 설정

#### 3.1 파일 압축 해제 및 구조 확인

```bash
# 프로젝트 압축 해제
tar -xzf laborlab_2_full.tar.gz

# 또는 소스만 받은 경우
tar -xzf laborlab_2_source.tar.gz
# 데이터는 별도로 복사

# 디렉토리 구조 확인
cd laborlab_2
ls -la

# 필수 파일 확인 스크립트 실행
bash scripts/check_deployment.sh
```

#### 3.2 DoWhy 라이브러리 확인

```bash
# 프로젝트 루트로 이동 (laborlab_2의 상위 디렉토리)
cd ..

# DoWhy 라이브러리가 있는지 확인
ls -la dowhy/

# 없으면 dowhy 폴더를 프로젝트 루트에 복사
```

#### 3.3 Python 환경 설정 (Docker 없이 실행하는 경우)

```bash
# Python 3.11 이상 확인
python3 --version

# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 로컬 패키지 설치 (offline_packages가 있는 경우)
pip install --no-index --find-links ./offline_packages -r requirements.txt

# 또는 일반 설치 (패키지가 이미 설치된 경우)
pip install -r requirements.txt

# DoWhy 라이브러리 설치 (editable mode)
cd ..
pip install -e ./dowhy
cd laborlab_2
```

#### 3.4 Docker 환경 설정 (Docker 사용하는 경우)

```bash
# Docker 이미지 로드
docker load < nvidia-cuda.tar
docker load < ollama.tar

# 이미지 확인
docker images

# Dockerfile.offline 사용하여 빌드
cd laborlab_2
docker build -f Dockerfile.offline -t laborlab-2:offline ..

# 또는 docker-compose 사용
docker-compose -f docker-compose.yml build
```

### 4단계: 실행

#### 방법 1: Python 직접 실행 (권장 - 간단)

```bash
# laborlab_2 디렉토리에서
cd laborlab_2

# 프로젝트 루트에서 실행
cd ..
python -m laborlab_2.src.main --config laborlab_2/config.json
```

#### 방법 2: Docker Compose 실행

```bash
cd laborlab_2
docker-compose up
```

#### 방법 3: Docker 직접 실행

```bash
cd laborlab_2
docker run --gpus all \
    -v $(pwd)/data:/app/laborlab_2/data:ro \
    -v $(pwd)/log:/app/laborlab_2/log \
    -v $(pwd)/config.json:/app/laborlab_2/config.json:ro \
    laborlab-2:offline
```

## 📦 전송 패키지 구성 예시

### 최소 구성 (소스 코드만)

```
배포_패키지/
├── laborlab_2_source.tar.gz
├── data/
│   ├── seis_data/
│   │   ├── seis_data.csv
│   │   ├── resume.json
│   │   ├── coverletters.json
│   │   ├── trainings.json
│   │   └── licenses.json
│   ├── graph_data/
│   │   └── graph_*.dot (모든 그래프 파일)
│   ├── variable_mapping.json
│   └── job_subcategories_*.csv
└── dowhy/  (DoWhy 라이브러리)
```

### 완전 구성 (Docker 포함)

```
배포_패키지/
├── laborlab_2_full.tar.gz
├── nvidia-cuda.tar
├── ollama.tar
├── offline_packages/  (Python wheel 파일들)
└── dowhy/  (DoWhy 라이브러리)
```

## ✅ 폐쇠망 환경 체크리스트

### 전송 전 확인

- [ ] `scripts/check_deployment.sh` 실행하여 모든 파일 확인
- [ ] `config.json` 설정 확인
- [ ] 데이터 파일 크기 확인 (전송 가능한 크기인지)
- [ ] DoWhy 라이브러리 포함 여부 확인

### 폐쇠망 환경에서 확인

- [ ] Python 3.11 이상 설치 확인
- [ ] Docker 설치 확인 (Docker 사용 시)
- [ ] GPU 드라이버 확인 (GPU 사용 시)
- [ ] 디스크 공간 확인 (최소 20GB 이상 권장)
- [ ] 파일 권한 확인

## 🚨 문제 해결

### Python 패키지 설치 실패

```bash
# 로컬 패키지 우선 사용
pip install --no-index --find-links ./offline_packages -r requirements.txt

# 특정 패키지만 설치
pip install --no-index --find-links ./offline_packages 패키지명
```

### DoWhy 라이브러리 오류

```bash
# 프로젝트 루트에서 DoWhy 설치 확인
cd ..
ls -la dowhy/
pip install -e ./dowhy
```

### Docker 이미지 로드 실패

```bash
# 이미지 확인
docker images

# 수동으로 이미지 로드
docker load < nvidia-cuda.tar
docker load < ollama.tar
```

## 📝 빠른 참조 명령어

### 온라인 환경에서

```bash
# 1. 프로젝트 패키징
cd laborlab_2
tar -czf ../laborlab_2_deploy.tar.gz --exclude='log' --exclude='data/checkpoint' --exclude='.git' .

# 2. 배포 확인
bash scripts/check_deployment.sh
```

### 폐쇠망 환경에서

```bash
# 1. 압축 해제
tar -xzf laborlab_2_deploy.tar.gz

# 2. DoWhy 확인
cd ..
ls dowhy/

# 3. 실행
cd laborlab_2
cd ..
python -m laborlab_2.src.main --config laborlab_2/config.json
```

## 💡 팁

1. **데이터 크기가 큰 경우**: 데이터는 별도로 전송하고, 소스 코드만 먼저 전송하여 테스트
2. **네트워크 제한**: USB나 외장 하드디스크로 전송
3. **권한 문제**: 폐쇠망 환경에서 실행 권한 확인
4. **로그 확인**: 실행 중 문제가 있으면 `log/` 폴더의 로그 파일 확인

