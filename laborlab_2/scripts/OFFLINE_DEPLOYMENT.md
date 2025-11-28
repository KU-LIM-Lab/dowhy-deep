# 폐쇠망 배포 가이드 (Docker 전용)

## 📋 개요

이 가이드는 **Docker를 사용하여** 온라인 환경에서 필요한 리소스를 준비하고, 폐쇠망 환경으로 전송하여 실행하는 방법을 설명합니다.

**중요**: 이 배포 방식은 Docker를 필수로 사용합니다.

## 🔄 단계별 가이드

### 1단계: 온라인 환경에서 준비 (현재 컴퓨터)

#### 1.1 프로젝트 파일 패키징

```bash
# laborlab_2 디렉토리로 이동
cd laborlab_2

# 자동 스크립트 사용 (권장)
bash scripts/package_for_offline.sh

# 또는 수동으로
tar -czf ../laborlab_2_deploy.tar.gz \
    --exclude='log' \
    --exclude='data/checkpoint' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    .
```

#### 1.2 Docker 이미지 빌드 및 저장

```bash
# 프로젝트 루트에서 Docker 이미지 빌드
cd laborlab_2
docker-compose build

# 빌드된 이미지 확인
docker images | grep laborlab

# 애플리케이션 이미지 저장
docker save laborlab_2-laborlab:latest -o laborlab-2-image.tar

# 베이스 이미지들 저장 (필요한 경우)
docker pull nvidia/cuda:12.4.0-runtime-ubuntu22.04
docker save nvidia/cuda:12.4.0-runtime-ubuntu22.04 -o nvidia-cuda.tar

docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o ollama.tar
```

#### 1.3 DoWhy 라이브러리 패키징

```bash
# 프로젝트 루트로 이동 (laborlab_2의 상위 디렉토리)
cd ..

# DoWhy 라이브러리 압축
tar -czf dowhy_library.tar.gz dowhy/
```

### 2단계: 폐쇠망 환경으로 전송할 파일

#### 필수 파일 목록

1. **프로젝트 소스 코드**
   - `laborlab_2_deploy.tar.gz`

2. **데이터 파일** (별도 전송 가능)
   - `data/seis_data/` - 모든 JSON 및 CSV 파일
   - `data/graph_data/` - 모든 .dot 파일
   - `data/variable_mapping.json`
   - `data/job_subcategories_*.csv`

3. **DoWhy 라이브러리**
   - `dowhy_library.tar.gz` 또는 `dowhy/` 폴더

4. **Docker 이미지** (필수)
   - `laborlab-2-image.tar` - 빌드된 애플리케이션 이미지
   - `ollama.tar` - Ollama 이미지
   - `nvidia-cuda.tar` - CUDA 베이스 이미지 (이미지에 포함되지 않은 경우)

### 3단계: 폐쇠망 환경에서 설정

#### 3.1 파일 압축 해제

```bash
# 프로젝트 압축 해제
tar -xzf laborlab_2_deploy.tar.gz

# DoWhy 라이브러리 압축 해제
tar -xzf dowhy_library.tar.gz

# 디렉토리 구조 확인 (다음과 같이 되어야 함)
# dowhy-deep/
#   ├── dowhy/          (DoWhy 라이브러리)
#   └── laborlab_2/     (프로젝트)
```

#### 3.2 배포 파일 확인

```bash
cd laborlab_2
bash scripts/check_deployment.sh
```

#### 3.3 Docker 이미지 로드

```bash
# Docker 이미지 로드
docker load < laborlab-2-image.tar
docker load < ollama.tar

# nvidia-cuda 이미지가 필요한 경우
docker load < nvidia-cuda.tar

# 이미지 확인
docker images
```

예상 출력:
```
REPOSITORY              TAG       IMAGE ID       CREATED         SIZE
laborlab_2-laborlab     latest    ...            ...             ...
ollama/ollama           latest    ...            ...             ...
nvidia/cuda             12.4.0-runtime-ubuntu22.04 ... ... ...
```

#### 3.4 Docker 환경 확인

```bash
# Docker 설치 확인
docker --version
docker-compose --version

# NVIDIA Docker 확인 (GPU 사용 시)
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi
```

### 4단계: 실행

#### Docker Compose로 실행 (권장)

```bash
cd laborlab_2

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f laborlab

# 실행 상태 확인
docker-compose ps
```

#### 실행 중 모니터링

```bash
# 실시간 로그 확인
docker-compose logs -f laborlab

# 컨테이너 상태 확인
docker-compose ps

# 컨테이너 내부 접속 (필요시)
docker-compose exec laborlab bash
```

### 5단계: 실행 완료 후

#### 결과 확인

```bash
# CSV 결과 파일 확인
ls -lh laborlab_2/log/*.csv

# 로그 파일 확인
ls -lh laborlab_2/log/*.log

# Checkpoint 파일 확인
ls -lh laborlab_2/data/checkpoint/
```

#### 컨테이너 관리

```bash
# 컨테이너 중지
docker-compose stop

# 컨테이너 중지 및 삭제
docker-compose down

# 컨테이너 재시작
docker-compose restart laborlab

# 컨테이너 재빌드 및 재시작
docker-compose up --build -d
```

## 📦 전송 패키지 구성 예시

### 완전 구성 (Docker 포함)

```
배포_패키지/
├── laborlab_2_deploy.tar.gz          (프로젝트 소스 코드)
├── dowhy_library.tar.gz               (DoWhy 라이브러리)
├── laborlab-2-image.tar               (빌드된 애플리케이션 이미지)
├── ollama.tar                          (Ollama 이미지)
└── nvidia-cuda.tar                     (CUDA 베이스 이미지, 선택사항)
```

## ✅ 폐쇠망 환경 체크리스트

### 전송 전 확인 (온라인 환경)

- [ ] `scripts/package_for_offline.sh` 실행하여 프로젝트 패키징
- [ ] `docker-compose build` 실행하여 이미지 빌드
- [ ] `docker save`로 이미지 저장
- [ ] `scripts/check_deployment.sh` 실행하여 모든 파일 확인
- [ ] `config.json` 설정 확인
- [ ] 데이터 파일 크기 확인 (전송 가능한 크기인지)
- [ ] DoWhy 라이브러리 포함 여부 확인

### 폐쇠망 환경에서 확인

- [ ] Docker 설치 확인 (`docker --version`)
- [ ] Docker Compose 설치 확인 (`docker-compose --version`)
- [ ] NVIDIA Docker 설치 확인 (GPU 사용 시)
- [ ] GPU 드라이버 확인 (GPU 사용 시)
- [ ] 디스크 공간 확인 (최소 30GB 이상 권장)
- [ ] 파일 권한 확인
- [ ] Docker 이미지 로드 확인 (`docker images`)

## 🚨 문제 해결

### Docker 이미지 로드 실패

```bash
# 이미지 확인
docker images

# 수동으로 이미지 로드
docker load < laborlab-2-image.tar
docker load < ollama.tar

# 이미지 태그 확인 및 수정 (필요시)
docker tag <IMAGE_ID> laborlab_2-laborlab:latest
```

### Docker Compose 실행 실패

```bash
# 로그 확인
docker-compose logs laborlab

# 컨테이너 재빌드
docker-compose build --no-cache

# 컨테이너 재시작
docker-compose restart laborlab
```

### GPU 인식 실패

```bash
# NVIDIA Docker 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi

# docker-compose.yml에서 GPU 설정 확인
cat docker-compose.yml | grep -A 5 "deploy:"
```

### DoWhy 라이브러리 오류

```bash
# 프로젝트 루트에서 DoWhy 확인
cd ..
ls -la dowhy/

# Docker 컨테이너 내부에서 확인
docker-compose exec laborlab ls -la /app/dowhy
```

## 📝 빠른 참조 명령어

### 온라인 환경에서

```bash
# 1. 프로젝트 패키징
cd laborlab_2
bash scripts/package_for_offline.sh

# 2. Docker 이미지 빌드
docker-compose build

# 3. Docker 이미지 저장
docker save laborlab_2-laborlab:latest -o laborlab-2-image.tar
docker save ollama/ollama:latest -o ollama.tar

# 4. DoWhy 라이브러리 압축
cd ..
tar -czf dowhy_library.tar.gz dowhy/
```

### 폐쇠망 환경에서

```bash
# 1. 압축 해제
tar -xzf laborlab_2_deploy.tar.gz
tar -xzf dowhy_library.tar.gz

# 2. Docker 이미지 로드
docker load < laborlab-2-image.tar
docker load < ollama.tar

# 3. 배포 확인
cd laborlab_2
bash scripts/check_deployment.sh

# 4. 실행
docker-compose up -d

# 5. 로그 확인
docker-compose logs -f laborlab
```

## 💡 팁

1. **이미지 크기**: Docker 이미지는 크기가 클 수 있으므로 (5-10GB), 전송 시간을 고려하세요.
2. **네트워크 제한**: USB나 외장 하드디스크로 전송하는 것이 안정적입니다.
3. **권한 문제**: 폐쇠망 환경에서 Docker 실행 권한 확인 (`sudo docker` 또는 `docker` 그룹 추가)
4. **로그 확인**: 실행 중 문제가 있으면 `docker-compose logs laborlab`로 확인
5. **디스크 공간**: Docker 이미지와 실행 중인 컨테이너를 위해 충분한 디스크 공간 확보 (최소 30GB 권장)

## 🔍 검증 방법

폐쇠망 환경에서 다음 명령으로 검증:

```bash
# 1. 네트워크 연결 차단 확인
ping 8.8.8.8  # 실패해야 함

# 2. Docker 이미지 확인
docker images

# 3. Docker Compose 실행 테스트
cd laborlab_2
docker-compose config  # 설정 파일 검증

# 4. 컨테이너 실행 테스트
docker-compose up -d
docker-compose ps
docker-compose logs laborlab
```
