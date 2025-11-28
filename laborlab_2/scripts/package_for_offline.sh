#!/bin/bash
# 폐쇠망 배포를 위한 패키징 스크립트 (Docker 전용)
# 온라인 환경에서 실행하여 배포 패키지를 생성합니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy_package"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "폐쇠망 배포 패키지 생성 (Docker 전용)"
echo "=========================================="
echo ""

# 배포 디렉토리 생성
mkdir -p "${DEPLOY_DIR}"
cd "${PROJECT_DIR}"

# 1. 소스 코드 및 설정 파일 복사
echo "[1] 소스 코드 및 설정 파일 복사 중..."
mkdir -p "${DEPLOY_DIR}/laborlab_2"

# 필수 파일 복사
cp -r src/ "${DEPLOY_DIR}/laborlab_2/"
cp config.json "${DEPLOY_DIR}/laborlab_2/"
cp requirements.txt "${DEPLOY_DIR}/laborlab_2/"
cp Dockerfile "${DEPLOY_DIR}/laborlab_2/"
cp docker-compose.yml "${DEPLOY_DIR}/laborlab_2/"
cp README.md "${DEPLOY_DIR}/laborlab_2/"
cp OFFLINE_DEPLOYMENT.md "${DEPLOY_DIR}/laborlab_2/" 2>/dev/null || true

# scripts 폴더 복사
cp -r scripts/ "${DEPLOY_DIR}/laborlab_2/" 2>/dev/null || true

echo "  ✓ 소스 코드 복사 완료"

# 2. 데이터 파일 복사 (선택사항)
echo ""
echo "[2] 데이터 파일 복사 중..."
read -p "  데이터 파일도 포함하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "${DEPLOY_DIR}/laborlab_2/data"
    
    # 필수 데이터 파일만 복사
    if [ -d "data/seis_data" ]; then
        cp -r data/seis_data "${DEPLOY_DIR}/laborlab_2/data/"
        echo "  ✓ seis_data 복사 완료"
    fi
    
    if [ -d "data/graph_data" ]; then
        cp -r data/graph_data "${DEPLOY_DIR}/laborlab_2/data/"
        echo "  ✓ graph_data 복사 완료"
    fi
    
    if [ -f "data/variable_mapping.json" ]; then
        cp data/variable_mapping.json "${DEPLOY_DIR}/laborlab_2/data/"
        echo "  ✓ variable_mapping.json 복사 완료"
    fi
    
    if [ -f "data/job_subcategories_KECO.csv" ]; then
        cp data/job_subcategories_*.csv "${DEPLOY_DIR}/laborlab_2/data/" 2>/dev/null || true
        echo "  ✓ job_subcategories 파일 복사 완료"
    fi
else
    echo "  ⏭️ 데이터 파일 제외"
fi

# 3. 배포 확인 스크립트 실행
echo ""
echo "[3] 배포 파일 확인 중..."
cd "${DEPLOY_DIR}/laborlab_2"
if [ -f "scripts/check_deployment.sh" ]; then
    bash scripts/check_deployment.sh || echo "  ⚠️ 일부 파일이 누락되었을 수 있습니다."
fi

# 4. 압축 파일 생성
echo ""
echo "[4] 압축 파일 생성 중..."
cd "${PROJECT_DIR}"
tar -czf "laborlab_2_deploy_${TIMESTAMP}.tar.gz" -C deploy_package laborlab_2

# 5. DoWhy 라이브러리 확인
echo ""
echo "[5] DoWhy 라이브러리 확인..."
if [ -d "../dowhy" ]; then
    echo "  ✓ DoWhy 라이브러리 발견: ../dowhy"
    read -p "  DoWhy 라이브러리도 압축에 포함하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "${PROJECT_DIR}/.."
        tar -czf "dowhy_library_${TIMESTAMP}.tar.gz" dowhy/
        echo "  ✓ DoWhy 라이브러리 압축 완료: dowhy_library_${TIMESTAMP}.tar.gz"
    fi
else
    echo "  ⚠️ DoWhy 라이브러리를 찾을 수 없습니다."
    echo "     프로젝트 루트(../dowhy)에 DoWhy 라이브러리가 있어야 합니다."
fi

# 6. Docker 이미지 빌드 및 저장
echo ""
echo "[6] Docker 이미지 빌드 및 저장 중..."
cd "${PROJECT_DIR}"

# Docker Compose로 이미지 빌드
echo "  - Docker 이미지 빌드 중..."
if docker-compose build; then
    echo "    ✓ Docker 이미지 빌드 완료"
    
    # 빌드된 이미지 확인
    IMAGE_NAME=$(docker images | grep "laborlab_2-laborlab" | head -1 | awk '{print $1":"$2}')
    if [ -z "$IMAGE_NAME" ]; then
        IMAGE_NAME=$(docker images | grep "laborlab" | head -1 | awk '{print $1":"$2}')
    fi
    
    if [ -n "$IMAGE_NAME" ]; then
        echo "  - 이미지 저장 중: ${IMAGE_NAME}"
        docker save "${IMAGE_NAME}" -o "${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar"
        echo "    ✓ 애플리케이션 이미지 저장 완료: laborlab-2-image_${TIMESTAMP}.tar"
    else
        echo "    ⚠️ 이미지 이름을 찾을 수 없습니다. 수동으로 저장하세요:"
        echo "       docker images"
        echo "       docker save <IMAGE_NAME> -o laborlab-2-image.tar"
    fi
else
    echo "    ⚠️ Docker 이미지 빌드 실패"
    echo "       수동으로 빌드 및 저장하세요:"
    echo "       docker-compose build"
    echo "       docker save laborlab_2-laborlab:latest -o laborlab-2-image.tar"
fi

# Ollama 이미지 저장
echo ""
read -p "  Ollama 이미지를 저장하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if docker images | grep -q "ollama/ollama"; then
        docker save ollama/ollama:latest -o "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar"
        echo "    ✓ Ollama 이미지 저장 완료: ollama_${TIMESTAMP}.tar"
    else
        echo "    - Ollama 이미지 다운로드 중..."
        if docker pull ollama/ollama:latest; then
            docker save ollama/ollama:latest -o "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar"
            echo "    ✓ Ollama 이미지 저장 완료: ollama_${TIMESTAMP}.tar"
        else
            echo "    ⚠️ Ollama 이미지 다운로드 실패"
        fi
    fi
fi

# CUDA 베이스 이미지 저장 (선택사항)
echo ""
read -p "  CUDA 베이스 이미지를 저장하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if docker images | grep -q "nvidia/cuda.*12.4.0"; then
        CUDA_IMAGE=$(docker images | grep "nvidia/cuda.*12.4.0.*runtime" | head -1 | awk '{print $1":"$2}')
        if [ -n "$CUDA_IMAGE" ]; then
            docker save "${CUDA_IMAGE}" -o "${PROJECT_DIR}/nvidia-cuda_${TIMESTAMP}.tar"
            echo "    ✓ CUDA 이미지 저장 완료: nvidia-cuda_${TIMESTAMP}.tar"
        fi
    else
        echo "    - CUDA 이미지 다운로드 중..."
        if docker pull nvidia/cuda:12.4.0-runtime-ubuntu22.04; then
            docker save nvidia/cuda:12.4.0-runtime-ubuntu22.04 -o "${PROJECT_DIR}/nvidia-cuda_${TIMESTAMP}.tar"
            echo "    ✓ CUDA 이미지 저장 완료: nvidia-cuda_${TIMESTAMP}.tar"
        else
            echo "    ⚠️ CUDA 이미지 다운로드 실패"
        fi
    fi
fi

# 7. 요약
echo ""
echo "=========================================="
echo "패키징 완료!"
echo "=========================================="
echo ""
echo "생성된 파일:"
echo "  - ${PROJECT_DIR}/laborlab_2_deploy_${TIMESTAMP}.tar.gz"
if [ -f "${PROJECT_DIR}/../dowhy_library_${TIMESTAMP}.tar.gz" ]; then
    echo "  - ${PROJECT_DIR}/../dowhy_library_${TIMESTAMP}.tar.gz"
fi
if [ -f "${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar" ]; then
    echo "  - ${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar"
fi
if [ -f "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar" ]; then
    echo "  - ${PROJECT_DIR}/ollama_${TIMESTAMP}.tar"
fi
if [ -f "${PROJECT_DIR}/nvidia-cuda_${TIMESTAMP}.tar" ]; then
    echo "  - ${PROJECT_DIR}/nvidia-cuda_${TIMESTAMP}.tar"
fi
echo ""
echo "폐쇠망 환경으로 전송할 파일:"
echo "  1. laborlab_2_deploy_${TIMESTAMP}.tar.gz"
if [ -f "${PROJECT_DIR}/../dowhy_library_${TIMESTAMP}.tar.gz" ]; then
    echo "  2. dowhy_library_${TIMESTAMP}.tar.gz"
else
    echo "  2. dowhy/ 폴더 (별도 전송 필요)"
fi
if [ -f "${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar" ]; then
    echo "  3. laborlab-2-image_${TIMESTAMP}.tar"
fi
if [ -f "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar" ]; then
    echo "  4. ollama_${TIMESTAMP}.tar"
fi
if [ -f "${PROJECT_DIR}/nvidia-cuda_${TIMESTAMP}.tar" ]; then
    echo "  5. nvidia-cuda_${TIMESTAMP}.tar (선택사항)"
fi
echo ""
echo "폐쇠망 환경에서 실행:"
echo "  1. tar -xzf laborlab_2_deploy_${TIMESTAMP}.tar.gz"
echo "  2. tar -xzf dowhy_library_${TIMESTAMP}.tar.gz (DoWhy가 별도인 경우)"
echo "  3. docker load < laborlab-2-image_${TIMESTAMP}.tar"
echo "  4. docker load < ollama_${TIMESTAMP}.tar"
echo "  5. cd laborlab_2"
echo "  6. docker-compose up -d"
echo ""
