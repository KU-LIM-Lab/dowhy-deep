#!/bin/bash
# 폐쇠망 배포를 위한 Docker 이미지 저장 스크립트
# 온라인 환경에서 실행하여 Docker 이미지를 저장합니다.
# 소스 코드 및 DoWhy 라이브러리는 수기로 전송합니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Docker 이미지 저장 (폐쇠망 배포용)"
echo "=========================================="
echo ""

cd "${PROJECT_DIR}"

# 0. Docker 연결 확인
echo "[0] Docker 연결 확인 중..."
if ! docker info > /dev/null 2>&1; then
    echo "    ⚠️ Docker에 연결할 수 없습니다."
    exit 1
fi
echo "    ✓ Docker 연결 확인 완료"

# 1. Laborlab Docker 이미지 빌드 및 저장
echo ""
echo "[1] Laborlab Docker 이미지 빌드 및 저장 중..."
echo "  - Docker 이미지 빌드 중..."
if docker-compose build; then
    echo "    ✓ Docker 이미지 빌드 완료"
    
    # 빌드된 이미지 확인
    IMAGE_NAME=$(docker images | grep "laborlab_2-laborlab" | head -1 | awk '{print $1":"$2}')
    if [ -z "$IMAGE_NAME" ]; then
        IMAGE_NAME=$(docker images | grep "laborlab" | head -1 | awk '{print $1":"$2}')
    fi
    
    if [ -n "$IMAGE_NAME" ]; then
        # Python 버전 확인
        echo "  - Python 버전 확인 중..."
        PYTHON_VERSION=$(docker run --rm "${IMAGE_NAME}" python --version 2>&1 | grep -o "Python [0-9.]*" || echo "확인 실패")
        PYTHON3_VERSION=$(docker run --rm "${IMAGE_NAME}" python3 --version 2>&1 | grep -o "Python [0-9.]*" || echo "확인 실패")
        echo "    python 버전: ${PYTHON_VERSION}"
        echo "    python3 버전: ${PYTHON3_VERSION}"
        
        # pandas 설치 확인
        echo "  - pandas 설치 확인 중..."
        if docker run --rm "${IMAGE_NAME}" python -c "import pandas; print('pandas OK:', pandas.__version__)" 2>&1 | grep -q "pandas OK"; then
            echo "    ✓ pandas가 정상적으로 설치되어 있습니다."
        else
            echo "    ⚠️ pandas 설치 확인 실패 - 이미지를 다시 빌드하세요."
        fi
        
        echo "  - 이미지 저장 중: ${IMAGE_NAME}"
        docker save "${IMAGE_NAME}" -o "${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar"
        echo "    ✓ Laborlab 이미지 저장 완료: laborlab-2-image_${TIMESTAMP}.tar"
    else
        echo "    ⚠️ 이미지 이름을 찾을 수 없습니다. 수동으로 저장하세요:"
        echo "       docker images"
        echo "       docker save <IMAGE_NAME> -o laborlab-2-image.tar"
        exit 1
    fi
else
    echo "    ⚠️ Docker 이미지 빌드 실패"
    echo "       수동으로 빌드 및 저장하세요:"
    echo "       docker-compose build"
    echo "       docker save laborlab_2-laborlab:latest -o laborlab-2-image.tar"
    exit 1
fi

# 2. Ollama 이미지 저장
echo ""
echo "[2] Ollama 이미지 저장 중..."
if docker images | grep -q "ollama/ollama"; then
    echo "  - 로컬 Ollama 이미지 저장 중..."
    docker save ollama/ollama:latest -o "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar"
    echo "    ✓ Ollama 이미지 저장 완료: ollama_${TIMESTAMP}.tar"
else
    echo "  - Ollama 이미지 다운로드 중..."
    if docker pull ollama/ollama:latest; then
        docker save ollama/ollama:latest -o "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar"
        echo "    ✓ Ollama 이미지 저장 완료: ollama_${TIMESTAMP}.tar"
    else
        echo "    ⚠️ Ollama 이미지 다운로드 실패"
        exit 1
    fi
fi

# 3. 요약
echo ""
echo "=========================================="
echo "이미지 저장 완료!"
echo "=========================================="
echo ""
echo "생성된 파일:"
if [ -f "${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar" ]; then
    echo "  - ${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar"
fi
if [ -f "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar" ]; then
    echo "  - ${PROJECT_DIR}/ollama_${TIMESTAMP}.tar"
fi
echo ""
echo "폐쇠망 환경으로 전송할 파일:"
if [ -f "${PROJECT_DIR}/laborlab-2-image_${TIMESTAMP}.tar" ]; then
    echo "  1. laborlab-2-image_${TIMESTAMP}.tar"
fi
if [ -f "${PROJECT_DIR}/ollama_${TIMESTAMP}.tar" ]; then
    echo "  2. ollama_${TIMESTAMP}.tar"
fi
echo ""
echo "폐쇠망 환경에서 실행:"
echo "  1. docker load < laborlab-2-image_${TIMESTAMP}.tar"
echo "  2. docker load < ollama_${TIMESTAMP}.tar"
echo "  3. 소스 코드 및 DoWhy 라이브러리를 수기로 전송한 후"
echo "  4. cd laborlab_2"
echo "  5. docker-compose up -d"
echo ""
