#!/bin/bash
# 폐쇠망 환경을 위한 리소스 다운로드 스크립트
# 온라인 환경에서 실행하여 필요한 모든 파일을 다운로드합니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="${SCRIPT_DIR}/offline_resources"
PYTHON_PACKAGES_DIR="${OFFLINE_DIR}/python_packages"
APT_PACKAGES_DIR="${OFFLINE_DIR}/apt_packages"
DOCKER_IMAGES_DIR="${OFFLINE_DIR}/docker_images"

echo "=========================================="
echo "폐쇠망 환경 리소스 준비 스크립트"
echo "=========================================="

# 디렉토리 생성
mkdir -p "${PYTHON_PACKAGES_DIR}"
mkdir -p "${APT_PACKAGES_DIR}"
mkdir -p "${DOCKER_IMAGES_DIR}"

echo ""
echo "1. Docker 이미지 다운로드 중..."
echo "----------------------------------------"

# CUDA 베이스 이미지
echo "  - nvidia/cuda:12.4.0-runtime-ubuntu22.04 다운로드 중..."
docker pull nvidia/cuda:12.4.0-runtime-ubuntu22.04
docker save nvidia/cuda:12.4.0-runtime-ubuntu22.04 -o "${DOCKER_IMAGES_DIR}/nvidia-cuda-12.4.0-runtime-ubuntu22.04.tar"
echo "    ✓ 완료"

# Ollama 이미지
echo "  - ollama/ollama:latest 다운로드 중..."
docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o "${DOCKER_IMAGES_DIR}/ollama-latest.tar"
echo "    ✓ 완료"

echo ""
echo "2. Python 패키지 다운로드 중..."
echo "----------------------------------------"
echo "  이 작업은 시간이 오래 걸릴 수 있습니다 (5-10분)..."
echo "  예상 크기: 약 5-10GB"

# Python 패키지 다운로드
cd "${SCRIPT_DIR}"
pip download -r requirements.txt \
    -d "${PYTHON_PACKAGES_DIR}" \
    --platform linux_x86_64 \
    --only-binary :all: \
    --no-deps || {
    echo "  경고: 일부 패키지 다운로드 실패, 의존성 포함하여 재시도..."
    pip download -r requirements.txt \
        -d "${PYTHON_PACKAGES_DIR}" \
        --platform linux_x86_64
}

# 의존성도 함께 다운로드
echo "  - 의존성 패키지 다운로드 중..."
pip download -r requirements.txt \
    -d "${PYTHON_PACKAGES_DIR}" \
    --platform linux_x86_64

echo "    ✓ 완료"

echo ""
echo "3. APT 패키지 다운로드 중..."
echo "----------------------------------------"
echo "  주의: 이 단계는 Ubuntu 22.04 환경에서 실행해야 합니다."

# APT 패키지 다운로드
if command -v apt-get &> /dev/null; then
    cd "${APT_PACKAGES_DIR}"
    
    # 필요한 패키지 목록
    PACKAGES="python3.11 python3.11-dev python3-pip gcc g++ make git curl"
    
    echo "  - APT 패키지 다운로드 중..."
    apt-get download $(apt-cache depends --recurse --no-recommends --no-suggests \
        --no-conflicts --no-breaks --no-replaces --no-enhances ${PACKAGES} | \
        grep "^\w" | sort -u) 2>/dev/null || {
        echo "  경고: apt-get download 실패, 대체 방법 사용..."
        apt-get install --download-only -y ${PACKAGES} || true
    }
    
    echo "    ✓ 완료"
else
    echo "  경고: apt-get을 찾을 수 없습니다. Ubuntu 환경이 아닐 수 있습니다."
    echo "  수동으로 APT 패키지를 다운로드해야 합니다."
fi

echo ""
echo "4. TabPFN 소스 코드 다운로드 (선택사항)..."
echo "----------------------------------------"
read -p "  TabPFN 소스 코드를 다운로드하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v git &> /dev/null; then
        cd "${OFFLINE_DIR}"
        echo "  - TabPFN 저장소 클론 중..."
        git clone https://github.com/PriorLabs/TabPFN.git
        cd TabPFN
        git checkout 86bad3f492d72d849c583d57f0ddda8ea3216ed0
        cd ..
        tar -czf TabPFN-source.tar.gz TabPFN/
        rm -rf TabPFN/
        echo "    ✓ 완료"
    else
        echo "  경고: git을 찾을 수 없습니다. 수동으로 다운로드해야 합니다."
    fi
fi

echo ""
echo "5. 압축 파일 생성 중..."
echo "----------------------------------------"
cd "${SCRIPT_DIR}"
tar -czf offline_resources.tar.gz -C offline_resources .

echo ""
echo "=========================================="
echo "완료!"
echo "=========================================="
echo ""
echo "다운로드된 리소스:"
echo "  - Docker 이미지: ${DOCKER_IMAGES_DIR}/"
echo "  - Python 패키지: ${PYTHON_PACKAGES_DIR}/"
echo "  - APT 패키지: ${APT_PACKAGES_DIR}/"
echo "  - 압축 파일: ${SCRIPT_DIR}/offline_resources.tar.gz"
echo ""
echo "오프라인 환경으로 전송할 파일:"
echo "  - offline_resources.tar.gz"
echo ""
echo "오프라인 환경에서 다음 명령 실행:"
echo "  tar -xzf offline_resources.tar.gz"
echo "  docker load < offline_resources/docker_images/nvidia-cuda-12.4.0-runtime-ubuntu22.04.tar"
echo "  docker load < offline_resources/docker_images/ollama-latest.tar"
echo ""

