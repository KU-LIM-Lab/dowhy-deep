#!/bin/bash
# 폐쇠망 배포를 위한 패키징 스크립트
# 온라인 환경에서 실행하여 배포 패키지를 생성합니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy_package"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "폐쇠망 배포 패키지 생성"
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
cp Dockerfile.offline "${DEPLOY_DIR}/laborlab_2/"
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
    echo "  ⚠️ DoWhy 라이브러리는 별도로 전송해야 합니다."
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

# 6. 요약
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
echo ""
echo "폐쇠망 환경으로 전송할 파일:"
echo "  1. laborlab_2_deploy_${TIMESTAMP}.tar.gz"
if [ -f "${PROJECT_DIR}/../dowhy_library_${TIMESTAMP}.tar.gz" ]; then
    echo "  2. dowhy_library_${TIMESTAMP}.tar.gz"
else
    echo "  2. dowhy/ 폴더 (별도 전송 필요)"
fi
echo ""
echo "폐쇠망 환경에서 실행:"
echo "  1. tar -xzf laborlab_2_deploy_${TIMESTAMP}.tar.gz"
echo "  2. tar -xzf dowhy_library_${TIMESTAMP}.tar.gz (DoWhy가 별도인 경우)"
echo "  3. cd laborlab_2"
echo "  4. cd .."
echo "  5. python -m laborlab_2.src.main --config laborlab_2/config.json"
echo ""

