#!/bin/bash
# 폐쇠망 배포 전 필수 파일 확인 스크립트

echo "=========================================="
echo "LaborLab 2 배포 파일 확인"
echo "=========================================="
echo ""

ERROR_COUNT=0

# 소스 코드 확인
echo "[1] 소스 코드 확인"
REQUIRED_SRC_FILES=(
    "src/__init__.py"
    "src/main.py"
    "src/preprocess.py"
    "src/estimation.py"
    "src/utils.py"
    "src/llm_scorer.py"
    "src/llm_reference.py"
)

for file in "${REQUIRED_SRC_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ((ERROR_COUNT++))
    fi
done
echo ""

# 설정 파일 확인
echo "[2] 설정 파일 확인"
REQUIRED_CONFIG_FILES=(
    "config.json"
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
)

for file in "${REQUIRED_CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ((ERROR_COUNT++))
    fi
done
echo ""

# 데이터 파일 확인
echo "[3] 데이터 파일 확인"

# seis_data 확인
if [ -d "data/seis_data" ]; then
    echo "  ✓ data/seis_data/ 디렉토리"
    REQUIRED_DATA_FILES=(
        "data/seis_data/seis_data.csv"
        "data/seis_data/resume.json"
        "data/seis_data/coverletters.json"
        "data/seis_data/trainings.json"
        "data/seis_data/licenses.json"
    )
    
    for file in "${REQUIRED_DATA_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "    ✓ $file"
        else
            echo "    ✗ $file (MISSING)"
            ((ERROR_COUNT++))
        fi
    done
else
    echo "  ✗ data/seis_data/ 디렉토리 (MISSING)"
    ((ERROR_COUNT++))
fi

# graph_data 확인
if [ -d "data/graph_data" ]; then
    GRAPH_COUNT=$(find data/graph_data -name "*.dot" | wc -l)
    if [ "$GRAPH_COUNT" -gt 0 ]; then
        echo "  ✓ data/graph_data/ 디렉토리 ($GRAPH_COUNT개 그래프 파일)"
    else
        echo "  ✗ data/graph_data/ (그래프 파일 없음)"
        ((ERROR_COUNT++))
    fi
else
    echo "  ✗ data/graph_data/ 디렉토리 (MISSING)"
    ((ERROR_COUNT++))
fi

# 기타 데이터 파일
OTHER_DATA_FILES=(
    "data/variable_mapping.json"
    "data/job_subcategories.csv"
)

for file in "${OTHER_DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ((ERROR_COUNT++))
    fi
done
echo ""

# DoWhy 라이브러리 확인
echo "[4] DoWhy 라이브러리 확인"
if [ -d "../dowhy" ]; then
    echo "  ✓ ../dowhy/ 디렉토리 존재"
    if [ -f "../dowhy/__init__.py" ]; then
        echo "    ✓ DoWhy 라이브러리 확인됨"
    else
        echo "    ✗ DoWhy 라이브러리가 올바르지 않음"
        ((ERROR_COUNT++))
    fi
else
    echo "  ⚠ ../dowhy/ 디렉토리 없음 (프로젝트 루트에 있어야 함)"
    echo "    경고: DoWhy 라이브러리가 프로젝트 루트에 있어야 합니다."
fi
echo ""

# 요약
echo "=========================================="
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✓ 모든 필수 파일이 존재합니다!"
    echo "  배포 준비 완료"
    exit 0
else
    echo "✗ $ERROR_COUNT개의 필수 파일/폴더가 누락되었습니다."
    echo "  배포 전에 누락된 파일을 확인하세요."
    exit 1
fi

