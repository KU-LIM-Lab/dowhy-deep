#!/bin/bash

# DoWhy 인과추론 분석 실행 스크립트

# 기본 설정
DATA_DIR="${1:-data}"
GRAPH_FILE="${2:-data/main_graph}"
TREATMENT="${3:-ACCR_CD}"
OUTCOME="${4:-ACQ_180_YN}"

# 추정 방법 목록
ESTIMATORS=("linear_regression" "tabpfn")

# log 폴더 생성
mkdir -p "log"

# 환경변수 설정
export TERMINAL_OUTPUT_DIR="log"

# 각 추정 방법으로 반복 실행
for ESTIMATOR in "${ESTIMATORS[@]}"; do
    echo "========================================"
    echo "🔧 추정 방법: $ESTIMATOR 실행 중..."
    echo "========================================"
    
    # Python 스크립트 실행
    python3 -m src.main \
        --data-dir "$DATA_DIR" \
        --graph "$GRAPH_FILE" \
        --estimator "$ESTIMATOR" \
        --treatment "$TREATMENT" \
        --outcome "$OUTCOME"
    
    echo ""
    echo "✅ $ESTIMATOR 분석 완료!"
    echo ""
done

echo "🎉 모든 추정 방법 분석 완료!"
