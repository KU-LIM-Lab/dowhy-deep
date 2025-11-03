#!/bin/bash

# DoWhy 인과추론 분석 실행 스크립트
# 
# 사용법:
#   ./run_analysis.sh [옵션]
#
# 예시:
#   ./run_analysis.sh --data data/dummy_data.csv --graph data/dummy_graph --estimator linear_regression
#   ./run_analysis.sh --data data/dummy_data.csv --graph data/dummy_graph --estimator tabpfn

set -e  # 오류 발생 시 스크립트 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 기본 설정
DEFAULT_DATA="data/dummy_data.csv"
DEFAULT_GRAPH="data/dummy_graph"
DEFAULT_ESTIMATOR="linear_regression"
DEFAULT_TREATMENT="ACCR_CD"
DEFAULT_OUTCOME="ACQ_180_YN"

# 도움말 함수
show_help() {
    cat << EOF
DoWhy 인과추론 분석 실행 스크립트

사용법:
    $0 [옵션]

옵션:
    -d, --data FILE          데이터 파일 경로 (기본값: $DEFAULT_DATA)
    -g, --graph FILE         그래프 파일 경로 (기본값: $DEFAULT_GRAPH)
    -e, --estimator METHOD   추정 방법 (기본값: $DEFAULT_ESTIMATOR)
                              선택지: linear_regression, tabpfn, propensity_score, instrumental_variable
    -t, --treatment VAR      처치 변수명 (기본값: $DEFAULT_TREATMENT)
    -o, --outcome VAR        결과 변수명 (기본값: $DEFAULT_OUTCOME)
    --no-logs                로그 파일 저장 비활성화
    --verbose                상세한 출력 활성화
    -h, --help               이 도움말 표시

예시:
    # 기본 실행
    $0

    # TabPFN 사용
    $0 --estimator tabpfn

    # 커스텀 데이터와 그래프 사용
    $0 --data my_data.csv --graph my_graph --estimator linear_regression

    # 로그 비활성화
    $0 --no-logs

    # 상세 출력
    $0 --verbose

EOF
}

# 인자 파싱
DATA_FILE="$DEFAULT_DATA"
GRAPH_FILE="$DEFAULT_GRAPH"
ESTIMATOR="$DEFAULT_ESTIMATOR"
TREATMENT="$DEFAULT_TREATMENT"
OUTCOME="$DEFAULT_OUTCOME"
NO_LOGS=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA_FILE="$2"
            shift 2
            ;;
        -g|--graph)
            GRAPH_FILE="$2"
            shift 2
            ;;
        -e|--estimator)
            ESTIMATOR="$2"
            shift 2
            ;;
        -t|--treatment)
            TREATMENT="$2"
            shift 2
            ;;
        -o|--outcome)
            OUTCOME="$2"
            shift 2
            ;;
        --no-logs)
            NO_LOGS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            echo "도움말을 보려면 $0 --help를 실행하세요."
            exit 1
            ;;
    esac
done

# Python 환경 확인
check_python() {
    if ! command -v python &> /dev/null; then
        log_error "Python이 설치되어 있지 않습니다."
        exit 1
    fi
    
    # Python 버전 확인
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python 버전: $PYTHON_VERSION"
}

# 필요한 파일 존재 확인
check_files() {
    if [[ ! -f "$DATA_FILE" ]]; then
        log_error "데이터 파일을 찾을 수 없습니다: $DATA_FILE"
        exit 1
    fi
    
    if [[ ! -f "$GRAPH_FILE" ]]; then
        log_error "그래프 파일을 찾을 수 없습니다: $GRAPH_FILE"
        exit 1
    fi
    
    log_success "필요한 파일들이 모두 존재합니다."
}

# 추정 방법 유효성 검사
validate_estimator() {
    case $ESTIMATOR in
        linear_regression|tabpfn|propensity_score|instrumental_variable)
            log_info "추정 방법: $ESTIMATOR"
            ;;
        *)
            log_error "지원하지 않는 추정 방법: $ESTIMATOR"
            log_info "지원하는 방법: linear_regression, tabpfn, propensity_score, instrumental_variable"
            exit 1
            ;;
    esac
}

# 의존성 확인
check_dependencies() {
    log_info "Python 패키지 의존성을 확인하는 중..."
    
    # pandas 확인
    if ! python -c "import pandas" 2>/dev/null; then
        log_error "pandas가 설치되어 있지 않습니다. 'pip install pandas'를 실행하세요."
        exit 1
    fi
    
    # numpy 확인
    if ! python -c "import numpy" 2>/dev/null; then
        log_error "numpy가 설치되어 있지 않습니다. 'pip install numpy'를 실행하세요."
        exit 1
    fi
    
    # matplotlib 확인
    if ! python -c "import matplotlib" 2>/dev/null; then
        log_error "matplotlib가 설치되어 있지 않습니다. 'pip install matplotlib'를 실행하세요."
        exit 1
    fi
    
    # networkx 확인
    if ! python -c "import networkx" 2>/dev/null; then
        log_error "networkx가 설치되어 있지 않습니다. 'pip install networkx'를 실행하세요."
        exit 1
    fi
    
    # dowhy 확인
    if ! python -c "import dowhy" 2>/dev/null; then
        log_error "dowhy가 설치되어 있지 않습니다. 'pip install dowhy'를 실행하세요."
        exit 1
    fi
    
    log_success "모든 필수 패키지가 설치되어 있습니다."
}

# Python 명령어 구성
build_python_command() {
    PYTHON_CMD="python -m src.main"
    PYTHON_CMD="$PYTHON_CMD --data $DATA_FILE"
    PYTHON_CMD="$PYTHON_CMD --graph $GRAPH_FILE"
    PYTHON_CMD="$PYTHON_CMD --estimator $ESTIMATOR"
    PYTHON_CMD="$PYTHON_CMD --treatment $TREATMENT"
    PYTHON_CMD="$PYTHON_CMD --outcome $OUTCOME"
    
    if [[ "$NO_LOGS" == true ]]; then
        PYTHON_CMD="$PYTHON_CMD --no-logs"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        PYTHON_CMD="$PYTHON_CMD --verbose"
    fi
    
    echo "$PYTHON_CMD"
}

# 메인 실행 함수
main() {
    log_info "DoWhy 인과추론 분석을 시작합니다..."
    echo "========================================"
    
    # 환경 확인
    check_python
    check_files
    validate_estimator
    check_dependencies
    
    echo "========================================"
    log_info "분석 설정:"
    echo "  - 데이터 파일: $DATA_FILE"
    echo "  - 그래프 파일: $GRAPH_FILE"
    echo "  - 추정 방법: $ESTIMATOR"
    echo "  - 처치 변수: $TREATMENT"
    echo "  - 결과 변수: $OUTCOME"
    echo "  - 로그 저장: $([ "$NO_LOGS" == true ] && echo "비활성화" || echo "활성화")"
    echo "  - 상세 출력: $([ "$VERBOSE" == true ] && echo "활성화" || echo "비활성화")"
    echo "========================================"
    
    # Python 명령어 실행
    PYTHON_CMD=$(build_python_command)
    log_info "실행 명령어: $PYTHON_CMD"
    echo ""
    
    # 실행 시간 측정
    START_TIME=$(date +%s)
    
    if eval "$PYTHON_CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo ""
        log_success "분석이 성공적으로 완료되었습니다!"
        log_info "실행 시간: ${DURATION}초"
        
        # 결과 파일 위치 안내
        if [[ "$NO_LOGS" == false ]]; then
            log_info "로그 파일과 결과는 'log/' 폴더에서 확인할 수 있습니다."
        fi
    else
        log_error "분석 중 오류가 발생했습니다."
        exit 1
    fi
}

# 스크립트 실행
main "$@"
