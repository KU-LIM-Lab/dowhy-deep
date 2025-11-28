# LaborLab 2 - 인과추론 분석 파이프라인

LaborLab의 리팩토링 버전으로, 폐쇠망 GPU 서버 환경에서 실행 가능한 통합 파이프라인입니다.

## 📋 주요 변경사항

1. **통합 파이프라인**: `main.py`와 `run_batch_experiments.py`를 병합하여 하나의 `main.py`로 통합
2. **모듈 구조 개선**: `graph_parser`와 `logger`를 `util` 모듈로 분리
3. **설정 파일 기반**: `config.json`에서 데이터 경로 및 실험 세팅 관리
4. **자동 실행**: `docker-compose up` 시 전체 파이프라인 자동 실행
5. **결과 저장**: CSV와 log 파일 두 가지 형식으로 결과 저장

## 📁 프로젝트 구조

```
laborlab_2/
├── src/                          # 소스 코드
│   ├── main.py                   # 메인 파이프라인 (통합)
│   ├── preprocess.py             # 데이터 전처리 모듈
│   ├── estimation.py             # 인과효과 추정 모듈
│   ├── llm_scorer.py             # LLM 기반 점수 계산
│   ├── llm_reference.py          # LLM 프롬프트 설정
│   └── util/                     # 유틸리티 모듈
│       ├── graph_parser.py       # 그래프 파일 파싱
│       └── logger.py             # 로깅 유틸리티
├── data/                         # 데이터 디렉토리
│   ├── seis_data/                # 정형 및 비정형 데이터
│   │   ├── seis_data.csv         # 정형 데이터
│   │   ├── resume.json           # 이력서 데이터
│   │   ├── coverletters.json     # 자기소개서 데이터
│   │   ├── trainings.json        # 직업훈련 데이터
│   │   └── licenses.json         # 자격증 데이터
│   ├── graph_data/               # 인과 그래프 파일
│   │   ├── graph_1.dot
│   │   └── ...
│   ├── variable_mapping.json    # 변수 매핑 정보
│   └── job_subcategories.csv    # 직종 코드 매핑
├── config.json                   # 실험 설정 파일
├── Dockerfile                    # Docker 이미지 설정
├── docker-compose.yml            # Docker Compose 설정
├── requirements.txt              # Python 의존성
└── log/                          # 결과 저장 디렉토리
    ├── experiment_results_*.csv  # CSV 결과
    └── batch_experiments_*.log  # 로그 파일
```

## 🚀 사용법

### 1. 설정 파일 수정

`config.json` 파일을 수정하여 실험 세팅을 변경합니다:

```json
{
  "data_dir": "data",
  "seis_data_dir": "seis_data",
  "graph_data_dir": "graph_data",
  "output_dir": "log",
  "auto_extract_treatments": true,
  "outcomes": ["ACQ_180_YN"],
  "estimators": ["linear_regression", "tabpfn"],
  "no_logs": false,
  "verbose": false
}
```

### 2. Docker Compose로 실행

```bash
# 컨테이너 빌드 및 실행 (자동으로 파이프라인 실행)
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f laborlab

# 컨테이너 중지
docker-compose down
```

### 3. 직접 실행 (Docker 없이)

```bash
# Python 환경 설정
python --version  # Python 3.11 이상 필요

# 의존성 설치 (로컬 환경)
pip install -r requirements.txt

# 파이프라인 실행
python -m src.main --config config.json
```

## ⚙️ 설정 파일 설명

### `config.json`

- `data_dir`: 데이터 디렉토리 경로 (기본값: "data")
- `seis_data_dir`: seis_data 디렉토리 이름 (기본값: "seis_data")
- `graph_data_dir`: 그래프 데이터 디렉토리 이름 (기본값: "graph_data")
- `output_dir`: 결과 저장 디렉토리 (기본값: "log")
- `auto_extract_treatments`: 그래프에서 자동으로 treatment 추출 여부 (기본값: true)
- `graphs`: 수동으로 지정할 그래프 파일 목록 (auto_extract_treatments가 true이면 무시)
- `treatments`: 수동으로 지정할 treatment 목록 (auto_extract_treatments가 true이면 무시)
- `outcomes`: 결과 변수 목록 (기본값: ["ACQ_180_YN"])
- `no_logs`: 로그 저장 비활성화 여부 (기본값: false)
- `verbose`: 상세 출력 활성화 여부 (기본값: false)
- `experiment_list`: 실험 조합 리스트 (배열 형식)
  - 각 실험은 `[graph_file, treatment, outcome, estimator]` 형식
  - 예: `["graph_1.dot", "BFR_OCTR_CT", "ACQ_180_YN", "tabpfn"]`
  - graph_file은 `graph_data_dir` 내의 파일명 또는 절대 경로
  - treatment는 그래프 파일의 `subgraph cluster_treatments` 블록에서 정의된 `treatment_var` 값
  - outcome은 일반적으로 "ACQ_180_YN"
  - estimator는 "tabpfn" 또는 "linear_regression"

**참고**: 
- Local ollama를 사용하므로 API 키 설정이 필요하지 않습니다.
- `experiment_list`가 정의되어 있으면 자동 생성 로직은 무시됩니다.
- `experiment_list`가 없으면 기존 방식(auto_extract_treatments 등)을 사용합니다.

## 📊 결과 확인

실험 결과는 `log/` 디렉토리에 저장됩니다:

- **CSV 결과**: `experiment_results_YYYYMMDD_HHMMSS.csv`
  - 각 실험의 결과를 테이블 형식으로 저장
  - 컬럼: graph_name, treatment, estimator, ate_value, refutation 결과, 메트릭 등
  
- **로그 파일**: `batch_experiments_YYYYMMDD_HHMMSS.log`
  - 상세한 실행 로그
  - 각 단계별 소요 시간 및 결과

- **JSON 결과**: `batch_experiments_YYYYMMDD_HHMMSS.json`
  - 전체 실험 결과를 JSON 형식으로 저장

## 🔧 폐쇠망 환경 설정

### 1. Python 패키지 로컬 설치

폐쇠망 환경에서는 인터넷 연결이 없으므로, 모든 패키지를 로컬에서 설치해야 합니다:

```bash
# 방법 1: 미리 다운로드한 패키지 설치
pip install --find-links /path/to/local/packages -r requirements.txt

# 방법 2: wheel 파일 직접 설치
pip install package_name.whl
```

### 2. Docker 이미지 로컬 로드

```bash
# Docker 이미지 로드
docker load < laborlab_2_image.tar

# 또는 docker-compose build 시 로컬 패키지 사용
docker-compose build --no-cache
```

### 3. GPU 설정

GPU를 사용하려면 NVIDIA Docker가 설치되어 있어야 합니다:

```bash
# NVIDIA Docker 설치 확인
nvidia-docker --version

# GPU 사용 가능 여부 확인
nvidia-smi
```

`docker-compose.yml`에서 GPU 설정이 이미 포함되어 있습니다.

## 📝 참고 사항

- **데이터 구조**: laborlab_2는 `seis_data` 폴더 구조를 사용합니다
- **그래프 파일**: `graph_data` 폴더의 `.dot` 파일을 자동으로 인식합니다
- **자동 Treatment 추출**: `auto_extract_treatments: true`로 설정하면 그래프 파일에서 자동으로 treatment를 추출합니다
- **LLM 기능**: Ollama 컨테이너가 실행 중이어야 LLM 기능을 사용할 수 있습니다

## 🐛 문제 해결

### 1. GPU 인식 안 됨

```bash
# NVIDIA Docker 설치 확인
nvidia-docker --version

# GPU 드라이버 확인
nvidia-smi
```

### 2. 패키지 설치 실패

폐쇠망 환경에서는 모든 패키지를 로컬에서 설치해야 합니다. `requirements.txt`의 TabPFN은 로컬에 설치된 버전을 사용하도록 수정되어 있습니다.

### 3. 데이터 파일을 찾을 수 없음

`config.json`에서 `data_dir`, `seis_data_dir`, `graph_data_dir` 경로를 확인하세요.

## 📄 라이선스

프로젝트 라이선스에 따라 사용하세요.

