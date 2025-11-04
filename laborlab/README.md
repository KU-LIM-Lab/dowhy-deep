# LaborLab - DoWhy 기반 인과추론 분석 파이프라인

구직자 데이터를 활용한 인과추론 분석을 위한 End-to-End 파이프라인입니다. DoWhy 라이브러리를 사용하여 다양한 treatment와 outcome에 대한 인과효과를 추정하고 검증합니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 환경 설정](#설치-및-환경-설정)
- [데이터 구조](#데이터-구조)
- [파이프라인 설명](#파이프라인-설명)
- [사용법](#사용법)
  - [단일 실험 실행](#단일-실험-실행)
  - [배치 실험 실행](#배치-실험-실행)
  - [Docker를 사용한 실행](#docker를-사용한-실행)
- [설정 파일](#설정-파일)
- [결과 확인](#결과-확인)

## 🎯 프로젝트 개요

이 프로젝트는 다음과 같은 기능을 제공합니다:

- **정형 데이터와 비정형 데이터 통합**: CSV 데이터와 JSON 데이터(이력서, 자기소개서, 직업훈련, 자격증)를 통합 처리
- **자동 Treatment 추출**: 그래프 파일에서 treatment 정보를 자동으로 추출하여 실험 생성
- **다양한 추정 방법 지원**: TabPFN, Linear Regression, Propensity Score Matching 등
- **인과효과 검증**: 다양한 refutation 테스트와 민감도 분석 제공
- **배치 실험 지원**: 여러 그래프와 treatment 조합을 자동으로 실험

## 📁 프로젝트 구조

```
laborlab/
├── src/                          # 소스 코드
│   ├── main.py                   # 메인 실행 스크립트 (단일 실험)
│   ├── preprocess.py             # 데이터 전처리 모듈
│   ├── estimation.py             # 인과효과 추정 모듈
│   ├── graph_parser.py           # 그래프 파일 파싱 (treatment 추출)
│   ├── llm_scorer.py             # LLM 기반 점수 계산 (선택적)
│   └── llm_reference.py          # LLM 프롬프트 설정
├── data/                         # 데이터 디렉토리
│   ├── fixed_data/               # 정형 데이터
│   │   ├── data.csv              # 메인 데이터
│   │   └── job_subcategories.csv # 직종 코드 매핑
│   ├── variant_data/             # 비정형 데이터 (JSON)
│   │   ├── RESUME_JSON.json      # 이력서 데이터
│   │   ├── COVERLETTERS_JSON.json # 자기소개서 데이터
│   │   ├── TRAININGS_JSON.json   # 직업훈련 데이터
│   │   ├── LICENSES_JSON.json    # 자격증 데이터
│   │   └── variable_mapping.json # 변수 매핑 정보
│   └── graph_data/               # 인과 그래프 파일
│       ├── graph_1.dot           # 그래프 1 (GML/DOT 형식)
│       ├── graph_2.dot           # 그래프 2
│       └── ...
├── run_batch_experiments.py      # 배치 실험 실행 스크립트
├── experiment_config.json        # 실험 설정 파일
├── Dockerfile                    # Docker 이미지 설정
├── docker-compose.yml            # Docker Compose 설정
└── requirements.txt              # Python 의존성
```

## 🚀 설치 및 환경 설정

### 1. Python 환경 설정

```bash
# Python 3.11 이상 필요
python --version

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (선택적)

LLM 기능을 사용하려면:

```bash
export LLM_API_KEY="your-api-key-here"
```

### 3. Docker 설정 (선택적)

Docker를 사용하는 경우:

```bash
docker-compose build
```

## 📊 데이터 구조

### 정형 데이터 (`data/fixed_data/data.csv`)
- 구직자 정보: 연령, 학력, 경력, 희망 임금 등
- 주요 컬럼: `SEEK_CUST_NO`, `JHNT_CTN`, `ACCR_CD`, `CARR_MYCT1`, `ACQ_180_YN` 등

### 비정형 데이터 (JSON)
- **이력서** (`RESUME_JSON.json`): 학력, 경력, 자격증 등
- **자기소개서** (`COVERLETTERS_JSON.json`): 자기소개서 내용
- **직업훈련** (`TRAININGS_JSON.json`): 훈련 이력
- **자격증** (`LICENSES_JSON.json`): 자격증 정보

각 JSON 파일은 `SEEK_CUST_NO` 또는 `JHNT_CTN`으로 연결됩니다.

### 인과 그래프
- `.dot` 또는 GML 형식의 그래프 파일
- `subgraph cluster_treatments` 블록에 treatment 메타데이터 포함
- 각 treatment는 `treatment_var`, `treatment_name`, `treatment_def` 등의 속성 포함

## 🔄 파이프라인 설명

전체 파이프라인은 다음과 같은 단계로 구성됩니다:

```
1. 데이터 로드
   ├── 정형 데이터 (CSV) 로드
   ├── 비정형 데이터 (JSON) 로드
   └── 인과 그래프 로드

2. 데이터 전처리
   ├── 정형 데이터 기본 전처리
   │   ├── 변수 필터링
   │   └── 범주형 변수 처리
   └── 비정형 데이터 NLP 전처리
       ├── 이력서 점수 계산 (LLM 또는 기본값)
       ├── 자기소개서 점수 계산
       ├── 직업훈련 점수 계산
       └── 자격증 점수 계산

3. 데이터 병합
   ├── JHNT_CTN 기준 병합 (직업훈련, 자격증)
   └── SEEK_CUST_NO 기준 병합 (이력서, 자기소개서)

4. 인과모델 생성
   ├── CausalModel 초기화
   └── 인과효과 식별

5. 인과효과 추정
   └── 선택한 추정 방법으로 추정 (TabPFN, Linear Regression 등)

6. 검증 및 분석
   ├── Refutation 테스트
   ├── 민감도 분석
   └── 결과 시각화
```

## 💻 사용법

### 단일 실험 실행

단일 treatment와 outcome에 대한 실험을 실행합니다:

```bash
python -m src.main \
    --data-dir data \
    --graph data/graph_data/graph_1.dot \
    --treatment ACCR_CD \
    --outcome ACQ_180_YN \
    --estimator linear_regression
```

**옵션 설명:**
- `--data-dir`: 데이터 디렉토리 경로
- `--graph`: 인과 그래프 파일 경로 (생략 시 `data/main_graph` 또는 `data/graph_data/`에서 자동 탐색)
- `--treatment`: 처치 변수명
- `--outcome`: 결과 변수명
- `--estimator`: 추정 방법 (`tabpfn`, `linear_regression`, `propensity_score`, `instrumental_variable`)
- `--no-logs`: 로그 저장 비활성화
- `--verbose`: 상세 출력 활성화

### 배치 실험 실행

여러 그래프와 treatment 조합을 자동으로 실험합니다:

```bash
python run_batch_experiments.py --config experiment_config.json
```

또는 Docker 없이 실행:

```bash
python run_batch_experiments.py
```

### Docker를 사용한 실행

Docker Compose를 사용하여 전체 배치 실험을 실행:

```bash
# Docker 이미지 빌드
docker-compose build

# 실험 실행
docker-compose up

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

## ⚙️ 설정 파일

### `experiment_config.json`

배치 실험 설정 파일입니다:

```json
{
  "data_dir": "data",
  "graph_data_dir": "graph_data",
  "auto_extract_treatments": true,
  "graphs": [
    "main_graph",
    "dummy_graph"
  ],
  "treatments": [
    "ACCR_CD",
    "CARR_MYCT1"
  ],
  "outcomes": [
    "ACQ_180_YN"
  ],
  "estimators": [
    "tabpfn",
    "linear_regression"
  ],
  "no_logs": false,
  "verbose": false
}
```

**설정 옵션:**

- `auto_extract_treatments`: `true`로 설정하면 그래프 파일에서 자동으로 treatment 추출
- `graph_data_dir`: 그래프 파일이 있는 폴더 (기본값: `graph_data`)
- `graphs`: 수동으로 지정할 그래프 파일 목록 (auto_extract_treatments가 true이면 무시)
- `treatments`: 수동으로 지정할 treatment 목록 (auto_extract_treatments가 true이면 무시)
- `outcomes`: 결과 변수 목록
- `estimators`: 사용할 추정 방법 목록
- `no_logs`: 로그 저장 비활성화
- `verbose`: 상세 출력 활성화

### 그래프 파일에서 Treatment 자동 추출

`auto_extract_treatments: true`로 설정하면:

1. `graph_data` 폴더의 모든 `.dot` 파일을 자동으로 찾음
2. 각 그래프 파일에서 `subgraph cluster_treatments` 블록 추출
3. 각 treatment의 `treatment_var` 속성을 추출하여 실험 생성
4. 각 그래프별로 해당 그래프의 treatment만 사용하여 실험

**예시:**
```dot
subgraph cluster_treatments {
    label="Treatments (outcome: ACQ_180_YN)";
    T1 [
        label="T1: prev_training_any",
        role="treatment_meta",
        treatment_var="BFR_OCTR_CT",
        treatment_name="prev_training_any",
        treatment_def="BFR_OCTR_CT > 0",
        treatment_question="이전 직업훈련을 한 번이라도 받은 사람은 그렇지 않은 사람보다 6개월 이내 취업 확률이 얼마나 높은가?"
    ];
}
```

위 예시에서 `BFR_OCTR_CT`가 treatment로 추출됩니다.

## 📈 결과 확인

### 로그 파일

실험 결과는 `log/` 폴더에 저장됩니다:

- `python_output_YYYYMMDD_HHMMSS.log`: 터미널 출력 로그
- `{graph_name}_{treatment}_{timestamp}.log`: 각 실험별 상세 로그
- `batch_experiments_YYYYMMDD_HHMMSS.json`: 배치 실험 결과 요약

### 결과 JSON 파일 구조

```json
{
  "experiment_id": "exp_0001_graph_1_ACCR_CD_ACQ_180_YN_tabpfn",
  "status": "success",
  "duration_seconds": 123.45,
  "graph": "/path/to/graph_1.dot",
  "treatment": "ACCR_CD",
  "outcome": "ACQ_180_YN",
  "estimator": "tabpfn",
  "start_time": "2025-11-03T22:44:46",
  "end_time": "2025-11-03T22:46:50"
}
```

## 🔧 문제 해결

### 1. ModuleNotFoundError: No module named 'openai'

LLM 기능을 사용하지 않는 경우 정상입니다. `openai` 패키지가 없어도 기본값으로 작동합니다.

LLM 기능을 사용하려면:
```bash
pip install openai
export LLM_API_KEY="your-api-key"
```

### 2. 그래프 파일을 찾을 수 없음

- `data/main_graph` 파일이 있는지 확인
- 또는 `data/graph_data/` 폴더에 그래프 파일이 있는지 확인
- `--graph` 옵션으로 직접 경로 지정

### 3. 데이터 파일 경로 오류

데이터 구조 확인:
```
data/
├── fixed_data/
│   └── data.csv
└── variant_data/
    ├── RESUME_JSON.json
    ├── COVERLETTERS_JSON.json
    ├── TRAININGS_JSON.json
    └── LICENSES_JSON.json
```

## 📝 참고 사항

- **LLM 기능**: LLM 기반 점수 계산은 선택적 기능입니다. `openai` 패키지가 없어도 기본값(50점)으로 작동합니다.
- **그래프 형식**: GML 형식과 DOT 형식을 지원합니다.
- **Treatment 추출**: 그래프 파일에 `treatment_meta` role이 있는 노드만 treatment로 인식됩니다.
- **병합 키**: 직업훈련과 자격증은 `JHNT_CTN` 기준, 이력서와 자기소개서는 `SEEK_CUST_NO` 기준으로 병합됩니다.

## 📚 추가 정보

### Notebook 폴더

`notebook/` 폴더에는 개발 및 테스트용 노트북과 스크립트가 포함되어 있습니다:
- `causal_validation.py`: 인과효과 검증 테스트
- `*.ipynb`: 개발 및 탐색적 분석 노트북

### 로그 폴더

`log/` 폴더에는 실험 결과와 로그 파일이 저장됩니다. 이 폴더는 자동으로 생성되며, 실험 실행 시마다 새로운 로그 파일이 생성됩니다.

## 📄 라이선스

프로젝트 라이선스에 따라 사용하세요.

