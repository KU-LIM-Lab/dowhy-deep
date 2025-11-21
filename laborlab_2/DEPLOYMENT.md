# 폐쇠망 배포 가이드

폐쇠망 GPU 서버에 LaborLab 2를 배포하기 위해 필요한 파일 및 폴더 목록입니다.

## 📦 필수 파일 및 폴더

### 1. 소스 코드 (`src/` 폴더 전체)

```
laborlab_2/
└── src/
    ├── __init__.py
    ├── main.py              # 메인 파이프라인
    ├── preprocess.py        # 데이터 전처리
    ├── estimation.py        # 인과효과 추정
    ├── utils.py             # 유틸리티 (그래프 파싱, 로깅)
    ├── llm_scorer.py       # LLM 점수 계산
    └── llm_reference.py     # LLM 프롬프트 설정
```

**필수 파일:**
- `src/__init__.py`
- `src/main.py`
- `src/preprocess.py`
- `src/estimation.py`
- `src/utils.py`
- `src/llm_scorer.py`
- `src/llm_reference.py`

### 2. 설정 파일

```
laborlab_2/
├── config.json              # 실험 설정 파일 (필수)
├── requirements.txt         # Python 의존성 목록 (필수)
├── Dockerfile              # Docker 이미지 빌드 파일 (필수)
├── docker-compose.yml      # Docker Compose 설정 (필수)
└── .dockerignore           # Docker 빌드 제외 파일 (선택)
```

**필수 파일:**
- `config.json`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`

### 3. 데이터 파일 (`data/` 폴더 전체)

```
laborlab_2/
└── data/
    ├── seis_data/          # 정형 및 비정형 데이터 (필수)
    │   ├── seis_data.csv
    │   ├── resume.json
    │   ├── coverletters.json
    │   ├── trainings.json
    │   └── licenses.json
    ├── graph_data/         # 인과 그래프 파일 (필수)
    │   ├── graph_1.dot
    │   ├── graph_2.dot
    │   └── ... (모든 .dot 파일)
    ├── variable_mapping.json  # 변수 매핑 정보 (필수)
    └── job_subcategories.csv  # 직종 코드 매핑 (필수)
```

**필수 파일:**
- `data/seis_data/seis_data.csv`
- `data/seis_data/resume.json`
- `data/seis_data/coverletters.json`
- `data/seis_data/trainings.json`
- `data/seis_data/licenses.json`
- `data/graph_data/*.dot` (모든 그래프 파일)
- `data/variable_mapping.json`
- `data/job_subcategories.csv`

**선택 파일:**
- `data/metadata.xlsx` (메타데이터가 필요한 경우)

### 4. 문서 파일 (선택)

```
laborlab_2/
├── README.md               # 사용 가이드 (권장)
└── DEPLOYMENT.md           # 배포 가이드 (현재 파일)
```

### 5. Docker 관련 파일

**필수:**
- `Dockerfile`
- `docker-compose.yml`

**선택:**
- `.dockerignore`

### 6. Python 패키지 (로컬 설치용)

폐쇠망 환경에서는 인터넷 연결이 없으므로, 다음 패키지들을 로컬에 미리 준비해야 합니다:

```
packages/                   # 로컬 패키지 저장소 (생성 필요)
├── numpy-*.whl
├── pandas-*.whl
├── scikit-learn-*.whl
├── scipy-*.whl
├── statsmodels-*.whl
├── networkx-*.whl
├── sympy-*.whl
├── joblib-*.whl
├── tqdm-*.whl
├── causal-learn-*.whl
├── econml-*.whl
├── numba-*.whl
├── torch-*.whl             # GPU 버전 (CUDA 12.4)
├── tabpfn-*.whl           # 로컬 빌드 필요
├── matplotlib-*.whl
├── pydot-*.whl
├── python-dateutil-*.whl
├── openpyxl-*.whl
├── openai-*.whl           # 선택적
└── ollama-*.whl           # 선택적
```

### 7. DoWhy 라이브러리

DoWhy는 프로젝트 루트(`dowhy_deep/`)에 있어야 합니다. 전체 `dowhy/` 폴더가 필요합니다.

```
dowhy_deep/
└── dowhy/                  # DoWhy 라이브러리 전체 (필수)
    ├── __init__.py
    ├── causal_model.py
    ├── causal_estimator.py
    └── ... (모든 DoWhy 모듈)
```

### 8. Ollama 모델 (선택)

LLM 기능을 사용하는 경우:

```
laborlab_2/
└── ollama_models/          # Ollama 모델 파일 (선택)
    ├── blobs/
    └── manifests/
```

## 📋 배포 체크리스트

### 필수 파일 확인

```bash
# 소스 코드 확인
ls -la laborlab_2/src/
# __init__.py, main.py, preprocess.py, estimation.py, utils.py, llm_scorer.py, llm_reference.py

# 설정 파일 확인
ls -la laborlab_2/
# config.json, requirements.txt, Dockerfile, docker-compose.yml

# 데이터 파일 확인
ls -la laborlab_2/data/seis_data/
# seis_data.csv, resume.json, coverletters.json, trainings.json, licenses.json

ls -la laborlab_2/data/graph_data/
# graph_*.dot 파일들

# 필수 데이터 파일 확인
test -f laborlab_2/data/variable_mapping.json && echo "OK" || echo "MISSING"
test -f laborlab_2/data/job_subcategories.csv && echo "OK" || echo "MISSING"
```

### 폐쇠망 배포 시나리오

1. **파일 압축**
   ```bash
   # laborlab_2 폴더 압축
   tar -czf laborlab_2.tar.gz laborlab_2/
   
   # DoWhy 라이브러리 압축 (필요한 경우)
   tar -czf dowhy.tar.gz dowhy/
   
   # 로컬 패키지 압축
   tar -czf packages.tar.gz packages/
   ```

2. **폐쇠망 서버로 전송**
   - USB, 외장하드, 또는 승인된 전송 방법 사용

3. **폐쇠망 서버에서 압축 해제**
   ```bash
   tar -xzf laborlab_2.tar.gz
   tar -xzf dowhy.tar.gz  # 필요한 경우
   tar -xzf packages.tar.gz  # 필요한 경우
   ```

4. **패키지 설치**
   ```bash
   cd laborlab_2
   pip install --find-links ../packages -r requirements.txt
   ```

5. **Docker 이미지 빌드**
   ```bash
   docker-compose build
   ```

6. **실행**
   ```bash
   docker-compose up
   ```

## 🚫 제외할 파일/폴더

다음 파일/폴더는 배포 시 제외해도 됩니다:

- `__pycache__/` - Python 캐시
- `*.pyc`, `*.pyo` - 컴파일된 Python 파일
- `.git/` - Git 저장소
- `log/` - 로그 파일 (실행 후 생성됨)
- `.memo.md` - 메모 파일
- `python_3_11_slim.tar` - Docker 이미지 (필요시 별도 전송)

## 📝 최소 배포 패키지

폐쇠망에 최소한으로 전송해야 할 파일 목록:

```
laborlab_2/
├── src/                    # 전체 폴더
│   ├── __init__.py
│   ├── main.py
│   ├── preprocess.py
│   ├── estimation.py
│   ├── utils.py
│   ├── llm_scorer.py
│   └── llm_reference.py
├── data/                   # 전체 폴더
│   ├── seis_data/
│   ├── graph_data/
│   ├── variable_mapping.json
│   └── job_subcategories.csv
├── config.json
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

**총 파일 수:** 약 10개 (소스) + 데이터 파일들 + 설정 파일 4개

## 🔍 파일 크기 확인

```bash
# 소스 코드 크기
du -sh laborlab_2/src/

# 데이터 크기
du -sh laborlab_2/data/

# 전체 크기
du -sh laborlab_2/
```

## ⚠️ 주의사항

1. **DoWhy 라이브러리**: 프로젝트 루트에 `dowhy/` 폴더가 있어야 합니다.
2. **Python 버전**: Python 3.11 이상 필요
3. **CUDA 버전**: CUDA 12.4 필요 (GPU 사용 시)
4. **Docker**: Docker 및 NVIDIA Docker 필요
5. **패키지**: 모든 Python 패키지를 로컬에서 설치 가능해야 함

