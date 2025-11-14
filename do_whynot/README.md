# do_whynot

한국고용정보원의 구직인증데이터 및 이력서/자기소개서/직업훈련/자격증 데이터를 기반으로
**다양한 DAG 시나리오에서 인과효과(ATE)를 추정하고 검증하여 취업확률을 예측**하는 파이프라인

---

## 전체 프로세스 개요

1. **데이터 로드 & 전처리**
   - `data/` 아래 CSV & JSON 파일 로드
   - 기본 구직인증 테이블 데이터에 이력서, 자소서, 자격증, 직업훈련 데이터 병합
   - binary columns 인코딩, datetime columns 정수화(소요일수화), 전체 NA 컬럼 제거, label encoding 진행
   - preprocessed_df.csv가 `data/output/`에 저장

2. **DAG 로딩**
   - `dags/`의 DAG txt 파일(`dag_1.txt` ~) 로드
   - dag_parser.py로 treatment 컬럼명 추출
   - dot_nx로 nx graph 처리

3. **llm inference**
   - batch별 자기소개서 내용(`SELF_INTRO_CONT`)에 대한 label inference
   - 결과는 `data/output/`에 `preprocessed_df.csv`로 저장

3. **인과효과(ATE) 추정**
   - Linear Regressor 기반 ATE 추정(baseline)
   - TabPFN 기반 ATE 추정
   - Multi-class의 경우, initial batch에서 treatment 및 control value 선택

4. **인과효과(ATE) Refutation**
   - Placebo Treatment  
   - Random Common Cause  

5. **결과 저장**
   - `logs/`에 DAG별 ATE, p-value, refutation 결과 저장
   - batch_results_*.csv 및 all_validation_results.csv 저장

6. **취업확률 예측**
   - 도출된 top_5_dags_info를 바탕으로 취업확률 예측
   - TabPFN estimator의 predict_fn 사용
   - `data/output/`에 `prediction_dag_{dag_num}.csv`로 결과 저장

---

## 디렉토리 구조

```bash
do_whynot/
├── main.py
├── config.py
├── requirements.txt
│
├── dags/
│   ├── dag_1.txt
│   ├── dag_2.txt
│   └── ...
│
├── data/
│   ├── synthetic_data_raw.csv
│   ├── synthetic_data_raw_10000.csv
│   │
│   ├── RESUME_JSON/ver1/*.json
│   ├── COVERLETTERS_JSON/ver1/*.json
│   ├── TRAININGS_JSON/output/*.json
│   ├── LICENSES_JSON/output/*.json
│   └── output/
│
├── models/
│   ├── config.json
│   ├── metrics_best.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── models.safetensors   # drive에서 저장 및 models에 로드
│   └── pytorch_model.bin    # drive에서 저장 및 models에 로드
│
└── src/
    ├── dag_parser.py
    ├── preprocessor.py
    ├── estimation.py
    ├── inference_top1.py
    ├── interpretator.py
    ├── prediction.py
    └── eda.py
```

---

## ⚙️ config.py 사용 방법

`do_whynot/config.py`는 파이프라인 전체 설정을 관리하는 핵심 파일입니다.


### 주요 옵션 설명

| 설정값 | 설명 |
|--------|------|
| **IS_TEST_MODE** | True일 경우 작은 데이터 샘플만 사용하여 빠르게 실행 |
| **TEST_SAMPLE_SIZE** | 테스트 실행 시 사용할 샘플 개수 |
| **BATCH_SIZE** | TabPFN 기반 ATE 추정 시 내부 데이터 배치 크기 |
| **DAG_INDICES** | 실행할 DAG 인덱스 목록 |
| **EXCLUDE_COLS** | 분석 제외 컬럼 리스트 |
| **MULTICLASS_THRESHOLD** | 카테고리 개수가 너무 큰 변수 필터링 기준 |


---

## 모델 파일 다운로드

모델 가중치는 다음의 드라이브에서 받아야 합니다.

[모델 다운로드 링크](https://drive.google.com/drive/folders/1dVU1o4YUhJajlOtVRfjd4AXWTw-_5_wf)

받은 후 다음 위치에 저장:

```
do_whynot/models/models.safetensors
do_whynot/models/pytorch_model.bin
```

---

## 🛠 설치 & 실행 방법

### 1. Graphviz 설치

```bash
sudo apt-get update
sudo apt-get install -y graphviz graphviz-dev
```

### 2. Python 패키지 설치

```bash
pip install -r do_whynot/requirements.txt
```

### 3. 실행

```bash
python do_whynot/main.py
```

결과는 데이터 관련 결과물은 `data/output/`에, 실행 결과 및 로그는 `logs/`에 저장됩니다.

---
