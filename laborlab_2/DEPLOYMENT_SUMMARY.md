# 폐쇠망 배포 요약

## 📦 필수 파일 목록 (간단 버전)

### 1. 소스 코드 (7개 파일)
```
src/
├── __init__.py
├── main.py
├── preprocess.py
├── estimation.py
├── utils.py          ← util 폴더 대신 단일 파일
├── llm_scorer.py
└── llm_reference.py
```

### 2. 설정 파일 (4개 파일)
```
├── config.json
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### 3. 데이터 파일
```
data/
├── seis_data/
│   ├── seis_data.csv
│   ├── resume.json
│   ├── coverletters.json
│   ├── trainings.json
│   └── licenses.json
├── graph_data/
│   └── *.dot (모든 그래프 파일)
├── variable_mapping.json
└── job_subcategories.csv
```

### 4. DoWhy 라이브러리
```
../dowhy/  (프로젝트 루트에 있어야 함)
```

## 🚀 빠른 배포 가이드

### 1단계: 파일 확인
```bash
cd laborlab_2
./check_deployment.sh
```

### 2단계: 압축
```bash
# laborlab_2 폴더만 압축
tar -czf laborlab_2.tar.gz laborlab_2/

# DoWhy 라이브러리 압축 (프로젝트 루트에서)
tar -czf dowhy.tar.gz dowhy/
```

### 3단계: 폐쇠망 서버에서 압축 해제
```bash
tar -xzf laborlab_2.tar.gz
tar -xzf dowhy.tar.gz
```

### 4단계: 패키지 설치 (로컬 패키지 사용)
```bash
cd laborlab_2
pip install --find-links /path/to/local/packages -r requirements.txt
```

### 5단계: Docker 실행
```bash
docker-compose up --build
```

## 📋 체크리스트

- [ ] 소스 코드 7개 파일 모두 존재
- [ ] 설정 파일 4개 모두 존재
- [ ] 데이터 파일 (seis_data 5개, graph_data 모든 .dot 파일, variable_mapping.json, job_subcategories.csv)
- [ ] DoWhy 라이브러리 (../dowhy/)
- [ ] Python 패키지 (로컬 설치 가능)
- [ ] Docker 이미지 (또는 빌드 가능한 환경)

## ⚠️ 주의사항

1. **utils.py**: `util/` 폴더가 아닌 단일 `utils.py` 파일 사용
2. **DoWhy 위치**: 프로젝트 루트(`dowhy_deep/`)에 `dowhy/` 폴더 필요
3. **패키지**: 모든 Python 패키지를 로컬에서 설치 가능해야 함
4. **GPU**: CUDA 12.4, NVIDIA Docker 필요

