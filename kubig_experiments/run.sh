#!/bin/bash

# 1. 시스템 라이브러리 설치 (Graphviz)
# APT 패키지 목록 업데이트 및 Graphviz 설치
echo "--- Installing System Dependencies (Graphviz) ---"
sudo apt-get update
sudo apt-get install -y graphviz graphviz-dev

# 2. Python 의존성 설치
# Python 가상 환경 설정 또는 활성화 후 실행하는 것이 좋습니다.
echo "--- Installing Python Dependencies ---"
pip install -r kubig_experiments/requirements.txt

# 3. 메인 파이프라인 실행
echo "--- Running Validation Pipeline ---"
python kubig_experiments/validation_pipeline.py