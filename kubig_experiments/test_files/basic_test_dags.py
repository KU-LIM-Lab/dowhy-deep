#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import pandas as pd
from dowhy import CausalModel

# --- logger 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 데이터 로드
df = pd.read_csv("./kubig_experiments/data/synthetic_data.csv")

# DAG 폴더 내의 파일들
dag_dir = Path("./kubig_experiments/dags")
dag_files = sorted(dag_dir.glob("sample_dag_*.txt"))

# DAG 파일과 treatment 변수 매핑 -- 추후 txt에서 treament 추출하는 로직으로 변경 예정
treatments = ["FRFT_AFTR_JHNT_REQR_DYCT", "BFR_OCTR_YN"]

logging.info("발견된 DAG 파일 개수: %d", len(dag_files))

for idx, dag_file in enumerate(dag_files):
    trt = treatments[idx]
    logging.info("===== Processing %s (Treatment: %s) =====", dag_file.name, trt)

    # 그래프 불러오기
    graph_txt = dag_file.read_text(encoding="utf-8")

    # 모델 정의
    model = CausalModel(
        data=df,
        treatment=trt,
        outcome="ACQ_180_YN",
        graph=graph_txt
    )

    # Identify 단계
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    logging.info("[%s] Identify: %s", dag_file.name, str(identified).splitlines()[0])

    # Treatment 타입 판별
    if pd.api.types.is_numeric_dtype(df[trt]) and df[trt].nunique() > 2:
        method = "backdoor.linear_regression"
        refuter = dict(method_name="data_subset_refuter",
                       subset_fraction=0.8,
                       num_simulations=100)
    else:
        method = "backdoor.propensity_score_matching"
        refuter = dict(method_name="placebo_treatment_refuter",
                       num_simulations=5)

    # Estimate 단계
    est = model.estimate_effect(
        identified,
        method_name=method,
        test_significance=True
    )
    logging.info("[%s] ATE: %s", dag_file.name, est.value)

    # Refute 단계
    ref = model.refute_estimate(
        identified,
        est,
        **refuter
    )
    logging.info("[%s] Refuter(%s): %s",
                 dag_file.name,
                 refuter["method_name"],
                 str(ref).splitlines()[0])
