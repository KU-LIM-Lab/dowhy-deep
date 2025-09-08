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

# ===== A) 연속형 처리: FRFT_AFTR_JHNT_REQR_DYCT (PSM X) =====
graph_txt = Path("./kubig_experiments/dags/sample_dag_1.txt").read_text(encoding="utf-8")  # DOT 또는 GML
model = CausalModel(data=df, treatment="FRFT_AFTR_JHNT_REQR_DYCT", outcome="ACQ_180_YN", graph=graph_txt)

identified = model.identify_effect(proceed_when_unidentifiable=True)
logging.info("[A] Identify: %s", str(identified).splitlines()[0])

est = model.estimate_effect(identified, method_name="backdoor.linear_regression", test_significance=True)
logging.info("[A] ATE: %s", est.value)

ref_a = model.refute_estimate(
    identified, est,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=100
)
logging.info("[A] Refuter(data_subset): %s", str(ref_a).splitlines()[0])

# ===== B) 이진 처리: BFR_OCTR_YN (PSM O) =====
graph_txt = Path("./kubig_experiments/dags/sample_dag_2.txt").read_text(encoding="utf-8")
model2 = CausalModel(data=df, treatment="BFR_OCTR_YN", outcome="ACQ_180_YN", graph=graph_txt)

identified2 = model2.identify_effect(proceed_when_unidentifiable=True)
logging.info("[B] Identify: %s", str(identified2).splitlines()[0])

est2 = model2.estimate_effect(identified2, method_name="backdoor.propensity_score_matching", test_significance=True)
logging.info("[B] ATE: %s", est2.value)

ref_b = model2.refute_estimate(
    identified2, est2,
    method_name="placebo_treatment_refuter",
    num_simulations=5
)
logging.info("[B] Refuter(placebo): %s", str(ref_b).splitlines()[0])
