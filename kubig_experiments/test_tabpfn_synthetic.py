#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

@pytest.mark.parametrize("treatment", ["FRFT_AFTR_JHNT_REQR_DYCT", "BFR_OCTR_YN"])
def test_tabpfn_estimator_runs(treatment):
    """
    우리 synthetic_data.csv와 DAG에 대해
    - Baseline estimator (PSM or OLS)이 정상적으로 값 반환
    - TabPFN estimator가 실행 가능하고 finite float을 반환
    하는지 확인합니다.
    """
    # 데이터/그래프 불러오기
    df = pd.read_csv("./kubig_experiments/data/synthetic_data.csv")
    dag_dir = Path("./kubig_experiments/dags")
    dag_files = sorted(dag_dir.glob("sample_dag_*.txt"))
    dag_file = dag_files[0] if treatment == "FRFT_AFTR_JHNT_REQR_DYCT" else dag_files[1]
    graph_txt = dag_file.read_text(encoding="utf-8")

    # 모델 정의
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome="ACQ_180_YN",
        graph=graph_txt,
    )

    # Identify
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    assert identified is not None

    # Baseline 추정기 (참고용)
    if pd.api.types.is_numeric_dtype(df[treatment]) and df[treatment].nunique() > 2:
        method = "backdoor.linear_regression"
    else:
        method = "backdoor.propensity_score_matching"

    est = model.estimate_effect(
        identified,
        method_name=method,
        test_significance=True,
    )
    assert isinstance(est.value, (float, np.floating))
    assert np.isfinite(est.value)

    # TabPFN 추정기 (sanity check only)
    est_tabpfn = model.estimate_effect(
        identified,
        method_name="backdoor.tabpfn",
        method_params={"estimator": TabpfnEstimator, "N_ensemble_configurations": 8},
    )
    assert est_tabpfn is not None
    assert isinstance(est_tabpfn.value, (float, np.floating))
    assert np.isfinite(est_tabpfn.value)

    print(f"[Baseline {method}] ATE: {est.value}")
    print(f"[TabPFN] ATE: {est_tabpfn.value}")

