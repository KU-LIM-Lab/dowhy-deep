#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

from kubig_experiments.src.dag_parser import parse_edges_from_dot, extract_roles_general

def _dot_to_nx(graph_txt: str) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(parse_edges_from_dot(graph_txt))
    return g

@pytest.mark.parametrize("dag_idx", [0, 1])  # sample_dag_1.txt, sample_dag_2.txt
def test_tabpfn_estimator_runs(dag_idx):
    """
    - DAG에서 일반화 규칙으로 Treatment/Confounder/Mediator 자동 추출
    - Baseline 및 TabPFN 추정기 실행 검증
    - Validation 단계: Placebo treatment, Random treatment, DoWhy Refutation
    """
    df = pd.read_csv("./kubig_experiments/data/synthetic_data.csv")
    dag_dir = Path("./kubig_experiments/dags")
    dag_files = sorted(dag_dir.glob("sample_dag_*.txt"))
    dag_file = dag_files[dag_idx]
    graph_txt = dag_file.read_text(encoding="utf-8")

    roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
    treatment = roles["treatment"]

    print(f"\n[{dag_file.name}] Roles:")
    print("  X (treatment):", roles["treatment"])
    print("  M (mediators):", roles["mediators"])
    print("  C (confounders):", roles["confounders"])

    nx_graph = _dot_to_nx(graph_txt)
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome="ACQ_180_YN",
        graph=nx_graph,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    assert identified is not None

    if pd.api.types.is_numeric_dtype(df[treatment]) and df[treatment].nunique() > 2:
        method = "backdoor.linear_regression"
    else:
        method = "backdoor.propensity_score_matching"

    est = model.estimate_effect(identified, method_name=method, test_significance=True)
    assert isinstance(est.value, (float, np.floating)) and np.isfinite(est.value)

    est_tabpfn = model.estimate_effect(
        identified,
        method_name="backdoor.tabpfn",
        method_params={"estimator": TabpfnEstimator, "N_ensemble_configurations": 8},
    )
    assert est_tabpfn is not None
    assert isinstance(est_tabpfn.value, (float, np.floating)) and np.isfinite(est_tabpfn.value)

    print(f"[{dag_file.name}] [{treatment}] Baseline({method}) ATE: {est.value}")
    print(f"[{dag_file.name}] [{treatment}] TabPFN ATE: {est_tabpfn.value}")

    # ------------------------
    # Validation: DoWhy 내장 Refutation
    # ------------------------
    refuters = ["placebo_treatment_refuter", "random_common_cause"]
    for ref in refuters:
        refutation = model.refute_estimate(
            identified,
            est,
            method_name=ref,
        )
        print(f"[{dag_file.name}] Refutation ({ref}): {refutation}")
        # refutation의 결과값이 원래 추정치와 크게 차이나면 문제 있음
        assert refutation is not None
