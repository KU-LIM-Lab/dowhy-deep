import pytest
import numpy as np
import pandas as pd

tabpfn = pytest.importorskip("tabpfn")
torch   = pytest.importorskip("torch")

from dowhy import CausalModel
from dowhy.datasets import linear_dataset
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator


def _make_binary_dataset(beta: float, num_common_causes: int = 4, n: int = 1200):
    np.random.seed(2025)
    data = linear_dataset(
        beta=beta,
        num_common_causes=num_common_causes,
        num_instruments=0,
        num_effect_modifiers=0,
        num_treatments=1,
        num_samples=n,
        treatment_is_binary=True,
        outcome_is_binary=True,
    )

    # binary outcome (True→1, False→0)
    y_col = data["outcome_name"]
    df = data["df"].copy()
    df[y_col] = df[y_col].astype(int)

    if "W0" in df.columns:
        df["W0"] = pd.cut(df["W0"], bins=3, labels=["low", "medium", "high"])

    data["df"] = df
    return data


def _build_model(data):
    """공통 CausalModel + 식별자 반환"""
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        common_causes=data["common_causes_names"],
        graph=data["gml_graph"],
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model, identified_estimand


def _estimate_ate_with_tabpfn(model, identified_estimand):
    return model.estimate_effect(
        identified_estimand,
        method_name="backdoor.tabpfn",
        method_params={
            "estimator": TabpfnEstimator,
            "n_estimators": 8,
            "model_type": "classifier", 
        },
    )


def _estimate_ate_with_linear(model, identified_estimand):
    return model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=False,
    )


@pytest.mark.parametrize("beta", [2.0, -2.0])
def test_compare_tabpfn_vs_linear(beta):
    data = _make_binary_dataset(beta=beta, num_common_causes=4, n=1200)
    model, identified = _build_model(data)

    est_lin   = _estimate_ate_with_linear(model, identified)
    est_tabpf = _estimate_ate_with_tabpfn(model, identified)

    # validation check
    assert est_lin is not None and isinstance(est_lin.value, (float, np.floating)) and np.isfinite(est_lin.value), \
        f"Linear ATE invalid: {getattr(est_lin, 'value', None)}"
    assert est_tabpf is not None and isinstance(est_tabpf.value, (float, np.floating)) and np.isfinite(est_tabpf.value), \
        f"TabPFN ATE invalid: {getattr(est_tabpf, 'value', None)}"
    
    assert abs(est_tabpf.value) > 1e-3, "TabPFN 추정된 효과가 0에 너무 가깝습니다."
    assert abs(est_lin.value)   > 1e-6, "Linear 추정된 효과가 0에 너무 가깝습니다."

    # 핵심: 두 방법 모두 부호가 beta와 일치
    assert np.sign(est_tabpf.value) == np.sign(beta), \
        f"[TabPFN] ATE sign mismatch: beta={beta}, ATE={est_tabpf.value:.4f}"
    assert np.sign(est_lin.value) == np.sign(beta), \
        f"[Linear] ATE sign mismatch: beta={beta}, ATE={est_lin.value:.4f}"

    # report result
    diff = est_tabpf.value - est_lin.value
    ratio = (est_tabpf.value / est_lin.value) if est_lin.value != 0 else np.nan

    print(
        f"\n[Compare ATE | beta={beta:+}] "
        f"Linear={est_lin.value:+.4f} | TabPFN={est_tabpf.value:+.4f} "
        f"| Δ(TabPFN-Linear)={diff:+.4f} | ratio={ratio:+.3f}"
    )


@pytest.mark.parametrize("beta", [2.0, -2.0])
def test_tabpfn_estimator_sign_follows_beta(beta):
    data = _make_binary_dataset(beta=beta, num_common_causes=4, n=1200)
    model, identified = _build_model(data)
    estimate = _estimate_ate_with_tabpfn(model, identified)

    assert estimate is not None
    assert isinstance(estimate.value, (float, np.floating))
    assert abs(estimate.value) > 1e-3
    assert np.sign(estimate.value) == np.sign(beta), (
        f"ATE 부호 불일치: beta={beta}, estimated={estimate.value:.4f}"
    )
    print(f"\n[Binary ATE] beta={beta:+}, ATE={estimate.value:+.4f}")
