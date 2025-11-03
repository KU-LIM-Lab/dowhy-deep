import pytest
import numpy as np
import pandas as pd

tabpfn = pytest.importorskip("tabpfn")
torch   = pytest.importorskip("torch")

from dowhy import CausalModel
from dowhy.datasets import linear_dataset
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator


def _make_multiclass_dataset(beta: float, num_common_causes: int = 4, n: int = 1000, num_classes: int = 3):
    rng = np.random.default_rng(2025)

    base = linear_dataset(
        beta=0.0,
        num_common_causes=num_common_causes,
        num_instruments=0,
        num_effect_modifiers=0,
        num_treatments=1,
        num_samples=n,
        treatment_is_binary=True,
        outcome_is_binary=False,
    )
    df = base["df"].copy()

    t_col = base["treatment_name"]
    w_names = base["common_causes_names"]
    y_col = base["outcome_name"]

    # T -> 1D (n,)
    T = np.asarray(df[t_col]).astype(np.float64).reshape(-1)
    W = np.asarray(df[w_names]).astype(np.float64)

    K = num_classes
    thetas = rng.normal(0, 0.7, size=(K, W.shape[1]))
    intercepts = rng.normal(0, 0.3, size=(K,))
    noise = rng.normal(0, 0.3, size=(len(df), K))

    logits = intercepts + W @ thetas.T + noise  # (n, K)
    logits[:, 1] = logits[:, 1] + beta * T  # (n,) + (n,) OK

    # softmax → label
    logits -= logits.max(axis=1, keepdims=True)
    expz = np.exp(logits)
    probs = expz / expz.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=probs[i]) for i in range(len(df))], dtype=int)

    df[y_col] = y

    if "W0" in df.columns:
        df["W0"] = pd.cut(df["W0"], bins=3, labels=["low", "medium", "high"])

    base["df"] = df
    return base



# ============ helpers ============
def _build_model(data, *, outcome_name=None, graph=None):
    outcome = outcome_name or data["outcome_name"]
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=outcome,
        common_causes=data["common_causes_names"],
        graph=(graph if graph is not None else data.get("gml_graph", None)),
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
def test_tabpfn_multiclass_targets_class1_prob(beta):
    """
    TabPFN 멀티클래스에서 proba[:,1] 기반 ATE의 부호가 beta와 일치하는지 확인.
    (TabpfnEstimator.predict_fn이 proba[:,1]을 사용)
    """
    data = _make_multiclass_dataset(beta=beta, num_common_causes=4, n=1000, num_classes=3)
    model, identified = _build_model(data)
    est_tabpf = _estimate_ate_with_tabpfn(model, identified)

    assert est_tabpf is not None and isinstance(est_tabpf.value, (float, np.floating)) and np.isfinite(est_tabpf.value)
    assert abs(est_tabpf.value) > 1e-3, "TabPFN (multiclass) ATE가 0에 너무 가깝습니다."
    assert np.sign(est_tabpf.value) == np.sign(beta), \
        f"[TabPFN multiclass] sign mismatch for class-1 prob: beta={beta}, ATE={est_tabpf.value:.4f}"

    print(f"\n[Multiclass ATE (class=1 prob) | beta={beta:+}] TabPFN={est_tabpf.value:+.4f}")


@pytest.mark.parametrize("beta", [2.0, -2.0])
def test_compare_multiclass_ovr_linear_vs_tabpfn(beta):
    """
    멀티클래스에서 OVR( y1 = 1{Y==1} ) 이진 outcome으로
    - Linear(OLS, backdoor.linear_regression)
    - TabPFN(멀티클래스 proba[:,1])
    의 부호 비교.
    """
    data = _make_multiclass_dataset(beta=beta, num_common_causes=4, n=1000, num_classes=3)
    df = data["df"].copy()
    y_mc = data["outcome_name"]
    y1 = "y_is_1"

    df[y1] = (df[y_mc] == 1).astype(int)

    data_ovr = {
        "df": df,
        "treatment_name": data["treatment_name"],
        "outcome_name": y1,
        "common_causes_names": data["common_causes_names"],
    }

    # Linear OVR
    model_lin, id_lin = _build_model(data_ovr, outcome_name=y1, graph=None)
    est_lin = _estimate_ate_with_linear(model_lin, id_lin)

    # TabPFN multiclass
    model_tab, id_tab = _build_model(data)
    est_tab = _estimate_ate_with_tabpfn(model_tab, id_tab)

    # validation check
    for name, est in [("Linear-OVR", est_lin), ("TabPFN", est_tab)]:
        assert est is not None and isinstance(est.value, (float, np.floating)) and np.isfinite(est.value), f"{name} ATE invalid"
        assert abs(est.value) > 1e-6, f"{name} ATE too close to 0"

    assert np.sign(est_lin.value) == np.sign(beta), f"[Linear-OVR] sign mismatch: beta={beta}, ATE={est_lin.value:.4f}"
    assert np.sign(est_tab.value) == np.sign(beta), f"[TabPFN] sign mismatch: beta={beta}, ATE={est_tab.value:.4f}"

    diff = est_tab.value - est_lin.value
    ratio = (est_tab.value / est_lin.value) if est_lin.value != 0 else np.nan
    print(
        f"\n[Multiclass OVR (class=1) | beta={beta:+}] "
        f"Linear-OVR={est_lin.value:+.4f} | TabPFN={est_tab.value:+.4f} "
        f"| Δ(TabPFN-Linear)={diff:+.4f} | ratio={ratio:+.3f}"
    )
