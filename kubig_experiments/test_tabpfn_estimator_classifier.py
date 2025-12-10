import pytest
import numpy as np

from dowhy import CausalModel
from dowhy.datasets import linear_dataset
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator


# Skip all tests in this file if tabpfn/torch are not available
tabpfn = pytest.importorskip("tabpfn")
torch = pytest.importorskip("torch")


def _make_binary_data(beta: float = 5.0, n: int = 300, k: int = 3):
    return linear_dataset(
        beta=beta,
        num_common_causes=k,
        num_instruments=0,
        num_effect_modifiers=0,
        num_samples=n,
        treatment_is_binary=True,
        outcome_is_binary=True,
    )


@pytest.mark.parametrize("model_type", ["auto", "classifier"])
def test_tabpfn_uses_classifier_and_estimates_ate(model_type):
    data = _make_binary_data(beta=5.0)

    cm = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        common_causes=data["common_causes_names"],
        graph=data["gml_graph"],
    )

    estimand = cm.identify_effect(proceed_when_unidentifiable=True)

    estimate = cm.estimate_effect(
        estimand,
        method_name="backdoor.tabpfn",
        method_params={
            "estimator": TabpfnEstimator,
            "model_type": model_type,
            "n_estimators": 4,
            "use_multi_gpu": False,
        },
    )

    assert estimate is not None and np.isfinite(float(estimate.value))
    # Classifier path selected
    assert estimate.estimator.tabpfn_model.resolved_model_type == "Classifier"
    # ATE sanity for binary outcome
    assert -1.0 <= float(estimate.value) <= 1.0
    assert float(estimate.value) > 0


def test_tabpfn_classifier_predict_proba_shape_and_bounds():
    data = _make_binary_data(beta=8.0)

    cm = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        common_causes=data["common_causes_names"],
        graph=data["gml_graph"],
    )

    estimand = cm.identify_effect(proceed_when_unidentifiable=True)

    # Fit estimator directly to access underlying model for probability checks
    est = TabpfnEstimator(
        estimand,
        confidence_intervals=False,
        method_params={"model_type": "classifier", "n_estimators": 4, "use_multi_gpu": False},
    )
    est.fit(data["df"])  # build encoders and model

    # Reuse its model/feature construction for predict_proba check
    features, model = est._build_model(data["df"])  # returns (design_matrix_with_intercept, wrapper)
    proba = model.predict_proba(features[:, 1:])  # remove intercept

    assert proba is not None
    assert proba.ndim == 2 and proba.shape[0] == features.shape[0]
    assert proba.shape[1] in (1, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


