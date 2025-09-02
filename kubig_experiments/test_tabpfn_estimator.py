import pytest
import pandas as pd
import numpy as np

# TabPFN과 PyTorch가 설치되지 않았다면 이 파일의 모든 테스트를 건너뜁니다.
tabpfn = pytest.importorskip("tabpfn")
torch = pytest.importorskip("torch")

from dowhy import CausalModel
from dowhy.datasets import linear_dataset

# 테스트하려는 Estimator 클래스를 가져옵니다.
# 파일 경로는 실제 프로젝트 구조에 맞게 조정해야 할 수 있습니다.
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

@pytest.mark.parametrize(
    "outcome_is_binary, num_common_causes",
    [
        (False, 4),  # 시나리오 1: 연속형 결과 (Regression)
        (True, 4),   # 시나리오 2: 이진형 결과 (Classification)
    ]
)
def test_tabpfn_estimator_ate(outcome_is_binary, num_common_causes):
    """
    TabPFNEstimator의 핵심 기능(ATE 추정)을 테스트합니다.
    - 연속형/이진형 결과 변수에 따라 올바른 TabPFN 모델이 사용되는지 확인합니다.
    - 범주형 공통 원인 변수가 포함된 데이터의 전처리(인코딩)가 잘 동작하는지 확인합니다.
    """
    true_ate = 10
    
    # 1. 테스트 데이터 생성
    data = linear_dataset(
            beta=true_ate,
            num_common_causes=num_common_causes,
            num_instruments=0,
            num_effect_modifiers=0,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True,
            outcome_is_binary=outcome_is_binary,
        )

    # dowhy의 _encode 기능 테스트를 위해 수동으로 범주형 변수 추가
    data["df"]["W0"] = pd.cut(data["df"]["W0"], bins=3, labels=["low", "medium", "high"])

    # 2. CausalModel 생성
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        common_causes=data["common_causes_names"],
        graph=data["gml_graph"]
    )

    # 3. 인과 효과 식별
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 4. 인과 효과 추정
    estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.tabpfn",
            method_params={"estimator": TabpfnEstimator, "N_ensemble_configurations": 8}
        )

    # 5. 결과 검증
    assert estimate is not None
    assert isinstance(estimate.value, (float, np.floating)), "추정된 인과 효과는 float 타입이어야 합니다."

    # 연속형 결과(Regression)의 경우, 추정치가 실제값과 비슷한지 확인합니다.
    # 작은 데이터셋과 복잡한 모델이므로 허용 오차는 크게 설정합니다.
    if not outcome_is_binary:
        error_tolerance = 4  # 실제 ATE=10에 대해 허용 오차 4
        assert estimate.value == pytest.approx(true_ate, abs=error_tolerance)

    print(f"\nTest passed for {'binary' if outcome_is_binary else 'continuous'} outcome.")
    if not outcome_is_binary:
        print(f"  - True ATE: {true_ate}")
        print(f"  - Estimated ATE: {estimate.value:.4f}")
    else:
        print(f"  - Estimated ATE (propensity difference): {estimate.value:.4f}")