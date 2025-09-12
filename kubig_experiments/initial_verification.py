import dowhy
import numpy as np
import pandas as pd
from dowhy import CausalModel
from dowhy.datasets import linear_dataset

# Custom estimator import
from dowhy.causal_estimators.linear_test_estimator import LinearTestEstimator

# Test 시나리오 정의
test_scenarios = [
    {"beta": 1, "num_common_causes": 2, "treatment_is_binary": False, "title": "기본 시나리오: 연속형 변수"},
    {"beta": 1, "num_common_causes": 2, "treatment_is_binary": True, "title": "이진 치료 변수 시나리오"},
    {"beta": 1, "num_common_causes": 5, "treatment_is_binary": False, "title": "공통 원인이 많은 시나리오"},
    {"beta": 0.5, "num_common_causes": 3, "treatment_is_binary": False, "title": "약한 인과 효과 시나리오"},
]

results = []

for scenario in test_scenarios:
    print(f"테스트 시작: {scenario['title']}")

    # 1. 가상 데이터 생성 (시나리오별로 파라미터 변경)
    data_dict = linear_dataset(
        beta=scenario["beta"],
        num_common_causes=scenario["num_common_causes"],
        num_instruments=0,
        num_effect_modifiers=0,
        num_samples=2000,
        treatment_is_binary=scenario["treatment_is_binary"],
    )
    df = data_dict['df']
    # print("가상 데이터셋 컬럼:", df.columns.tolist())
    
    # 2. Causal Model 설정
    model = CausalModel(
        data=df,
        graph=data_dict['gml_graph'],
        treatment=data_dict['treatment_name'],
        outcome=data_dict['outcome_name']
    )

    # 3. Identifiy the estimand
    identified_estimand = model.identify_effect(
        estimand_type="nonparametric-ate"
    )
    
    # 4.1 내장 LinearRegressionEstimator를 사용한 추정
    builtin_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    
    # 4.2 직접 구현한 LinearTestEstimator를 사용한 추정
    custom_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_test",
        method_params={"estimator": LinearTestEstimator}
    )
    
    results.append({
        "scenario": scenario['title'],
        "builtin_ate": builtin_estimate.value,
        "custom_ate": custom_estimate.value,
    })
    
# 모든 테스트 결과 출력
print("\n" + "=" * 80)
print("테스트 결과")
print("=" * 80)

for result in results:
    difference = abs(result['builtin_ate'] - result['custom_ate'])
    status = "성공 ✅" if difference < 1e-4 else "실패 ❌"
    
    print(f"[{result['scenario']}]")
    print(f"  - 내장 Estimator ATE: {result['builtin_ate']:.6f}")
    print(f"  - 커스텀 Estimator ATE: {result['custom_ate']:.6f}")
    print(f"  - 결과 일치 여부: {status}")
    print("-" * 30)