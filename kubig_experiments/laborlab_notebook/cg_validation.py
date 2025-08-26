# 파일명: run_causal_refutation.py

import argparse
import pandas as pd
import dowhy
from dowhy import CausalModel

def run_dowhy_refutation(model):
    """
    DoWhy의 내장 검증(refutation) 기능을 사용하여 모델과 추정치를 검증합니다.
    """
    print("\n--- DoWhy 내장 검증(Refutation) 시작 ---")

    # 1. 인과 효과 식별 (Identify)
    # 그래프 구조에 기반하여 인과 효과를 통계적으로 어떻게 추정할지 결정합니다.
    try:
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("\n[단계 1] 인과 효과 식별 완료.")
        print(identified_estimand)
    except Exception as e:
        print(f"  - 오류: 인과 효과 식별 중 오류 발생 - {e}")
        return

    # 2. 인과 효과 추정 (Estimate)
    # 식별된 방법에 따라 실제 데이터로 인과 효과의 크기를 추정합니다.
    # 여기서는 간단한 선형회귀 모델을 사용합니다.
    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True
        )
        print("\n[단계 2] 인과 효과 추정 완료.")
        print(f"  - 추정된 인과 효과 (ATE): {estimate.value:.4f}")
    except Exception as e:
        print(f"  - 오류: 인과 효과 추정 중 오류 발생 - {e}")
        return

    # 3. 검증 (Refute)
    # 추정된 인과 효과가 강건한지 다양한 방법으로 테스트합니다.
    print("\n[단계 3] 추정치에 대한 강건성 검증 수행.")

    # 검증 1: 가상 원인 테스트 (Placebo Treatment)
    try:
        refute_placebo = model.refute_estimator(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute" # 기존 treatment를 무작위로 섞어 가상 원인 생성
        )
        print("\n[검증 1] 가상 원인(Placebo Treatment) 테스트 결과:")
        print(f"  - 추정된 효과: {refute_placebo.estimated_effect:.4f}")
        print(f"  - P-value: {refute_placebo.p_value:.4f}")
        if refute_placebo.p_value > 0.05:
            print("  - 해석: 가상 원인의 효과가 통계적으로 0과 다르지 않습니다. 모델이 강건하다는 긍정적인 신호입니다. 👍")
        else:
            print("  - 해석: 가상 원인이 유의미한 효과를 보입니다. 모델 설정을 재검토해야 할 수 있습니다. 👎")
    except Exception as e:
        print(f"\n  - 오류 (가상 원인 테스트): {e}")

    # 검증 2: 관측되지 않은 공통 원인 추가
    try:
        refute_unobserved = model.refute_estimator(
            identified_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            confounders_effect_on_treatment="binary_flip", # 0.1의 확률로 treatment를 바꿈
            confounders_effect_on_outcome=0.1,
            effect_strength_on_treatment=0.1
        )
        print("\n[검증 2] 미관측 공통 원인(Unobserved Common Cause) 추가 테스트 결과:")
        print(f"  - 기존 추정치: {estimate.value:.4f}")
        print(f"  - 새로운 추정치 (미관측 교란변수 추가 후): {refute_unobserved.new_effect:.4f}")
        print("  - 해석: 새로운 추정치가 기존 추정치와 큰 차이가 없다면, 모델이 미관측 교란변수에 상대적으로 강건함을 의미합니다.")
    except Exception as e:
        print(f"\n  - 오류 (미관측 공통 원인 테스트): {e}")

    print("\n--- 검증 완료 ---")


def main():
    parser = argparse.ArgumentParser(description="DoWhy를 사용하여 인과 모델을 생성하고 내장 검증 기능을 수행합니다.")
    parser.add_argument("--graph", type=str, required=True, help="GML 형식의 인과 그래프 파일 경로")
    parser.add_argument("--data", type=str, required=True, help="CSV 형식의 데이터 파일 경로")
    args = parser.parse_args()

    print("입력된 파일:")
    print(f"  - 그래프 파일: {args.graph}")
    print(f"  - 데이터 파일: {args.data}")

    with open(args.graph, 'r', encoding='utf-8') as f:
        causal_graph_gml = f.read()
    
    df = pd.read_csv(args.data)

    try:
        model = CausalModel(
            data=df,
            treatment='ui_received',
            outcome='employed_within_180d',
            graph=causal_graph_gml
        )
        print("\n✅ Causal Model을 성공적으로 생성했습니다.")
        # DoWhy 내장 검증 함수 호출
        run_dowhy_refutation(model)
    except Exception as e:
        # 데이터 타입 문제 등을 처리하기 위한 예외 처리
        print(f"\n❌ 모델 생성 또는 검증 중 오류 발생: {e}")
        print("GML의 변수명과 데이터의 컬럼명이 일치하는지, 데이터 타입이 분석에 적합한지 확인하세요.")
        print("(예: 'ui_received'와 'employed_within_180d'는 숫자형(0/1)이어야 할 수 있습니다.)")

if __name__ == "__main__":
    main()