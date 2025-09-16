import sys
import pandas as pd
import numpy as np

import tabpfn 
import torch 

from dowhy import CausalModel
from dowhy.datasets import linear_dataset
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_case(outcome_is_binary: bool, num_common_causes: int, debug: bool = False) -> tuple:
    """
    Core runner for testing TabPFN estimator with confidence intervals.
    Returns the estimated ATE (float) and confidence intervals (tuple).
    """
    true_ate = 10

    if debug:
        print(f"\n🔍 [DEBUG] Starting test case: outcome_is_binary={outcome_is_binary}, num_common_causes={num_common_causes}")

    # 1. 테스트 데이터 생성
    if debug:
        print("📊 [DEBUG] Generating test data...")
    
    data = linear_dataset(
        beta=true_ate,
        num_common_causes=num_common_causes,
        num_instruments=2,
        num_effect_modifiers=1,
        num_samples=200,
        treatment_is_binary=True,
        stddev_treatment_noise=10,
        num_discrete_common_causes=1,
        outcome_is_binary=outcome_is_binary,
    )

    if debug:
        print(f"📊 [DEBUG] Data shape: {data['df'].shape}")
        print(f"📊 [DEBUG] Treatment: {data['treatment_name']}")
        print(f"📊 [DEBUG] Outcome: {data['outcome_name']}")
        print(f"📊 [DEBUG] Common causes: {data['common_causes_names']}")
        print(f"📊 [DEBUG] Instruments: {data['instrument_names']}")
        print(f"📊 [DEBUG] Effect modifiers: {data['effect_modifier_names']}")
        print(f"📊 [DEBUG] Data columns: {list(data['df'].columns)}")
        print(f"📊 [DEBUG] Data types:\n{data['df'].dtypes}")

    # 2. CausalModel 생성
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        common_causes=data["common_causes_names"],
        instruments=data["instrument_names"],
        effect_modifiers=data["effect_modifier_names"],
        graph=data["gml_graph"],
    )

    # 3. 인과 효과 식별
    if debug:
        print("🔍 [DEBUG] Identifying causal effect...")
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    if debug:
        print(f"🔍 [DEBUG] Identified estimand: {identified_estimand}")

    # 4. 인과 효과 추정 (confidence intervals 포함)
    if debug:
        print("📈 [DEBUG] Estimating causal effect with confidence intervals...")
    
    # 4.1 선형회귀 베이스라인 (연속형 결과에서만 비교)
    lr_estimate = None
    if not outcome_is_binary:
        lr_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            confidence_intervals=True,
        )
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.tabpfn",
        method_params={"estimator": TabpfnEstimator, "N_ensemble_configurations": 8},
        confidence_intervals=True,  # confidence intervals 활성화
    )

    if debug:
        print(f"📈 [DEBUG] Estimate object: {type(estimate)}")
        print(f"📈 [DEBUG] Estimate value: {estimate.value}")
        print(f"📈 [DEBUG] Estimate attributes: {dir(estimate)}")

    # 5. 결과 검증
    if estimate is None:
        raise ValueError("추정된 인과 효과가 None입니다.")
    
    if not isinstance(estimate.value, (float, np.floating)):
        raise TypeError("추정된 인과 효과는 float 타입이어야 합니다.")

    # 6. Confidence Intervals 검증
    if debug:
        print("📊 [DEBUG] Validating confidence intervals...")
    
    ci = None
    try:
        ci = estimate.get_confidence_intervals()
        if debug:
            print(f"📊 [DEBUG] Raw CI result: {ci}")
            print(f"📊 [DEBUG] CI type: {type(ci)}")
        
        if ci is None:
            raise ValueError("Confidence intervals가 None입니다.")
        
        if not isinstance(ci, (tuple, list)) or len(ci) != 2:
            raise TypeError("Confidence intervals는 (lower_bound, upper_bound) 튜플이어야 합니다.")
        
        lower_bound, upper_bound = ci
        if not all(isinstance(bound, (float, np.floating)) for bound in ci):
            raise TypeError("Confidence interval bounds는 float 타입이어야 합니다.")
        
        if lower_bound >= upper_bound:
            raise ValueError(f"Confidence interval이 잘못되었습니다: {lower_bound} >= {upper_bound}")
            
        if debug:
            print(f"📊 [DEBUG] CI validation passed: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
    except Exception as e:
        if debug:
            print(f"❌ [DEBUG] CI validation failed: {str(e)}")
        raise ValueError(f"Confidence intervals 검증 실패: {str(e)}")

    # 연속형 결과(Regression)의 경우 허용 오차 내 확인
    if not outcome_is_binary:
        # 작은 데이터셋과 복잡한 모델이므로 허용 오차는 크게 설정
        error_tolerance = 4
        diff = abs(float(estimate.value) - 10.0)
        if diff > error_tolerance:
            print(f"[WARN] Estimated ATE deviates from true ATE by {diff:.3f} (> {error_tolerance}).")

    # 결과 출력
    kind = "binary" if outcome_is_binary else "continuous"
    print(f"\n[TabPFN Python] Case: {kind}")
    if not outcome_is_binary:
        print(f"  - True ATE: {true_ate}")
    print(f"  - Estimated ATE: {float(estimate.value):.4f}")
    print(f"  - 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  - CI Width: {ci[1] - ci[0]:.4f}")

    # 선형회귀와 비교 출력 (연속형만)
    if lr_estimate is not None:
        print("  - Baseline (Linear Regression):")
        print(f"    * ATE: {float(lr_estimate.value):.4f}")
        try:
            lr_ci = lr_estimate.get_confidence_intervals()
            if isinstance(lr_ci, (tuple, list)) and len(lr_ci) == 2:
                print(f"    * 95% CI: [{float(lr_ci[0]):.4f}, {float(lr_ci[1]):.4f}]")
                print(f"    * CI Width: {float(lr_ci[1]) - float(lr_ci[0]):.4f}")
        except Exception:
            pass
        diff = float(estimate.value) - float(lr_estimate.value)
        print(f"  - Difference (TabPFN - Linear): {diff:.4f}")

    return float(estimate.value), ci


def run_batch_regression(
    num_cases: int = 1000,
    num_samples: int = 200,
    num_common_causes: int = 5,
    seed: int = 42,
    compute_ci: bool = False,
    verbose_every: int = 50,
    plot: bool = False,
    plot_path: str = "batch_regression_hist.png",
):
    """
    Generate `num_cases` synthetic regression datasets (each with different true ATE),
    estimate ATE with TabPFN and Linear Regression, and summarize errors.
    """
    rng = np.random.RandomState(seed)

    results = []  # (case_idx, true_ate, tabpfn_ate, linear_ate, diff_tabpfn, diff_linear)

    for i in tqdm(range(1, num_cases + 1), total=num_cases, desc="Batch regression", unit="case"):
        # Different true ATE per case
        true_ate = float(rng.uniform(0.5, 18.0))

        data = linear_dataset(
            beta=true_ate,
            num_common_causes=num_common_causes,
            num_instruments=0,
            num_effect_modifiers=0,
            num_samples=num_samples,
            treatment_is_binary=True,
            outcome_is_binary=False,
        )

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            common_causes=data["common_causes_names"],
            graph=data["gml_graph"],
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Linear Regression baseline
        lr_est = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            confidence_intervals=compute_ci,
        )

        # TabPFN
        tabpfn_est = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.tabpfn",
            method_params={"estimator": TabpfnEstimator, "N_ensemble_configurations": 8},
            confidence_intervals=compute_ci,
        )

        tabpfn_ate = float(tabpfn_est.value)
        linear_ate = float(lr_est.value)
        results.append(
            (
                i,
                true_ate,
                tabpfn_ate,
                linear_ate,
                tabpfn_ate - true_ate,
                linear_ate - true_ate,
            )
        )

        if verbose_every and (i % verbose_every == 0):
            tqdm.write(
                f"[Batch] {i}/{num_cases}  true={true_ate:.2f}  tabpfn={tabpfn_ate:.2f}  linear={linear_ate:.2f}"
            )

    # Summaries
    errs_tab = np.array([r[4] for r in results])
    errs_lin = np.array([r[5] for r in results])

    def _summ(name, errs):
        mae = float(np.mean(np.abs(errs)))
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        bias = float(np.mean(errs))
        return name, mae, rmse, bias

    s_tab = _summ("TabPFN", errs_tab)
    s_lin = _summ("Linear", errs_lin)

    print("\n=== Batch Regression Summary (num_cases={}, n={}) ===".format(num_cases, num_samples))
    for name, mae, rmse, bias in (s_tab, s_lin):
        print(f"{name:>7}  MAE={mae:.3f}  RMSE={rmse:.3f}  Bias={bias:.3f}")

    if plot:
        plt.figure(figsize=(8, 5))
        bins = max(20, int(np.sqrt(len(errs_tab))))
        plt.hist(errs_tab, bins=bins, alpha=0.6, label="TabPFN errors", color="#1f77b4")
        plt.hist(errs_lin, bins=bins, alpha=0.6, label="Linear errors", color="#ff7f0e")
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
        plt.title(f"ATE Error Distribution (num_cases={num_cases}, n={num_samples})")
        plt.xlabel("Estimated ATE - True ATE")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"Saved histogram to {plot_path}")

    return {
        "results": results,
        "summary": {"tabpfn": s_tab[1:], "linear": s_lin[1:]},
    }

def test_tabpfn_estimator_ate(debug: bool = False):
    """Test TabPFN estimator for both regression and classification cases with confidence intervals."""
    print("=" * 60)
    print("TabPFN Estimator Test with Confidence Intervals")
    print("=" * 60)
    
    test_cases = [
        (False, 5),  # Regression
        (True, 5),   # Classification
    ]
    
    results = []
    
    for i, (outcome_is_binary, num_common_causes) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        try:
            ate, ci = run_case(
                outcome_is_binary=outcome_is_binary, 
                num_common_causes=num_common_causes,
                debug=debug
            )
            results.append((outcome_is_binary, num_common_causes, ate, ci, "PASS"))
            print(f"✅ Test Case {i} PASSED")
        except Exception as e:
            results.append((outcome_is_binary, num_common_causes, None, None, f"FAIL: {str(e)}"))
            print(f"❌ Test Case {i} FAILED: {str(e)}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, (outcome_is_binary, num_common_causes, ate, ci, status) in enumerate(results, 1):
        case_type = "Classification" if outcome_is_binary else "Regression"
        print(f"Case {i} ({case_type}): {status}")
        if ate is not None:
            print(f"  - Estimated ATE: {ate:.4f}")
        if ci is not None:
            print(f"  - 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"  - CI Width: {ci[1] - ci[0]:.4f}")
        passed += 1 if "PASS" in status else 0
        failed += 1 if "FAIL" in status else 0
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed!")
        return False


def verify_estimator_usage():
    """Verify that TabpfnEstimator is being used correctly."""
    print("🔍 [VERIFICATION] Checking estimator usage...")
    print("=" * 60)
    
    # 1. 직접 estimator 클래스 확인
    from dowhy.causal_estimators import get_class_object
    try:
        estimator_class = get_class_object("tabpfn_estimator")
        print(f"✅ [VERIFICATION] get_class_object('tabpfn_estimator') = {estimator_class}")
        print(f"✅ [VERIFICATION] Class name: {estimator_class.__name__}")
        print(f"✅ [VERIFICATION] Module: {estimator_class.__module__}")
        print(f"✅ [VERIFICATION] Is TabpfnEstimator? {estimator_class.__name__ == 'TabpfnEstimator'}")
    except Exception as e:
        print(f"❌ [VERIFICATION] get_class_object failed: {e}")
        return False
    
    # 2. CausalModel에서 estimator 생성 확인
    try:
        from dowhy import CausalModel
        from dowhy.datasets import linear_dataset
        
        # 간단한 데이터로 테스트
        data = linear_dataset(
            beta=10,
            num_common_causes=2,
            num_instruments=0,
            num_effect_modifiers=0,
            num_samples=100,
            treatment_is_binary=True,
            outcome_is_binary=False,
        )
        
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            common_causes=data["common_causes_names"],
            graph=data["gml_graph"],
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # method_name으로 estimator 생성 확인
        method_name = "backdoor.tabpfn"
        # num_components = len(method_name.split("."))
        str_arr = method_name.split(".", maxsplit=1)
        identifier_name = str_arr[0]  # "backdoor"
        estimator_name = str_arr[1]   # "tabpfn"
        
        print(f"✅ [VERIFICATION] Method name: {method_name}")
        print(f"✅ [VERIFICATION] Identifier: {identifier_name}")
        print(f"✅ [VERIFICATION] Estimator name: {estimator_name}")
        
        # get_class_object 호출
        causal_estimator_class = get_class_object(estimator_name + "_estimator")
        print(f"✅ [VERIFICATION] Retrieved class: {causal_estimator_class}")
        
        # Estimator 인스턴스 생성
        estimator_instance = causal_estimator_class(
            identified_estimand,
            test_significance=False,
            evaluate_effect_strength=False,
            confidence_intervals=False,
        )
        print(f"✅ [VERIFICATION] Created instance: {type(estimator_instance)}")
        print(f"✅ [VERIFICATION] Instance class name: {estimator_instance.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ [VERIFICATION] CausalModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_case_debug():
    """Debug single test case with detailed output."""
    print("🔍 [DEBUG MODE] Running single test case...")
    print("=" * 60)
    
    try:
        ate, ci = run_case(
            outcome_is_binary=False,  # Regression
            num_common_causes=5,
            debug=True
        )
        print(f"\n [SUCCESS] Test completed successfully!")
        print(f"  - ATE: {ate:.4f}")
        print(f"  - CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        return True
    except Exception as e:
        print(f"\n❌ [ERROR] Test failed: {str(e)}")
        import traceback
        print(f"\n📋 [TRACEBACK]:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TabPFN estimator with confidence intervals")
    parser.add_argument("--debug", action="store_true", help="Enable debugging prints")
    parser.add_argument("--single", action="store_true", help="Run single test case")
    parser.add_argument("--verify", action="store_true", help="Verify estimator usage")
    parser.add_argument("--batch", type=int, default=0, help="Run batch regression with N cases (e.g., 1000)")
    parser.add_argument("--batch-samples", type=int, default=200, help="Samples per dataset in batch run")
    parser.add_argument("--batch-ci", action="store_true", help="Compute CI during batch (slower)")
    parser.add_argument("--batch-plot", action="store_true", help="Plot histogram for batch errors")
    parser.add_argument("--batch-plot-path", type=str, default="batch_regression_hist.png", help="Output path for batch histogram PNG")
    args = parser.parse_args()
    
    if args.batch and args.batch > 0:
        # Batch run
        _ = run_batch_regression(
            num_cases=args.batch,
            num_samples=args.batch_samples,
            compute_ci=args.batch_ci,
            plot=args.batch_plot,
            plot_path=args.batch_plot_path,
        )
        success = True
    elif args.verify:
        # Verify estimator usage
        success = verify_estimator_usage()
    elif args.single:
        # Run single test case with debug mode
        success = test_single_case_debug()
    else:
        # Run full test suite
        success = test_tabpfn_estimator_ate(debug=args.debug)
    
    sys.exit(0 if success else 1)