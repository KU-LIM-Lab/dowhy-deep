"""
DoWhy 인과효과 추정 및 검증 모듈

이 모듈은 인과효과 추정, 검증 테스트, 민감도 분석 등의 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime
import os
import sys
import pickle
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from dowhy.causal_estimators.regression_estimator import RegressionEstimator

# 로컬 DoWhy 라이브러리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# DoWhy 내부 함수 임포트
from dowhy.causal_estimator import estimate_effect as dowhy_estimate_effect

def log_estimation_results(logger, estimate, method_name):
    """
    추정 결과를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        estimate: 추정된 인과효과 객체
        method_name (str): 추정 방법명
    """
    logger.info("="*60)
    logger.info("인과 효과 추정 결과")
    logger.info("="*60)
    logger.info(f"추정 방법: {method_name}")
    logger.info(f"추정된 인과 효과 (ATE): {estimate.value:.6f}")
    
    if hasattr(estimate, 'p_value') and estimate.p_value is not None:
        logger.info(f"P-value: {estimate.p_value:.6f}")
        significance = "유의함" if estimate.p_value <= 0.05 else "유의하지 않음"
        logger.info(f"통계적 유의성: {significance}")
    
    # 추정치의 신뢰구간이 있다면 로깅
    if hasattr(estimate, 'confidence_intervals'):
        logger.info(f"신뢰구간: {estimate.confidence_intervals}")


def predict_conditional_expectation(estimate, data_df, treatment_value=None, logger=None):
    """
    E(Y|A, X) 조건부 기대값 예측
    
    Args:
        estimate: CausalEstimate 객체
        data_df: 예측할 데이터프레임
        treatment_value: 처치 값 (None이면 실제 값 사용)
        logger: 로거 객체
    
    Returns:
        tuple: (metrics_dict, data_df_with_predictions)
            - metrics_dict: {'accuracy': float, 'f1_score': float, 'auc': float} 또는 None
            - data_df_with_predictions: outcome 열에 예측값이 채워진 데이터프레임
    """
    if not hasattr(estimate, 'estimator'):
        raise ValueError("estimate.estimator가 없습니다. estimate_causal_effect를 먼저 실행하세요.")
    
    estimator = estimate.estimator
    if not isinstance(estimator, RegressionEstimator):
        raise ValueError(f"{type(estimator).__name__}는 예측을 지원하지 않습니다.")
    
    if logger:
        logger.info(f"E(Y|A, X) 예측 시작: {len(data_df)}개")
        if treatment_value is not None:
            logger.info(f"처치 값: {treatment_value}")
    
    try:
        # 데이터프레임 복사 (원본 보호)
        data_df_clean = data_df.copy()
        
        # treatment와 outcome 변수는 반드시 유지해야 함
        # _treatment_name과 _outcome_name은 리스트일 수 있음 (private attribute)
        treatment_var = estimate._treatment_name[0] if isinstance(estimate._treatment_name, list) else estimate._treatment_name
        outcome_var = estimate._outcome_name[0] if isinstance(estimate._outcome_name, list) else estimate._outcome_name

        # treatment 변수가 있는지 확인
        if treatment_var not in data_df_clean.columns:
            raise ValueError(f"Treatment 변수 '{treatment_var}'가 데이터에 없습니다. 사용 가능한 컬럼: {list(data_df_clean.columns)}")
        
        # predict_fn을 사용하는 방식으로 예측 (RegressionEstimator의 표준 인터페이스)
        if treatment_value is not None:
            predictions = estimator.interventional_outcomes(data_df_clean, treatment_value)
        else:
            predictions = estimator.predict(data_df_clean)
        
        predictions_series = pd.Series(predictions, index=data_df_clean.index)
        
        # 데이터프레임 복사 후 예측값 채우기
        result_df = data_df_clean.copy()
        # _outcome_name은 리스트일 수 있음
        outcome_name = estimate._outcome_name[0] if isinstance(estimate._outcome_name, list) else estimate._outcome_name
        result_df[outcome_name] = predictions_series
        
        # 실제 Y 값과 비교하여 메트릭 계산
        metrics = {}
        if outcome_name in data_df_clean.columns:
            actual_y = data_df_clean[outcome_name]
            # actual_y가 숫자 타입인지 확인
            if not pd.api.types.is_numeric_dtype(actual_y):
                actual_y = pd.to_numeric(actual_y, errors='coerce')
            
            # NaN 제거
            valid_mask = ~(pd.isna(actual_y) | pd.isna(predictions_series))
            if valid_mask.sum() > 0:
                actual_y_clean = actual_y[valid_mask]
                predictions_clean = predictions_series[valid_mask]
                
                # 이진 분류인지 확인 (0과 1만 있는지)
                unique_values = set(actual_y_clean.dropna().unique())
                is_binary = len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values if not pd.isna(v))
                
                if is_binary:
                    # 이진 분류 메트릭 계산
                    predicted_classes = (predictions_clean > 0.5).astype(int)
                    metrics['accuracy'] = accuracy_score(actual_y_clean, predicted_classes)
                    metrics['f1_score'] = f1_score(actual_y_clean, predicted_classes, zero_division=0)
                    
                    # AUC 계산 (예측 확률 사용)
                    try:
                        # predictions가 확률인지 확인 (0~1 범위)
                        if predictions_clean.min() >= 0 and predictions_clean.max() <= 1:
                            metrics['auc'] = roc_auc_score(actual_y_clean, predictions_clean)
                        else:
                            # 확률이 아니면 sigmoid 변환 시도
                            from scipy.special import expit
                            prob_predictions = expit(predictions_clean)
                            metrics['auc'] = roc_auc_score(actual_y_clean, prob_predictions)
                    except Exception as e:
                        if logger:
                            logger.warning(f"AUC 계산 실패: {e}")
                        metrics['auc'] = None
                    
                    if logger:
                        logger.info(f"예측 완료: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics.get('auc', 'N/A')}")
                else:
                    # 연속형 변수인 경우
                    metrics['accuracy'] = None
                    metrics['f1_score'] = None
                    metrics['auc'] = None
                    if logger:
                        logger.info(f"예측 완료: 평균={predictions_clean.mean():.6f} (연속형 변수)")
            else:
                if logger:
                    logger.warning(f"유효한 데이터가 없어 메트릭을 계산할 수 없습니다.")
        else:
            if logger:
                logger.warning(f"실제 Y 값({outcome_name})을 찾을 수 없어 메트릭을 계산할 수 없습니다.")
        
        return metrics, result_df
        
    except Exception as e:
        if logger:
            logger.error(f"예측 실패: {e}")
        raise

def estimate_causal_effect(model, identified_estimand, estimator, logger=None):
    """인과효과를 추정하는 함수"""
    if logger:
        logger.info("="*60)
        logger.info("인과효과 추정 시작")
        logger.info("="*60)
    
    method_map = {
        'linear_regression': 'backdoor.linear_regression',
        'tabpfn': 'backdoor.tabpfn',
        'propensity_score': 'backdoor.propensity_score_stratification',
        'instrumental_variable': 'iv.instrumental_variable'
    }
    
    method = method_map.get(estimator, 'backdoor.linear_regression')
    
    if logger:
        logger.info(f"사용할 추정 방법: {method}")
        logger.info(f"요청된 추정기: {estimator}")
    
    try:
        # TabPFN의 경우 새 버전 사용 (표준 인터페이스)
        if estimator == 'tabpfn':
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                test_significance=True,
                method_params={
                    "n_estimators": 8,
                    "model_type": "auto"
                }
            )
        else:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                test_significance=True
            )
        
        if logger:
            logger.info("✅ 인과효과 추정 성공")
            logger.info(f"추정된 인과 효과 (ATE): {estimate.value:.6f}")
            if hasattr(estimate, 'p_value') and estimate.p_value is not None:
                logger.info(f"P-value: {estimate.p_value:.6f}")
                significance = "유의함" if estimate.p_value <= 0.05 else "유의하지 않음"
                logger.info(f"통계적 유의성: {significance}")
            
            # 신뢰구간 정보
            if hasattr(estimate, 'confidence_intervals'):
                logger.info(f"신뢰구간: {estimate.confidence_intervals}")
        
        return estimate
        
    except Exception as e:
        if logger:
            logger.error(f"❌ 인과효과 추정 실패: {e}")
        raise

def calculate_refutation_pvalue(refutation_result, test_type="placebo"):
    """
    Refutation 테스트 결과의 p-value를 계산합니다.
    
    Args:
        refutation_result: CausalRefutation 객체
        test_type: 테스트 타입 ("placebo", "unobserved", "subset", "dummy")
    
    Returns:
        float: p-value (계산 불가능한 경우 None)
    """
    try:
        # refutation_result에서 refutation_results 속성 확인
        if hasattr(refutation_result, 'refutation_results') and refutation_result.refutation_results:
            # refutation_results는 리스트일 수 있음
            results = refutation_result.refutation_results
            if isinstance(results, list) and len(results) > 0:
                # 각 결과에서 effect 값 추출
                effects = []
                for r in results:
                    if hasattr(r, 'value'):
                        effects.append(r.value)
                    elif isinstance(r, dict) and 'value' in r:
                        effects.append(r['value'])
                
                if len(effects) > 1:
                    # 효과들이 0과 유의하게 다른지 t-test
                    t_stat, p_value = stats.ttest_1samp(effects, 0)
                    return p_value
        
        # refutation_results가 없으면 new_effect와 estimated_effect 비교
        if hasattr(refutation_result, 'new_effect') and hasattr(refutation_result, 'estimated_effect'):
            if test_type == "placebo" or test_type == "dummy":
                # new_effect가 0과 유의하게 다른지 (단일 값이므로 직접 비교 불가)
                # 대신 new_effect의 절대값이 작으면 통과로 간주
                return None
            elif test_type == "unobserved" or test_type == "subset":
                # new_effect와 estimated_effect가 유의하게 다른지
                # 단일 값이므로 직접 t-test 불가, 차이의 절대값으로 판단
                return None
        
        return None
    except Exception as e:
        return None


def log_validation_results(logger, validation_results):
    """
    검증 결과를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        validation_results (dict): 검증 결과 딕셔너리
    """
    logger.info("="*60)
    logger.info("검증 결과 요약")
    logger.info("="*60)
    
    # 가상 원인 테스트
    if validation_results.get('placebo'):
        placebo = validation_results['placebo']
        effect_change = abs(placebo.new_effect - placebo.estimated_effect)
        p_value = calculate_refutation_pvalue(placebo, "placebo")
        # 효과 변화가 작으면 통과 (0.01 이하)
        status = "통과" if effect_change < 0.01 else "실패"
        logger.info(f"가상 원인 테스트: {status}")
        logger.info(f"  - 기존 추정치: {placebo.estimated_effect:.6f}")
        logger.info(f"  - 가상처치 후 추정치: {placebo.new_effect:.6f}")
        logger.info(f"  - 효과 변화: {effect_change:.6f}")
        if p_value is not None:
            logger.info(f"  - P-value: {p_value:.6f}")
            logger.info(f"  - 통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
    
    # 미관측 교란 테스트
    if validation_results.get('unobserved'):
        unobserved = validation_results['unobserved']
        change_rate = abs(unobserved.new_effect - unobserved.estimated_effect) / abs(unobserved.estimated_effect) if abs(unobserved.estimated_effect) > 0 else float('inf')
        p_value = calculate_refutation_pvalue(unobserved, "unobserved")
        # 변화율이 20% 미만이면 강건함
        status = "강건함" if change_rate < 0.2 else "민감함"
        logger.info(f"미관측 교란 테스트: {status}")
        logger.info(f"  - 기존 추정치: {unobserved.estimated_effect:.6f}")
        logger.info(f"  - 교란 추가 후 추정치: {unobserved.new_effect:.6f}")
        logger.info(f"  - 변화율: {change_rate:.2%}")
        if p_value is not None:
            logger.info(f"  - P-value: {p_value:.6f}")
            logger.info(f"  - 통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
    
    # 부분표본 안정성 테스트
    if validation_results.get('subset'):
        subset = validation_results['subset']
        effect_change = abs(subset.new_effect - subset.estimated_effect)
        p_value = calculate_refutation_pvalue(subset, "subset")
        # 효과 변화가 작으면 통과 (10% 이하)
        change_rate = abs(subset.estimated_effect) > 0 and abs(effect_change / subset.estimated_effect) or float('inf')
        status = "통과" if change_rate < 0.1 else "실패"
        logger.info(f"부분표본 안정성 테스트: {status}")
        logger.info(f"  - 기존 추정치: {subset.estimated_effect:.6f}")
        logger.info(f"  - 부분표본 추정치: {subset.new_effect:.6f}")
        logger.info(f"  - 효과 변화: {effect_change:.6f}")
        if p_value is not None:
            logger.info(f"  - P-value: {p_value:.6f}")
            logger.info(f"  - 통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
    
    # 더미 결과 테스트
    if validation_results.get('dummy'):
        dummy = validation_results['dummy']
        p_value = calculate_refutation_pvalue(dummy, "dummy")
        # new_effect가 0에 가까우면 통과 (0.01 이하)
        status = "통과" if abs(dummy.new_effect) < 0.01 else "실패"
        logger.info(f"더미 결과 테스트: {status}")
        logger.info(f"  - 더미 결과 추정치: {dummy.new_effect:.6f}")
        if p_value is not None:
            logger.info(f"  - P-value: {p_value:.6f}")
            logger.info(f"  - 통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")

def run_validation_tests(model, identified_estimand, estimate, logger=None):
    """검증 테스트를 실행하는 함수 (4개 테스트 모두 포함)"""
    if logger:
        logger.info("="*60)
        logger.info("검증 테스트 실행 시작 (4개 테스트)")
        logger.info("="*60)
    
    validation_results = {}
    
    # 1. 가상 원인 테스트 (Placebo Treatment)
    print("1️⃣ 가상 원인 테스트 실행 중...")
    if logger:
        logger.info("1️⃣ 가상 원인 테스트 실행 중...")
    
    try:
        refute_placebo = model.refute_estimate(
            identified_estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=100
        )
        validation_results['placebo'] = refute_placebo
        
        effect_change = abs(refute_placebo.new_effect - refute_placebo.estimated_effect)
        p_value = calculate_refutation_pvalue(refute_placebo, "placebo")
        status = "통과" if effect_change < 0.01 else "실패"
        
        print(f"✅ 가상 원인 테스트 완료: {status}")
        print(f"   기존 추정치: {refute_placebo.estimated_effect:.6f}")
        print(f"   가상처치 후 추정치: {refute_placebo.new_effect:.6f}")
        print(f"   효과 변화: {effect_change:.6f}")
        if p_value is not None:
            print(f"   P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
        
        if logger:
            logger.info("✅ 가상 원인 테스트 성공")
            logger.info(f"테스트 결과: {status}")
            logger.info(f"기존 추정치: {refute_placebo.estimated_effect:.6f}")
            logger.info(f"가상처치 후 추정치: {refute_placebo.new_effect:.6f}")
            logger.info(f"효과 변화: {effect_change:.6f}")
            if p_value is not None:
                logger.info(f"P-value: {p_value:.6f}")
                logger.info(f"통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
            
    except Exception as e:
        validation_results['placebo'] = None
        print(f"❌ 가상 원인 테스트 실패: {e}")
        if logger:
            logger.error(f"❌ 가상 원인 테스트 실패: {e}")
    
    # 2. 미관측 교란 테스트 (Add Unobserved Common Cause)
    print("2️⃣ 미관측 교란 테스트 실행 중...")
    if logger:
        logger.info("2️⃣ 미관측 교란 테스트 실행 중...")
    
    try:
        refute_unobserved = model.refute_estimate(
            identified_estimand, estimate,
            method_name="add_unobserved_common_cause",
            confounders_effect_on_treatment="binary_flip",
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=0.10,
            effect_strength_on_outcome=0.10,
            num_simulations=100
        )
        validation_results['unobserved'] = refute_unobserved
        
        change_rate = abs(refute_unobserved.new_effect - refute_unobserved.estimated_effect) / abs(refute_unobserved.estimated_effect) if abs(refute_unobserved.estimated_effect) > 0 else float('inf')
        p_value = calculate_refutation_pvalue(refute_unobserved, "unobserved")
        status = "강건함" if change_rate < 0.2 else "민감함"
        
        print(f"✅ 미관측 교란 테스트 완료: {status}")
        print(f"   기존 추정치: {refute_unobserved.estimated_effect:.6f}")
        print(f"   교란 추가 후 추정치: {refute_unobserved.new_effect:.6f}")
        print(f"   변화율: {change_rate:.2%}")
        if p_value is not None:
            print(f"   P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
        
        if logger:
            logger.info("✅ 미관측 교란 테스트 성공")
            logger.info(f"테스트 결과: {status}")
            logger.info(f"기존 추정치: {refute_unobserved.estimated_effect:.6f}")
            logger.info(f"교란 추가 후 추정치: {refute_unobserved.new_effect:.6f}")
            logger.info(f"변화율: {change_rate:.2%}")
            if p_value is not None:
                logger.info(f"P-value: {p_value:.6f}")
                logger.info(f"통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
            
    except Exception as e:
        validation_results['unobserved'] = None
        print(f"❌ 미관측 교란 테스트 실패: {e}")
        if logger:
            logger.error(f"❌ 미관측 교란 테스트 실패: {e}")
    
    # 3. 부분표본 안정성 테스트 (Data Subset)
    print("3️⃣ 부분표본 안정성 테스트 실행 중...")
    if logger:
        logger.info("3️⃣ 부분표본 안정성 테스트 실행 중...")
    
    try:
        refute_subset = model.refute_estimate(
            identified_estimand, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,  # 80% 서브셋 사용
            num_simulations=100
        )
        validation_results['subset'] = refute_subset
        
        effect_change = abs(refute_subset.new_effect - refute_subset.estimated_effect)
        change_rate = abs(refute_subset.estimated_effect) > 0 and abs(effect_change / refute_subset.estimated_effect) or float('inf')
        p_value = calculate_refutation_pvalue(refute_subset, "subset")
        status = "통과" if change_rate < 0.1 else "실패"  # 10% 이내 변화면 통과
        
        print(f"✅ 부분표본 안정성 테스트 완료: {status}")
        print(f"   기존 추정치: {refute_subset.estimated_effect:.6f}")
        print(f"   부분표본 추정치: {refute_subset.new_effect:.6f}")
        print(f"   효과 변화: {effect_change:.6f} ({change_rate:.2%})")
        if p_value is not None:
            print(f"   P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
        
        if logger:
            logger.info("✅ 부분표본 안정성 테스트 성공")
            logger.info(f"테스트 결과: {status}")
            logger.info(f"기존 추정치: {refute_subset.estimated_effect:.6f}")
            logger.info(f"부분표본 추정치: {refute_subset.new_effect:.6f}")
            logger.info(f"효과 변화: {effect_change:.6f} ({change_rate:.2%})")
            if p_value is not None:
                logger.info(f"P-value: {p_value:.6f}")
                logger.info(f"통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
            
    except Exception as e:
        validation_results['subset'] = None
        print(f"❌ 부분표본 안정성 테스트 실패: {e}")
        if logger:
            logger.error(f"❌ 부분표본 안정성 테스트 실패: {e}")
    
    # 4. 더미 결과 테스트 (Dummy Outcome)
    print("4️⃣ 더미 결과 테스트 실행 중...")
    if logger:
        logger.info("4️⃣ 더미 결과 테스트 실행 중...")
    
    try:
        refute_dummy = model.refute_estimate(
            identified_estimand, estimate,
            method_name="dummy_outcome_refuter",
            num_simulations=100
        )
        validation_results['dummy'] = refute_dummy
        
        p_value = calculate_refutation_pvalue(refute_dummy, "dummy")
        # new_effect가 0에 가까우면 통과 (0.01 이하)
        status = "통과" if abs(refute_dummy.new_effect) < 0.01 else "실패"
        
        print(f"✅ 더미 결과 테스트 완료: {status}")
        print(f"   더미 결과 추정치: {refute_dummy.new_effect:.6f}")
        print(f"   (0에 가까울수록 좋음, 0.01 이하면 통과)")
        if p_value is not None:
            print(f"   P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
        
        if logger:
            logger.info("✅ 더미 결과 테스트 성공")
            logger.info(f"테스트 결과: {status}")
            logger.info(f"더미 결과 추정치: {refute_dummy.new_effect:.6f}")
            logger.info(f"(0에 가까울수록 좋음, 0.01 이하면 통과)")
            if p_value is not None:
                logger.info(f"P-value: {p_value:.6f}")
                logger.info(f"통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
            
    except Exception as e:
        validation_results['dummy'] = None
        print(f"❌ 더미 결과 테스트 실패: {e}")
        if logger:
            logger.error(f"❌ 더미 결과 테스트 실패: {e}")
    
    print("="*60)
    print("검증 테스트 완료 (4개 테스트)")
    print("="*60)
    
    if logger:
        logger.info("="*60)
        logger.info("검증 테스트 완료 (4개 테스트)")
        logger.info("="*60)
        log_validation_results(logger, validation_results)
    
    return validation_results

def log_sensitivity_analysis(logger, sensitivity_df, config):
    """
    민감도 분석 결과를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        sensitivity_df (pd.DataFrame): 민감도 분석 결과
        config (dict): 민감도 분석 설정
    """
    logger.info("="*60)
    logger.info("민감도 분석 결과")
    logger.info("="*60)
    
    logger.info(f"효과 강도 범위: {config['effect_strength_range'][0]} ~ {config['effect_strength_range'][1]}")
    logger.info(f"그리드 포인트 수: {config['num_points']}")
    logger.info(f"시뮬레이션 수: {config['num_simulations']}")
    logger.info(f"분석된 조합 수: {len(sensitivity_df)}")
    
    if not sensitivity_df.empty:
        logger.info(f"효과 범위: {sensitivity_df['new_effect'].min():.6f} ~ {sensitivity_df['new_effect'].max():.6f}")
        
        # 효과가 0에 가까운 지점 찾기
        min_abs_effect = sensitivity_df.loc[sensitivity_df['new_effect'].abs().idxmin()]
        logger.info(f"최소 절대 효과 지점:")
        logger.info(f"  - 처치 강도 (et): {min_abs_effect['effect_strength_on_treatment']:.2f}")
        logger.info(f"  - 결과 강도 (eo): {min_abs_effect['effect_strength_on_outcome']:.2f}")
        logger.info(f"  - 효과값: {min_abs_effect['new_effect']:.6f}")
        
        # 효과가 음수인 조합 수
        negative_effects = len(sensitivity_df[sensitivity_df['new_effect'] < 0])
        logger.info(f"음수 효과 조합 수: {negative_effects} ({negative_effects/len(sensitivity_df)*100:.1f}%)")
        
        # 효과가 0에 가까운 조합 수 (절대값 < 0.01)
        near_zero_effects = len(sensitivity_df[sensitivity_df['new_effect'].abs() < 0.01])
        logger.info(f"0에 가까운 효과 조합 수: {near_zero_effects} ({near_zero_effects/len(sensitivity_df)*100:.1f}%)")

def run_sensitivity_analysis(model, identified_estimand, estimate, config, logger=None):
    """
    민감도 분석을 실행하는 함수
    
    Args:
        model: CausalModel 객체
        identified_estimand: 식별된 추정량 객체
        estimate: 추정된 인과효과 객체
        config (dict): 민감도 분석 설정
        logger: 로거 객체 (선택사항)
    
    Returns:
        pd.DataFrame: 민감도 분석 결과 데이터프레임
    """
    try:
        effect_range = config['effect_strength_range']
        num_points = config['num_points']
        num_simulations = config['num_simulations']
        
        grid = np.linspace(effect_range[0], effect_range[1], num_points)
        
        rows = []
        for i, et in enumerate(grid):
            for j, eo in enumerate(grid):
                try:
                    ref = model.refute_estimate(
                        identified_estimand, estimate,
                        method_name="add_unobserved_common_cause",
                        confounders_effect_on_treatment="binary_flip",
                        confounders_effect_on_outcome="linear",
                        effect_strength_on_treatment=et,
                        effect_strength_on_outcome=eo,
                        num_simulations=num_simulations
                    )
                    rows.append((et, eo, ref.new_effect))
                except Exception as e:
                    rows.append((et, eo, np.nan))
                    if logger:
                        logger.warning(f"민감도 분석 그리드 포인트 ({et}, {eo}) 실행 실패: {e}")
        
        sensitivity_df = pd.DataFrame(rows, columns=[
            "effect_strength_on_treatment", 
            "effect_strength_on_outcome", 
            "new_effect"
        ])
        
        if logger:
            log_sensitivity_analysis(logger, sensitivity_df, config)
        
        return sensitivity_df
        
    except Exception as e:
        if logger:
            logger.error(f"민감도 분석 중 오류 발생: {e}")
        return pd.DataFrame()

def log_heatmap_info(logger, heatmap_path, config):
    """
    히트맵 정보를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        heatmap_path (str): 히트맵 파일 경로
        config (dict): 시각화 설정
    """
    logger.info("="*60)
    logger.info("시각화 결과")
    logger.info("="*60)
    
    if heatmap_path and os.path.exists(heatmap_path):
        file_size = os.path.getsize(heatmap_path)
        logger.info(f"히트맵 파일: {heatmap_path}")
        logger.info(f"파일 크기: {file_size:,} bytes")
        logger.info(f"이미지 해상도: {config['figsize'][0]}x{config['figsize'][1]} inches")
        logger.info(f"DPI: {config['dpi']}")
    else:
        logger.warning("히트맵 파일이 생성되지 않았습니다.")

def run_sensitivity_analysis(model, identified_estimand, estimate, logger=None):
    """민감도 분석을 실행하는 함수"""
    if logger:
        logger.info("="*60)
        logger.info("민감도 분석 실행 시작")
        logger.info("="*60)
        logger.info("효과 강도 범위: 0.0 ~ 0.5")
        logger.info("그리드 포인트 수: 11x11 = 121개")
        logger.info("시뮬레이션 수: 200회")
    
    try:
        grid = np.linspace(0.0, 0.5, 11)
        rows = []
        total_combinations = len(grid) * len(grid)
        processed = 0
        
        if logger:
            logger.info(f"총 {total_combinations}개 조합 분석 시작...")
        
        for i, et in enumerate(grid):
            for j, eo in enumerate(grid):
                processed += 1
                if logger and processed % 20 == 0:
                    logger.info(f"진행률: {processed}/{total_combinations} ({processed/total_combinations*100:.1f}%)")
                
                try:
                    ref = model.refute_estimate(
                        identified_estimand, estimate,
                        method_name="add_unobserved_common_cause",
                        confounders_effect_on_treatment="binary_flip",
                        confounders_effect_on_outcome="linear",
                        effect_strength_on_treatment=et,
                        effect_strength_on_outcome=eo,
                        num_simulations=200
                    )
                    rows.append((et, eo, ref.new_effect))
                except Exception as e:
                    rows.append((et, eo, np.nan))
                    if logger:
                        logger.warning(f"그리드 포인트 ({et:.2f}, {eo:.2f}) 실행 실패: {e}")
        
        sensitivity_df = pd.DataFrame(rows, columns=[
            "effect_strength_on_treatment", 
            "effect_strength_on_outcome", 
            "new_effect"
        ])
        
        if logger:
            logger.info("✅ 민감도 분석 완료")
            logger.info(f"분석된 조합 수: {len(sensitivity_df)}")
            
            if not sensitivity_df.empty:
                valid_effects = sensitivity_df.dropna()
                logger.info(f"유효한 결과 수: {len(valid_effects)}")
                logger.info(f"효과 범위: {valid_effects['new_effect'].min():.6f} ~ {valid_effects['new_effect'].max():.6f}")
                
                # 효과가 0에 가까운 지점 찾기
                min_abs_effect = valid_effects.loc[valid_effects['new_effect'].abs().idxmin()]
                logger.info(f"최소 절대 효과 지점: et={min_abs_effect['effect_strength_on_treatment']:.2f}, eo={min_abs_effect['effect_strength_on_outcome']:.2f}")
                logger.info(f"최소 절대 효과값: {min_abs_effect['new_effect']:.6f}")
                
                # 음수 효과 비율
                negative_effects = len(valid_effects[valid_effects['new_effect'] < 0])
                logger.info(f"음수 효과 조합: {negative_effects}개 ({negative_effects/len(valid_effects)*100:.1f}%)")
        
        return sensitivity_df
        
    except Exception as e:
        if logger:
            logger.error(f"❌ 민감도 분석 실패: {e}")
        return pd.DataFrame()

def create_sensitivity_heatmap(sensitivity_df, logger=None):
    """민감도 분석 결과를 히트맵으로 시각화하는 함수"""
    if logger:
        logger.info("="*60)
        logger.info("히트맵 생성 시작")
        logger.info("="*60)
    
    if sensitivity_df.empty:
        if logger:
            logger.warning("❌ 민감도 분석 결과가 비어있어 히트맵을 생성할 수 없습니다.")
        return None
    
    try:
        if logger:
            logger.info("피벗 테이블 생성 중...")
        
        # 피벗 테이블 생성
        pivot = sensitivity_df.pivot(
            index="effect_strength_on_treatment",
            columns="effect_strength_on_outcome",
            values="new_effect"
        ).sort_index(ascending=True)
        
        if logger:
            logger.info(f"피벗 테이블 크기: {pivot.shape}")
            logger.info("히트맵 시각화 생성 중...")
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[
                pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()
            ],
            cmap='RdYlBu_r'
        )
        
        # 색상막대 추가
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("New Effect (after unobserved confounding)", fontsize=12)
        
        # 0-컨투어 라인 추가
        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
        CS = ax.contour(X, Y, pivot.values, levels=[0.0], linewidths=2, colors='black')
        ax.clabel(CS, inline=True, fmt="effect=0", fontsize=10)
        
        # 축 레이블 및 제목
        ax.set_xlabel("Effect Strength on Outcome (eo)", fontsize=12)
        ax.set_ylabel("Effect Strength on Treatment (et)", fontsize=12)
        ax.set_title("Sensitivity Analysis: Effect of Unobserved Confounders", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if logger:
            logger.info("히트맵 저장 중...")
        
        # 그림 저장
        script_dir = Path(__file__).parent.parent
        log_dir = script_dir / "log"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensitivity_heatmap_{timestamp}.png"
        output_path = log_dir / filename
        
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        if logger:
            logger.info("✅ 히트맵 생성 성공")
            logger.info(f"저장 경로: {output_path}")
            
            # 파일 정보
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"파일 크기: {file_size:,} bytes")
                logger.info(f"이미지 해상도: 10x8 inches, DPI: 100")
        
        return output_path
        
    except Exception as e:
        if logger:
            logger.error(f"❌ 히트맵 생성 실패: {e}")
        return None

def print_summary_report(estimate, validation_results, sensitivity_df):
    """
    전체 분석 결과 요약 보고서를 출력하는 함수
    
    Args:
        estimate: 추정된 인과효과 객체
        validation_results (dict): 검증 결과 딕셔너리
        sensitivity_df (pd.DataFrame): 민감도 분석 결과
    """
    print("\n" + "="*80)
    print("📋 최종 분석 결과 요약 보고서")
    print("="*80)
    
    # 기본 추정 결과
    print(f"\n🎯 주요 추정 결과:")
    print(f"  - 추정된 인과 효과 (ATE): {estimate.value:.6f}")
    if hasattr(estimate, 'p_value') and estimate.p_value is not None:
        significance = "유의함" if estimate.p_value <= 0.05 else "유의하지 않음"
        print(f"  - 통계적 유의성: {significance} (p-value: {estimate.p_value:.6f})")
    
    # 검증 결과 요약 (4개 테스트)
    print(f"\n🔬 검증 결과 요약 (4개 테스트):")
    
    if validation_results.get('placebo'):
        placebo = validation_results['placebo']
        effect_change = abs(placebo.new_effect - placebo.estimated_effect)
        p_value = calculate_refutation_pvalue(placebo, "placebo")
        status = "통과" if effect_change < 0.01 else "실패"
        print(f"  1. 가상 원인 테스트: {status}")
        print(f"     - 효과 변화: {effect_change:.6f}")
        if p_value is not None:
            print(f"     - P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
    else:
        print(f"  1. 가상 원인 테스트: 실행 실패")
    
    if validation_results.get('unobserved'):
        unobserved = validation_results['unobserved']
        change_rate = abs(unobserved.new_effect - unobserved.estimated_effect) / abs(unobserved.estimated_effect) if abs(unobserved.estimated_effect) > 0 else float('inf')
        p_value = calculate_refutation_pvalue(unobserved, "unobserved")
        status = "강건함" if change_rate < 0.2 else "민감함"
        print(f"  2. 미관측 교란 테스트: {status}")
        print(f"     - 변화율: {change_rate:.2%}")
        if p_value is not None:
            print(f"     - P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
    else:
        print(f"  2. 미관측 교란 테스트: 실행 실패")
    
    if validation_results.get('subset'):
        subset = validation_results['subset']
        effect_change = abs(subset.new_effect - subset.estimated_effect)
        change_rate = abs(subset.estimated_effect) > 0 and abs(effect_change / subset.estimated_effect) or float('inf')
        p_value = calculate_refutation_pvalue(subset, "subset")
        status = "통과" if change_rate < 0.1 else "실패"
        print(f"  3. 부분표본 안정성 테스트: {status}")
        print(f"     - 효과 변화율: {change_rate:.2%}")
        if p_value is not None:
            print(f"     - P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
    else:
        print(f"  3. 부분표본 안정성 테스트: 실행 실패")
    
    if validation_results.get('dummy'):
        dummy = validation_results['dummy']
        p_value = calculate_refutation_pvalue(dummy, "dummy")
        status = "통과" if abs(dummy.new_effect) < 0.01 else "실패"
        print(f"  4. 더미 결과 테스트: {status}")
        print(f"     - 더미 결과 추정치: {dummy.new_effect:.6f}")
        if p_value is not None:
            print(f"     - P-value: {p_value:.6f} ({'유의함' if p_value <= 0.05 else '유의하지 않음'})")
    else:
        print(f"  4. 더미 결과 테스트: 실행 실패")
    
    # 민감도 분석 요약
    if not sensitivity_df.empty:
        print(f"\n📈 민감도 분석 요약:")
        print(f"  - 분석된 조합 수: {len(sensitivity_df)}")
        print(f"  - 효과 범위: {sensitivity_df['new_effect'].min():.6f} ~ {sensitivity_df['new_effect'].max():.6f}")
        
        # 효과가 0에 가까운 지점 찾기
        min_abs_effect = sensitivity_df.loc[sensitivity_df['new_effect'].abs().idxmin()]
        print(f"  - 최소 절대 효과 지점: et={min_abs_effect['effect_strength_on_treatment']:.2f}, eo={min_abs_effect['effect_strength_on_outcome']:.2f}")
    
    print(f"\n✅ 전체 분석 완료!")


# ============================================================================
# Checkpoint 저장/로드 함수
# ============================================================================

def save_checkpoint(estimate, checkpoint_dir, experiment_id, graph_name=None, logger=None):
    """
    CausalEstimate 객체를 checkpoint로 저장하는 함수
    
    Args:
        estimate: CausalEstimate 객체
        checkpoint_dir (str or Path): checkpoint 저장 디렉토리
        experiment_id (str): 실험 ID (파일명에 사용)
        graph_name (str, optional): 그래프 파일명 (metadata에 저장)
        logger: 로거 객체
    
    Returns:
        str: 저장된 checkpoint 파일 경로
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # checkpoint 파일명 생성
    checkpoint_filename = f"checkpoint_{experiment_id}.pkl"
    checkpoint_file = checkpoint_path / checkpoint_filename
    
    # experiment_id에서 graph_name 추출 (없으면 전달받은 값 사용)
    if graph_name is None and experiment_id:
        # experiment_id 형식: exp_0001_graph_name_treatment_outcome_estimator
        parts = experiment_id.split('_')
        if len(parts) >= 3:
            graph_name = parts[2]  # graph_name 위치
    
    # 메타데이터 저장
    metadata = {
        "experiment_id": experiment_id,
        "graph_name": graph_name,
        "treatment": estimate._treatment_name[0] if isinstance(estimate._treatment_name, list) else estimate._treatment_name,
        "outcome": estimate._outcome_name[0] if isinstance(estimate._outcome_name, list) else estimate._outcome_name,
        "ate_value": estimate.value,
        "control_value": estimate.control_value,
        "treatment_value": estimate.treatment_value,
        "estimator_type": type(estimate.estimator).__name__ if hasattr(estimate, 'estimator') else None,
        "saved_at": datetime.now().isoformat()
    }
    
    metadata_filename = f"metadata_{experiment_id}.json"
    metadata_file = checkpoint_path / metadata_filename
    
    try:
        # CausalEstimate 객체 저장
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(estimate, f)
        
        # 메타데이터 저장
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if logger:
            logger.info(f"✅ Checkpoint 저장 완료: {checkpoint_file}")
        print(f"✅ Checkpoint 저장 완료: {checkpoint_file}")
        
        return str(checkpoint_file)
        
    except Exception as e:
        error_msg = f"Checkpoint 저장 실패: {e}"
        if logger:
            logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise


def load_checkpoint(checkpoint_file, logger=None):
    """
    Checkpoint에서 CausalEstimate 객체를 로드하는 함수
    
    Args:
        checkpoint_file (str or Path): checkpoint 파일 경로
        logger: 로거 객체
    
    Returns:
        CausalEstimate: 로드된 CausalEstimate 객체
    """
    checkpoint_path = Path(checkpoint_file)
    
    if not checkpoint_path.exists():
        error_msg = f"Checkpoint 파일을 찾을 수 없습니다: {checkpoint_file}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(checkpoint_path, 'rb') as f:
            estimate = pickle.load(f)
        
        if logger:
            logger.info(f"✅ Checkpoint 로드 완료: {checkpoint_file}")
        print(f"✅ Checkpoint 로드 완료: {checkpoint_file}")
        
        return estimate
        
    except Exception as e:
        error_msg = f"Checkpoint 로드 실패: {e}"
        if logger:
            logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise


def find_checkpoint(checkpoint_dir, graph_name, treatment, outcome, estimator, logger=None):
    """
    주어진 조건에 맞는 checkpoint 파일을 찾는 함수
    
    Args:
        checkpoint_dir (str or Path): checkpoint 디렉토리
        graph_name (str): 그래프 파일명
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        logger: 로거 객체
    
    Returns:
        str or None: 찾은 checkpoint 파일 경로, 없으면 None
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        if logger:
            logger.warning(f"Checkpoint 디렉토리가 존재하지 않습니다: {checkpoint_dir}")
        return None
    
    # metadata 파일들을 읽어서 조건에 맞는 checkpoint 찾기
    metadata_files = list(checkpoint_path.glob("metadata_*.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 조건 확인: graph_name, treatment, outcome, estimator 모두 일치해야 함
            metadata_graph = metadata.get("graph_name", "")
            metadata_treatment = metadata.get("treatment", "")
            metadata_outcome = metadata.get("outcome", "")
            metadata_estimator = metadata.get("estimator_type", "").lower().replace("estimator", "")
            target_estimator = estimator.lower().replace("_", "")
            
            if (metadata_graph == graph_name and
                metadata_treatment == treatment and 
                metadata_outcome == outcome and
                metadata_estimator == target_estimator):
                
                # experiment_id에서 checkpoint 파일명 생성
                experiment_id = metadata.get("experiment_id", "")
                checkpoint_file = checkpoint_path / f"checkpoint_{experiment_id}.pkl"
                
                if checkpoint_file.exists():
                    if logger:
                        logger.info(f"✅ Checkpoint 발견: {checkpoint_file}")
                    return str(checkpoint_file)
                    
        except Exception as e:
            if logger:
                logger.warning(f"Metadata 파일 읽기 실패 ({metadata_file}): {e}")
            continue
    
    if logger:
        logger.warning(f"조건에 맞는 checkpoint를 찾을 수 없습니다: graph={graph_name}, treatment={treatment}, outcome={outcome}, estimator={estimator}")
    return None
