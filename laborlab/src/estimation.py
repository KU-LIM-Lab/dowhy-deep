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
        tuple: (data_df_with_predictions, accuracy)
            - data_df_with_predictions: ACQ_180_YN 열에 예측값이 채워진 데이터프레임
            - accuracy: 정확도 (이진 분류) 또는 None (연속형)
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
        if treatment_value is not None:
            predictions = estimator.interventional_outcomes(data_df, treatment_value)
        else:
            predictions = estimator.predict(data_df)
        
        predictions_series = pd.Series(predictions, index=data_df.index)
        
        # 데이터프레임 복사 후 예측값 채우기
        result_df = data_df.copy()
        outcome_name = estimate.outcome_name
        result_df[outcome_name] = predictions_series
        
        # 실제 Y 값과 비교하여 정확도 계산
        outcome_name = estimate.outcome_name
        accuracy = 0
        if outcome_name in data_df.columns:
            actual_y = data_df[outcome_name]
            predicted_classes = (predictions_series > 0.5).astype(int)
            accuracy = (predicted_classes == actual_y).mean()
            if logger:
                logger.info(f"예측 완료: 정확도={accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            if logger:
                logger.info(f"예측 완료: 평균={predictions_series.mean():.6f}")
                logger.warning(f"실제 Y 값({outcome_name})을 찾을 수 없어 정확도를 계산할 수 없습니다.")
        
        return accuracy, result_df
        
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
        # TabPFN의 경우 특별한 파라미터 설정 (legacy 버전 사용)
        if estimator == 'tabpfn':
            from dowhy.causal_estimators.tabpfn_estimator_legacy import TabpfnEstimator
            # tabpfn_estimator_legacy는 DoWhy의 표준 naming convention과 다르므로
            # 직접 estimator 인스턴스를 생성하여 DoWhy의 estimate_effect 함수에 전달
            tabpfn_estimator = TabpfnEstimator(
                identified_estimand,
                test_significance=True,
                method_params={
                    "N_ensemble_configurations": 8
                }
            )
            # DoWhy의 estimate_effect 함수를 직접 호출
            estimate = dowhy_estimate_effect(
                data=model._data,
                treatment=model._treatment,
                outcome=model._outcome,
                identifier_name="backdoor",
                estimator=tabpfn_estimator,
                control_value=0,
                treatment_value=1,
                target_units="ate",
                effect_modifiers=model._graph.get_effect_modifiers(model._treatment, model._outcome),
                fit_estimator=True,
                method_params={}
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
        status = "통과" if effect_change < 0.01 else "실패"
        logger.info(f"가상 원인 테스트: {status}")
        logger.info(f"  - 기존 추정치: {placebo.estimated_effect:.6f}")
        logger.info(f"  - 가상처치 후 추정치: {placebo.new_effect:.6f}")
        logger.info(f"  - 효과 변화: {effect_change:.6f}")
    
    # 미관측 교란 테스트
    if validation_results.get('unobserved'):
        unobserved = validation_results['unobserved']
        change_rate = abs(unobserved.new_effect - unobserved.estimated_effect) / abs(unobserved.estimated_effect)
        status = "강건함" if change_rate < 0.2 else "민감함"
        logger.info(f"미관측 교란 테스트: {status}")
        logger.info(f"  - 기존 추정치: {unobserved.estimated_effect:.6f}")
        logger.info(f"  - 교란 추가 후 추정치: {unobserved.new_effect:.6f}")
        logger.info(f"  - 변화율: {change_rate:.2%}")
    
    # 부분표본 안정성 테스트
    if validation_results.get('subset'):
        subset = validation_results['subset']
        logger.info(f"부분표본 안정성 테스트:")
        logger.info(f"  - 기존 추정치: {subset.estimated_effect:.6f}")
        logger.info(f"  - 부분표본 추정치: {subset.new_effect:.6f}")
    
    # 더미 결과 테스트
    if validation_results.get('dummy'):
        dummy = validation_results['dummy']
        status = "통과" if abs(dummy.new_effect) < 0.01 else "실패"
        logger.info(f"더미 결과 테스트: {status}")
        logger.info(f"  - 더미 결과 추정치: {dummy.new_effect:.6f}")

def run_validation_tests(model, identified_estimand, estimate, logger=None):
    """검증 테스트를 실행하는 함수"""
    if logger:
        logger.info("="*60)
        logger.info("검증 테스트 실행 시작")
        logger.info("="*60)
    
    validation_results = {}
    
    # 가상 원인 테스트
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
        
        if logger:
            logger.info("✅ 가상 원인 테스트 성공")
            effect_change = abs(refute_placebo.new_effect - refute_placebo.estimated_effect)
            status = "통과" if effect_change < 0.01 else "실패"
            logger.info(f"테스트 결과: {status}")
            logger.info(f"기존 추정치: {refute_placebo.estimated_effect:.6f}")
            logger.info(f"가상처치 후 추정치: {refute_placebo.new_effect:.6f}")
            logger.info(f"효과 변화: {effect_change:.6f}")
            
    except Exception as e:
        validation_results['placebo'] = None
        if logger:
            logger.error(f"❌ 가상 원인 테스트 실패: {e}")
    
    # 미관측 교란 테스트
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
        
        if logger:
            logger.info("✅ 미관측 교란 테스트 성공")
            change_rate = abs(refute_unobserved.new_effect - refute_unobserved.estimated_effect) / abs(refute_unobserved.estimated_effect)
            status = "강건함" if change_rate < 0.2 else "민감함"
            logger.info(f"테스트 결과: {status}")
            logger.info(f"기존 추정치: {refute_unobserved.estimated_effect:.6f}")
            logger.info(f"교란 추가 후 추정치: {refute_unobserved.new_effect:.6f}")
            logger.info(f"변화율: {change_rate:.2%}")
            
    except Exception as e:
        validation_results['unobserved'] = None
        if logger:
            logger.error(f"❌ 미관측 교란 테스트 실패: {e}")
    
    if logger:
        logger.info("="*60)
        logger.info("검증 테스트 완료")
        logger.info("="*60)
    
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
    
    # 검증 결과 요약
    print(f"\n🔬 검증 결과 요약:")
    
    if validation_results.get('placebo'):
        placebo = validation_results['placebo']
        effect_change = abs(placebo.new_effect - placebo.estimated_effect)
        print(f"  - 가상 원인 테스트: {'통과' if effect_change < 0.01 else '실패'}")
    
    if validation_results.get('unobserved'):
        unobserved = validation_results['unobserved']
        change_rate = abs(unobserved.new_effect - unobserved.estimated_effect) / abs(unobserved.estimated_effect)
        print(f"  - 미관측 교란 테스트: {'강건함' if change_rate < 0.2 else '민감함'}")
    
    if validation_results.get('subset'):
        subset = validation_results['subset']
        print(f"  - 부분표본 안정성: 추정치 변화 확인됨")
    
    if validation_results.get('dummy'):
        dummy = validation_results['dummy']
        print(f"  - 더미 결과 테스트: {'통과' if abs(dummy.new_effect) < 0.01 else '실패'}")
    
    # 민감도 분석 요약
    if not sensitivity_df.empty:
        print(f"\n📈 민감도 분석 요약:")
        print(f"  - 분석된 조합 수: {len(sensitivity_df)}")
        print(f"  - 효과 범위: {sensitivity_df['new_effect'].min():.6f} ~ {sensitivity_df['new_effect'].max():.6f}")
        
        # 효과가 0에 가까운 지점 찾기
        min_abs_effect = sensitivity_df.loc[sensitivity_df['new_effect'].abs().idxmin()]
        print(f"  - 최소 절대 효과 지점: et={min_abs_effect['effect_strength_on_treatment']:.2f}, eo={min_abs_effect['effect_strength_on_outcome']:.2f}")
    
    print(f"\n✅ 전체 분석 완료!")
