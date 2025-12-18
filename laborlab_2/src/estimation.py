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
import time
import itertools
import gc
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
from . import utils

# CUDA 0번 GPU 사용 (Docker 컨테이너 내부에서는 할당된 GPU가 0번으로 보임)
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)

from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy import CausalModel

# 로컬 DoWhy 라이브러리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# DoWhy 내부 함수 임포트
from dowhy.causal_estimator import estimate_effect as dowhy_estimate_effect

# DoWhy/TabPFN 내부 로깅 활성화 (INFO 레벨)
logging.getLogger("dowhy").setLevel(logging.INFO)
logging.getLogger("dowhy.causal_model").setLevel(logging.INFO)
logging.getLogger("dowhy.causal_estimator").setLevel(logging.INFO)
logging.getLogger("dowhy.causal_estimators.tabpfn_estimator").setLevel(logging.INFO)
logging.getLogger("tabpfn").setLevel(logging.INFO)

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
    
    total_samples = len(data_df)
    if logger:
        logger.info(f"E(Y|A, X) 예측 시작: {total_samples}개")
        if treatment_value is not None:
            logger.info(f"처치 값: {treatment_value}")
    
    print(f"🔮 예측 시작: {total_samples}개 샘플")
    
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
        
        # OrdinalEncoder가 저장되어 있으면 예측 데이터에도 적용
        if hasattr(estimate, '_ordinal_encoder') and hasattr(estimate, '_categorical_columns'):
            ordinal_encoder = estimate._ordinal_encoder
            categorical_columns = estimate._categorical_columns
            
            # 존재하는 컬럼 중 인코딩이 필요한 것만 필터링
            cols_to_encode = [
                col for col in categorical_columns 
                if col in data_df_clean.columns and not pd.api.types.is_integer_dtype(data_df_clean[col])
            ]
            
            if cols_to_encode:
                print(f"🔢 예측 데이터에 OrdinalEncoder 적용: {len(cols_to_encode)}개 변수")
                
                # 전처리: NaN 처리 및 문자열 변환
                for col in cols_to_encode:
                    data_df_clean[col] = data_df_clean[col].fillna('__nan__').astype(str)
                
                # 인코딩 적용
                try:
                    data_df_clean[cols_to_encode] = ordinal_encoder.transform(data_df_clean[cols_to_encode])
                    
                    # Unknown categories 로깅
                    for col in cols_to_encode:
                        unknown_count = (data_df_clean[col] == -1).sum()
                        if unknown_count > 0:
                            print(f"   ⚠️ '{col}': {unknown_count}개 unknown categories → -1로 인코딩됨")
                except Exception as e:
                    print(f"   ⚠️ OrdinalEncoder 오류: {e}")
                    print(f"   컬럼 타입: {[(col, str(data_df_clean[col].dtype)) for col in cols_to_encode]}")
                    raise
        
        # Unknown Categories 안전장치: 예측 시도 시 오류 발생하면 해당 행 제외
        def safe_predict(df_to_predict, treatment_val=None):
            """Unknown categories 오류 발생 시 해당 행을 제외하고 재시도"""
            try:
                if treatment_val is not None:
                    return estimator.interventional_outcomes(df_to_predict, treatment_val)
                else:
                    return estimator.predict(df_to_predict)
            except ValueError as e:
                if "unknown categories" in str(e).lower() or "found unknown" in str(e).lower():
                    error_msg = str(e)
                    
                    # 오류 메시지에서 컬럼 정보 추출
                    column_info = "알 수 없음"
                    if "column" in error_msg.lower():
                        # "in column 0" 또는 "in column 'col_name'" 형식 파싱
                        import re
                        col_match = re.search(r"column\s+(\d+|\w+)", error_msg, re.IGNORECASE)
                        if col_match:
                            col_ref = col_match.group(1)
                            # 숫자인 경우 인덱스로 변환
                            try:
                                col_idx = int(col_ref)
                                categorical_cols = df_to_predict.select_dtypes(include=['object', 'string', 'category']).columns
                                if col_idx < len(categorical_cols):
                                    column_info = categorical_cols[col_idx]
                            except:
                                column_info = col_ref
                    
                    # 카테고리 변수 식별 및 필터링
                    categorical_cols = df_to_predict.select_dtypes(include=['object', 'string', 'category']).columns
                    problematic_cols = []
                    
                    if len(categorical_cols) > 0 and hasattr(estimator, '_data'):
                        # Train 데이터의 카테고리 값만 유지
                        train_data = estimator._data
                        rows_before = len(df_to_predict)
                        
                        for col in categorical_cols:
                            if col in train_data.columns:
                                train_categories = set(train_data[col].dropna().unique())
                                test_categories = set(df_to_predict[col].dropna().unique())
                                unknown_categories = test_categories - train_categories
                                
                                if unknown_categories:
                                    problematic_cols.append({
                                        'column': col,
                                        'unknown_count': len(unknown_categories),
                                        'unknown_values': list(unknown_categories)[:10]  # 최대 10개만 표시
                                    })
                                    mask = df_to_predict[col].isin(train_categories) | df_to_predict[col].isna()
                                    df_to_predict = df_to_predict[mask].copy()
                        
                        rows_after = len(df_to_predict)
                        rows_removed = rows_before - rows_after
                        
                        # 로깅 및 프린트
                        if problematic_cols:
                            for prob_col in problematic_cols:
                                col_name = prob_col['column']
                                unknown_vals = prob_col['unknown_values']
                                unknown_count = prob_col['unknown_count']
                                
                                msg = (
                                    f"⚠️ Unknown Categories 감지 - 컬럼: '{col_name}', "
                                    f"알 수 없는 값: {unknown_count}개 "
                                    f"({unknown_vals[:5]}{'...' if len(unknown_vals) > 5 else ''})"
                                )
                                print(msg)
                                if logger:
                                    logger.warning(msg)
                        
                        if rows_removed > 0:
                            msg = f"📊 필터링 결과: {rows_before}건 → {rows_after}건 ({rows_removed}건 제거)"
                            print(msg)
                            if logger:
                                logger.info(msg)
                        
                        if len(df_to_predict) == 0:
                            error_msg = "필터링 후 예측 가능한 데이터가 없습니다."
                            print(f"❌ {error_msg}")
                            if logger:
                                logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                        # 재시도
                        if treatment_val is not None:
                            return estimator.interventional_outcomes(df_to_predict, treatment_val)
                        else:
                            return estimator.predict(df_to_predict)
                raise
        
        # 예측 수행 (진행률 표시)
        estimator_type = type(estimator).__name__
        is_tabpfn = 'tabpfn' in estimator_type.lower() or 'TabPFN' in estimator_type
        
        # TabPFN의 경우 배치 크기 확인
        batch_size = None
        if is_tabpfn:
            # TabPFN의 경우 배치 크기 확인
            if hasattr(estimate, 'estimator') and hasattr(estimate.estimator, '_method_params'):
                method_params = estimate.estimator._method_params
                if method_params and 'prediction_batch_size' in method_params:
                    batch_size = method_params['prediction_batch_size']
            if batch_size is None:
                batch_size = 512  # 기본 배치 크기
        
        prediction_start_time = time.time()
        
        # 배치로 나누어 처리 가능한 경우 progress bar 표시
        if batch_size and total_samples > batch_size:
            num_batches = (total_samples + batch_size - 1) // batch_size
            print(f"📊 배치 크기: {batch_size}, 총 배치 수: {num_batches}")
            
            predictions_list = []
            with tqdm(total=total_samples, desc="예측 진행", unit="샘플", ncols=100, leave=True) as pbar:
                for i in range(0, total_samples, batch_size):
                    batch_end = min(i + batch_size, total_samples)
                    batch_df = data_df_clean.iloc[i:batch_end]
                    
                    try:
                        # 배치 예측 수행 (안전장치 포함)
                        batch_predictions = safe_predict(batch_df, treatment_value)
                        
                        # 예측 결과를 리스트에 추가
                        if isinstance(batch_predictions, np.ndarray):
                            predictions_list.append(batch_predictions)
                        elif isinstance(batch_predictions, (list, tuple)):
                            predictions_list.extend(batch_predictions)
                        elif isinstance(batch_predictions, pd.Series):
                            predictions_list.append(batch_predictions.values)
                        else:
                            predictions_list.append([batch_predictions])
                        
                        pbar.update(len(batch_df))
                    except Exception as e:
                        # 배치 처리 실패 시 전체 데이터로 fallback
                        if logger:
                            logger.warning(f"배치 예측 실패, 전체 데이터로 처리: {e}")
                        pbar.close()
                        print("⚠️ 배치 처리 실패, 전체 데이터로 예측 수행 중...")
                        predictions = safe_predict(data_df_clean, treatment_value)
                        prediction_elapsed = time.time() - prediction_start_time
                        print(f"✅ 예측 완료: {len(predictions)}개 예측값 생성 (소요 시간: {prediction_elapsed:.2f}초)")
                        break
                else:
                    # 모든 배치가 성공적으로 처리된 경우
                    if predictions_list:
                        # 배치 결과 합치기
                        if isinstance(predictions_list[0], np.ndarray):
                            predictions = np.concatenate(predictions_list)
                        else:
                            predictions = np.array([item for sublist in predictions_list for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])])
                        prediction_elapsed = time.time() - prediction_start_time
                        print(f"\n✅ 예측 완료: {len(predictions)}개 예측값 생성 (소요 시간: {prediction_elapsed:.2f}초)")
                    else:
                        raise ValueError("예측 결과가 생성되지 않았습니다.")
        else:
            # 전체 데이터를 한 번에 처리하거나 배치 크기가 충분히 큰 경우
            with tqdm(total=1, desc="예측 수행", unit="배치", ncols=100, leave=True) as pbar:
                predictions = safe_predict(data_df_clean, treatment_value)
                pbar.update(1)
            
            prediction_elapsed = time.time() - prediction_start_time
            print(f"✅ 예측 완료: {len(predictions)}개 예측값 생성 (소요 시간: {prediction_elapsed:.2f}초)")
        
        if logger:
            logger.info(f"예측 완료: {len(predictions)}개 예측값 생성")
        
        predictions_series = pd.Series(predictions, index=data_df_clean.index)
        
        # 데이터프레임 복사 후 예측값 채우기
        result_df = data_df_clean.copy()
        # _outcome_name은 리스트일 수 있음
        outcome_name = estimate._outcome_name[0] if isinstance(estimate._outcome_name, list) else estimate._outcome_name
        # 예측값이 0~1 사이의 확률인 경우 0.5를 기준으로 바이너리하게 변환
        # (사용자 요청: 0~1 사이의 확률값을 0.5 기준으로 바이너리하게 변경)
        if predictions_series.min() >= 0 and predictions_series.max() <= 1:
            if logger:
                logger.info("예측 확률값을 0.5 기준으로 바이너리(0/1) 값으로 변환합니다.")
            print("ℹ️ 예측 확률값을 0.5 기준으로 바이너리(0/1) 값으로 변환합니다.")
            # 원본 확률값은 _prob 접미사를 붙여 저장 (내부 메트릭 계산 용도)
            result_df[f"{outcome_name}_prob"] = predictions_series
            result_df[outcome_name] = (predictions_series >= 0.5).astype(int)
        else:
            result_df[outcome_name] = predictions_series
        
        # 실제 Y 값과 비교하여 메트릭 계산
        metrics = {'accuracy': None, 'f1_score': None, 'auc': None}
        if outcome_name in data_df_clean.columns:
            actual_y = data_df_clean[outcome_name]
            metrics = utils.calculate_metrics(actual_y, predictions_series, logger=logger)
            
            if metrics.get('accuracy') is not None:
                if logger:
                    logger.info(f"예측 완료: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics.get('auc', 'N/A')}")
            else:
                if logger:
                    # NaN 제거 후 데이터가 있는 경우만 평균 계산
                    valid_mask = ~(pd.isna(actual_y) | pd.isna(predictions_series))
                    if valid_mask.sum() > 0:
                        logger.info(f"예측 완료: 평균={predictions_series[valid_mask].mean():.6f} (연속형 변수)")
        else:
            if logger:
                logger.warning(f"실제 Y 값({outcome_name})을 찾을 수 없어 메트릭을 계산할 수 없습니다.")
        
        return metrics, result_df
        
    except Exception as e:
        if logger:
            logger.error(f"예측 실패: {e}")
        raise

def cleanup_tabpfn_memory(estimate, device_id=0, logger=None, force_release=False):
    """
    TabPFN 모델의 GPU 메모리를 완전히 해제하는 함수
    
    Args:
        estimate: CausalEstimate 객체
        device_id: CUDA device ID (기본값: 0, Docker 컨테이너 내부 기준)
        logger: 로거 객체
        force_release: 강제 메모리 해제 여부 (기본값: False)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        
        torch.cuda.set_device(device_id)
        
        # TabPFN 모델 객체에서 메모리 해제
        if hasattr(estimate, 'estimator') and hasattr(estimate.estimator, 'tabpfn_model'):
            tabpfn_model = estimate.estimator.tabpfn_model
            if tabpfn_model is not None:
                # _single_model 해제
                if hasattr(tabpfn_model, '_single_model') and tabpfn_model._single_model is not None:
                    try:
                        del tabpfn_model._single_model
                    except:
                        pass
                    tabpfn_model._single_model = None
                
                # train_X, train_y 메모리 해제
                if hasattr(tabpfn_model, 'train_X'):
                    try:
                        del tabpfn_model.train_X
                    except:
                        pass
                if hasattr(tabpfn_model, 'train_y'):
                    try:
                        del tabpfn_model.train_y
                    except:
                        pass
                
                # 모델 객체 삭제
                try:
                    del tabpfn_model
                except:
                    pass
                estimate.estimator.tabpfn_model = None
        
        # Python garbage collection 강제 실행
        gc.collect()
        
        # GPU 캐시 정리 (PyTorch 메모리 풀 정리)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 메모리 통계 리셋 (다른 서비스와 공유 시 유용)
        try:
            torch.cuda.reset_peak_memory_stats(device_id)
        except:
            pass  # 일부 PyTorch 버전에서는 지원하지 않을 수 있음
        
        # force_release 옵션이 활성화된 경우 추가 정리 시도
        if force_release:
            # 여러 번 empty_cache 호출로 메모리 풀 강제 정리 시도
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        if logger:
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
            logger.debug(f"TabPFN 메모리 정리 완료 (CUDA {device_id}) - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB")
            if reserved > 0.1:  # 예약 메모리가 100MB 이상이면 경고
                logger.warning(
                    f"⚠️ GPU 메모리 예약량이 {reserved:.2f}GB입니다. "
                    f"PyTorch는 메모리 풀을 사용하므로 예약된 메모리는 다른 프로세스가 즉시 사용할 수 없을 수 있습니다. "
                    f"다른 서비스와 같은 GPU를 공유하는 경우 메모리 부족 문제가 발생할 수 있습니다."
                )
    except Exception as e:
        if logger:
            logger.warning(f"TabPFN 메모리 정리 중 오류: {e}")


def estimate_causal_effect(model, identified_estimand, estimator, logger=None, tabpfn_config=None):
    """인과효과를 추정하는 함수
    
    Args:
        model: CausalModel 객체
        identified_estimand: IdentifiedEstimand 객체
        estimator: 추정 방법 이름
        logger: 로거 객체 (선택적)
        tabpfn_config: TabPFN 설정 딕셔너리 (선택적)
    """
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
    
    print(f"📊 인과효과 추정 시작: {estimator}")
    
    estimate = None
    try:
        # TabPFN의 경우 새 버전 사용 (표준 인터페이스)
        if estimator == 'tabpfn':
            # CUDA 0번 GPU 사용 (Docker 컨테이너 내부에서는 할당된 GPU가 0번으로 보임)
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            
            # 기본 TabPFN 설정 (CUDA 0번 사용)
            # device_ids를 빈 리스트로 설정하여 단일 GPU 모드 사용
            # torch.cuda.set_device(0)으로 기본 device가 0번으로 설정됨
            default_tabpfn_config = {
                "n_estimators": 8,
                "model_type": "auto",
                "use_multi_gpu": False,
                "device_ids": [],  # 빈 리스트 = 단일 GPU 모드 (기본 device 사용, 즉 CUDA 3번)
                "max_num_classes": 10,
                "prediction_batch_size": 64  # 배치 크기 (기본값: 64)
            }
            
            # config에서 설정 가져오기 (없으면 기본값 사용)
            if tabpfn_config:
                method_params = {**default_tabpfn_config, **tabpfn_config}
                # device_ids는 항상 빈 리스트로 강제 설정 (단일 GPU 모드, CUDA 3번 사용)
                method_params["device_ids"] = []
                method_params["use_multi_gpu"] = False  # 단일 GPU 모드
            else:
                method_params = default_tabpfn_config
            
            # device_ids 자동 감지 로직 제거 (항상 CUDA 0번 사용)
            
            if logger:
                logger.info("TabPFN 단일 GPU 모드 사용 (CUDA 0번)")
                # GPU 상태 로깅
                device_id = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_id)
                logger.info(f"🖥️ GPU 정보: {device_name} (cuda:{device_id})")
            
            print("⏳ TabPFN 모델 추정 중... (이 과정은 시간이 걸릴 수 있습니다)")
            print(f"   - n_estimators: {method_params.get('n_estimators', 8)}")
            print(f"   - prediction_batch_size: {method_params.get('prediction_batch_size', 64)}")
            
            # Progress bar 표시 (TabPFN은 내부적으로 처리되므로 간단한 progress bar)
            estimate_start_time = time.time()
            with tqdm(total=100, desc="추정 진행", unit="%", ncols=100, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}%') as pbar:
                # TabPFN 추정은 내부적으로 처리되므로 progress bar는 대략적인 진행률만 표시
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method,
                    method_params=method_params
                )
                pbar.update(100)  # 완료 시 100% 표시
            
            estimate_elapsed = time.time() - estimate_start_time
            print(f"✅ TabPFN 추정 완료 (소요 시간: {estimate_elapsed:.2f}초)")
                        
            # 로드된 모델의 실제 device 확인
            if logger and hasattr(estimate, 'estimator'):
                estimator_obj = estimate.estimator
                if hasattr(estimator_obj, '_device'):
                    logger.info(f"🔧 TabpfnEstimator._device: {estimator_obj._device}")
                if hasattr(estimator_obj, 'tabpfn_model') and estimator_obj.tabpfn_model is not None:
                    tabpfn_model = estimator_obj.tabpfn_model
                    # TabPFNModelWrapper에서 내부 모델 확인
                    if hasattr(tabpfn_model, '_single_model') and tabpfn_model._single_model is not None:
                        inner_model = tabpfn_model._single_model
                        # 모델 파라미터의 device 확인
                        device_info = None
                        try:
                            # 방법 1: parameters() 메서드가 있는 경우 (PyTorch 모델)
                            if hasattr(inner_model, 'parameters'):
                                try:
                                    first_param = next(inner_model.parameters())
                                    device_info = str(first_param.device)
                                except StopIteration:
                                    device_info = "파라미터 없음"
                            # 방법 2: device 속성이 직접 있는 경우
                            elif hasattr(inner_model, 'device'):
                                device_info = str(inner_model.device)
                            # 방법 3: 모델 타입 확인
                            else:
                                model_type = type(inner_model).__name__
                                device_info = f"device 속성 없음 (타입: {model_type})"
                            
                            if device_info:
                                logger.info(f"🎯 TabPFN 내부 모델 device: {device_info}")
                        except Exception as e:
                            logger.info(f"🎯 TabPFN 내부 모델 device 확인 실패: {e}")
                    else:
                        logger.info("🎯 TabPFN _single_model: None (멀티프로세싱 모드이거나 아직 로드 안됨)")
        else:
            print(f"⏳ {estimator} 추정 중...")
            estimate_start_time = time.time()
            with tqdm(total=100, desc="추정 진행", unit="%", ncols=100, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}%') as pbar:
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method
                )
                pbar.update(100)  # 완료 시 100% 표시
            estimate_elapsed = time.time() - estimate_start_time
            print(f"✅ {estimator} 추정 완료 (소요 시간: {estimate_elapsed:.2f}초)")
        
        if logger:
            logger.info("✅ 인과효과 추정 성공")
            logger.info(f"추정된 인과 효과 (ATE): {estimate.value:.6f}")

        return estimate
        
    except Exception as e:
        # 실패 시에도 GPU 메모리 정리 (CUDA 0번)
        if estimator == 'tabpfn':
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)  # CUDA 0번으로 설정
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if logger:
                        logger.debug("에러 발생 후 GPU 메모리 캐시 정리 완료 (CUDA 0번)")
            except:
                pass
        
        if logger:
            logger.error(f"❌ 인과효과 추정 실패: {e}")
        raise

def extract_significance(estimate):
    """
    CausalEstimate 객체에서 p-value와 confidence_intervals를 추출합니다.
    
    Args:
        estimate: CausalEstimate 객체
    
    Returns:
        tuple: (p_value, confidence_intervals)
    """
    p_value = None
    confidence_intervals = None

    try:
        sig = estimate.test_stat_significance()
        # test_stat_significance는 dict 또는 dict 리스트를 반환할 수 있음
        if isinstance(sig, dict):
            p_value = sig.get("p_value")
        elif isinstance(sig, list) and sig:
            first_sig = sig[0]
            if isinstance(first_sig, dict):
                p_value = first_sig.get("p_value")
    except Exception:
        pass
    try:
        # get_confidence_intervals는 없는 경우 AttributeError가 발생할 수 있음
        confidence_intervals = estimate.get_confidence_intervals()
    except Exception:
        confidence_intervals = getattr(estimate, "confidence_intervals", None)

    return p_value[0], confidence_intervals

def calculate_refutation_pvalue(refutation_result, test_type="placebo", logger=None):
    """
    Refutation 테스트 결과의 p-value를 계산합니다.
    
    Args:
        refutation_result: CausalRefutation 객체
        test_type: 테스트 타입 ("placebo", "unobserved", "subset", "dummy")
    
    Returns:
        float: p-value (계산 불가능한 경우 None)
    """
    try:
        log = logger or logging.getLogger(__name__)
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
        log.error(f"calculate_refutation_pvalue 실패: {e}")
        return None


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
            num_simulations=20
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
            num_simulations=20
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
            num_simulations=20
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
        refute_dummys = model.refute_estimate(
            identified_estimand, estimate,
            method_name="dummy_outcome_refuter",
            num_simulations=20
        )
        refute_dummy = refute_dummys[0]
        validation_results['dummy'] = refute_dummy
        
        # for dummy in refute_dummy:
        #     logger.info(f"refute_dummy 결과1: {dummy}")
        #     logger.info(f"refute_dummy 결과2: {dir(dummy)}")
        p_value = calculate_refutation_pvalue(refute_dummy, "dummy", logger)
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
        logger.info("검증 결과 요약")
        logger.info("="*60)
        
        # 가상 원인 테스트
        if validation_results.get('placebo'):
            placebo = validation_results['placebo']
            effect_change = abs(placebo.new_effect - placebo.estimated_effect)
            p_value = calculate_refutation_pvalue(placebo, "placebo")
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
            status = "통과" if abs(dummy.new_effect) < 0.01 else "실패"
            logger.info(f"더미 결과 테스트: {status}")
            logger.info(f"  - 더미 결과 추정치: {dummy.new_effect:.6f}")
            if p_value is not None:
                logger.info(f"  - P-value: {p_value:.6f}")
                logger.info(f"  - 통계적 유의성: {'유의함' if p_value <= 0.05 else '유의하지 않음'}")
    
    return validation_results

def run_sensitivity_analysis(model, identified_estimand, estimate, logger=None):
    """민감도 분석을 실행하는 함수"""
    if logger:
        logger.info("="*60)
        logger.info("민감도 분석 실행 시작")
        logger.info("="*60)
        logger.info("효과 강도 범위: 0.0 ~ 0.5")
        logger.info("그리드 포인트 수: 5x5 = 25개")
        logger.info("시뮬레이션 수: 50회")
    
    try:
        grid = np.linspace(0.0, 0.5, 5)
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
                        num_simulations=50
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

def _json_default(obj):
    """JSON 직렬화 보조: numpy/pandas 객체를 기본 타입으로 변환"""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "isoformat"):  # datetime, Timestamp 등
        return obj.isoformat()
    return str(obj)


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
    
    # 메타데이터 저장 (numpy 타입을 Python 기본 타입으로 변환)
    metadata = {
        "experiment_id": experiment_id,
        "graph_name": graph_name,
        "treatment": estimate._treatment_name[0] if isinstance(estimate._treatment_name, list) else estimate._treatment_name,
        "outcome": estimate._outcome_name[0] if isinstance(estimate._outcome_name, list) else estimate._outcome_name,
        "ate_value": float(estimate.value) if estimate.value is not None else None,
        "control_value": float(estimate.control_value) if estimate.control_value is not None else None,
        "treatment_value": float(estimate.treatment_value) if estimate.treatment_value is not None else None,
        "estimator_type": type(estimate.estimator).__name__ if hasattr(estimate, 'estimator') else None,
        "saved_at": datetime.now().isoformat()
    }
    
    metadata_filename = f"metadata_{experiment_id}.json"
    metadata_file = checkpoint_path / metadata_filename
    
    try:
        # pickle 불가능한 BootstrapEstimates 제거
        if hasattr(estimate.estimator, '_bootstrap_estimates'):
            estimate.estimator._bootstrap_estimates = None

        # CausalEstimate 객체 저장
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(estimate, f)
        
        # 메타데이터 저장
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_default)
        
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


# ============================================================================
# 실험 관리 함수
# ============================================================================

def _get_graph_files(
    config: Dict[str, Any],
    data_dir_path: Path,
    graph_data_dir: str
) -> List[str]:
    """
    설정에 따라 그래프 파일 목록을 반환하는 내부 함수
    
    Args:
        config: 설정 딕셔너리
        data_dir_path: 데이터 디렉토리 경로
        graph_data_dir: 그래프 데이터 디렉토리명
    
    Returns:
        그래프 파일 경로 리스트
    """
    from . import utils
    graphs = config.get("graphs", [])
    auto_extract_treatments = config.get("auto_extract_treatments", False)
    
    if auto_extract_treatments:
        found_graphs = utils.find_all_graph_files(data_dir_path, graph_data_dir)
        return [str(g) for g in found_graphs]
    
    graph_files = []
    for graph in graphs:
        if isinstance(graph, str):
            graph_path = data_dir_path / graph_data_dir / graph
            if graph_path.exists():
                graph_files.append(str(graph_path))
            else:
                graph_path = Path(graph)
                if graph_path.exists():
                    graph_files.append(str(graph_path))
    
    return graph_files


def _extract_treatments_from_graphs(
    graph_files: List[str],
    auto_extract: bool
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    그래프 파일들에서 treatment와 outcome을 추출하는 내부 함수
    
    Args:
        graph_files: 그래프 파일 경로 리스트
        auto_extract: 자동 추출 여부
    
    Returns:
        (graph_treatments_map, graph_outcomes_map) 튜플
    """
    from . import utils
    graph_treatments_map = {}
    graph_outcomes_map = {}
    
    if auto_extract:
        for graph_file in graph_files:
            graph_path = Path(graph_file)
            extracted_treatments = utils.extract_treatments_from_graph(graph_path)
            
            if extracted_treatments:
                graph_treatments_map[graph_file] = [
                    t["treatment_var"] for t in extracted_treatments 
                    if t.get("treatment_var")
                ]
                if extracted_treatments[0].get("outcome"):
                    graph_outcomes_map[graph_file] = extracted_treatments[0]["outcome"]
    
    return graph_treatments_map, graph_outcomes_map


def _sort_estimators(estimators: List[str]) -> List[str]:
    """
    estimator 리스트를 정렬하는 내부 함수 (linear_regression, tabpfn 우선)
    
    Args:
        estimators: estimator 리스트
    
    Returns:
        정렬된 estimator 리스트
    """
    sorted_estimators = []
    priority_estimators = ["linear_regression", "tabpfn"]
    
    for est in priority_estimators:
        if est in estimators:
            sorted_estimators.append(est)
    
    for est in estimators:
        if est not in sorted_estimators:
            sorted_estimators.append(est)
    
    return sorted_estimators


def create_experiment_list(
    config: Dict[str, Any],
    data_dir_path: Path,
    graph_data_dir: str
) -> List[Tuple[str, str, str, str]]:
    """
    config.json에서 experiment_list를 읽어서 실험 조합 리스트 생성
    
    Args:
        config: 설정 딕셔너리
        data_dir_path: 데이터 디렉토리 경로
        graph_data_dir: 그래프 데이터 디렉토리명
    
    Returns:
        실험 조합 리스트 [(graph_file, treatment, outcome, estimator), ...]
    """
    # config.json에 experiment_list가 정의되어 있는지 확인
    experiment_list_config = config.get("experiment_list", [])
    
    if experiment_list_config:
        # config.json에서 직접 정의된 experiment_list 사용
        experiment_combinations = []
        graph_data_path = data_dir_path / graph_data_dir
        
        for exp in experiment_list_config:
            if isinstance(exp, list) and len(exp) >= 4:
                # 배열 형식: ["graph_1.dot", "BFR_OCTR_CT", "ACQ_180_YN", "tabpfn"]
                graph_name, treatment, outcome, estimator = exp[0], exp[1], exp[2], exp[3]
            elif isinstance(exp, dict):
                # 딕셔너리 형식: {"graph": "graph_1.dot", "treatment": "BFR_OCTR_CT", ...}
                graph_name = exp.get("graph", "")
                treatment = exp.get("treatment", "")
                outcome = exp.get("outcome", "ACQ_180_YN")
                estimator = exp.get("estimator", "tabpfn")
            else:
                print(f"⚠️ 잘못된 experiment_list 형식: {exp}")
                continue
            
            # 그래프 파일 경로 확인
            graph_path = graph_data_path / graph_name
            if not graph_path.exists():
                # 절대 경로로 시도
                graph_path = Path(graph_name)
                if not graph_path.exists():
                    print(f"⚠️ 그래프 파일을 찾을 수 없습니다: {graph_name}")
                    continue
            
            experiment_combinations.append(
                (str(graph_path), treatment, outcome, estimator)
            )
        
        return experiment_combinations
    
    # 기존 방식 (하위 호환성)
    treatments = config.get("treatments", [])
    outcomes = config.get("outcomes", ["ACQ_180_YN"])
    estimators = config.get("estimators", ["tabpfn"])
    auto_extract_treatments = config.get("auto_extract_treatments", False)
    
    # 그래프 파일 경로 처리
    graph_files = _get_graph_files(config, data_dir_path, graph_data_dir)
    
    if not graph_files:
        return []
    
    # treatment 자동 추출
    graph_treatments_map, graph_outcomes_map = _extract_treatments_from_graphs(
        graph_files, auto_extract_treatments
    )
    
    # estimator 정렬
    sorted_estimators = _sort_estimators(estimators)
    
    # 실험 조합 생성
    if auto_extract_treatments and graph_treatments_map:
        experiment_combinations = []
        for graph_file in graph_files:
            graph_treatments = graph_treatments_map.get(graph_file, treatments)
            graph_outcome = graph_outcomes_map.get(
                graph_file, 
                outcomes[0] if outcomes else "ACQ_180_YN"
            )
            
            for treatment in graph_treatments:
                for estimator in sorted_estimators:
                    experiment_combinations.append(
                        (graph_file, treatment, graph_outcome, estimator)
                    )
    else:
        experiment_combinations = list(itertools.product(
            graph_files,
            treatments,
            outcomes,
            sorted_estimators
        ))
    
    return experiment_combinations


def prepare_data_for_causal_model(
    merged_df: pd.DataFrame,
    config: Dict[str, Any],
    data_dir_path: Path,
    graph_data_dir: str
) -> pd.DataFrame:
    """
    인과 모델을 위한 데이터 준비 (그래프 변수에 맞게 데이터 정리)
    
    Args:
        merged_df: 병합된 데이터프레임
        config: 설정 딕셔너리
        data_dir_path: 데이터 디렉토리 경로
        graph_data_dir: 그래프 데이터 디렉토리명
    
    Returns:
        정리된 데이터프레임
    """
    from . import utils
    treatments = config.get("treatments", [])
    outcomes = config.get("outcomes", ["ACQ_180_YN"])
    auto_extract_treatments = config.get("auto_extract_treatments", False)
    
    # 그래프 파일 경로 처리
    graph_files = _get_graph_files(config, data_dir_path, graph_data_dir)
    
    if not graph_files:
        return merged_df
    
    # 모든 그래프의 변수 수집
    all_graph_variables = set()
    for graph_file in graph_files:
        graph_path = Path(graph_file)
        try:
            causal_graph = utils.create_causal_graph(str(graph_path))
            all_graph_variables.update(causal_graph.nodes())
        except Exception as e:
            print(f"⚠️ 그래프 파일 로드 실패 ({graph_path.name}): {e}")
    
    # treatment 자동 추출
    graph_treatments_map, graph_outcomes_map = _extract_treatments_from_graphs(
        graph_files, auto_extract_treatments
    )
    
    # 데이터 정리
    all_treatments = set()
    all_outcomes = set()
    for graph_file in graph_files:
        if graph_file in graph_treatments_map:
            all_treatments.update(graph_treatments_map[graph_file])
        if graph_file in graph_outcomes_map:
            all_outcomes.add(graph_outcomes_map[graph_file])
    if not auto_extract_treatments:
        all_treatments.update(treatments)
    if not graph_outcomes_map:
        all_outcomes.update(outcomes)
    
    essential_vars = all_treatments | all_outcomes | {"JHNT_CTN", "JHNT_MBN"}
    stratification_vars = {"HOPE_JSCD1_NAME"}
    required_vars = list(all_graph_variables | essential_vars | stratification_vars)
    
    merged_df_clean = utils.clean_dataframe_for_causal_model(
        merged_df, 
        required_vars=required_vars, 
        logger=None
    )
    
    data_variables = set(merged_df_clean.columns)
    vars_to_keep = (all_graph_variables | essential_vars | stratification_vars) & data_variables
    vars_to_remove = data_variables - vars_to_keep
    
    if vars_to_remove:
        print(f"🗑️ 그래프에 정의되지 않은 변수 제거 중 ({len(vars_to_remove)}개)...")
        merged_df_clean = merged_df_clean[list(vars_to_keep)]
    
    print(f"✅ 정리된 데이터: {len(merged_df_clean)}건, {len(merged_df_clean.columns)}개 변수")
    
    return merged_df_clean


# ============================================================================
# 분석 실행 함수
# ============================================================================

def run_analysis_without_preprocessing(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    logger: Optional[logging.Logger] = None,
    experiment_id: Optional[str] = None,
    job_category: Optional[str] = None,
    training_size: int = 5000,
    tabpfn_config: Optional[Dict[str, Any]] = None,
    do_refutation: bool = False
) -> Dict[str, Any]:
    """
    전처리된 데이터를 사용하여 인과추론 분석을 수행하는 함수
    (estimation → refutation → prediction만 수행)
    
    Args:
        merged_df_clean (pd.DataFrame): 전처리 및 정리된 데이터프레임
        graph_file (str): 그래프 파일 경로
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        logger (Optional[logging.Logger]): 로거 객체
        experiment_id (Optional[str]): 실험 ID (선택적)
        job_category (Optional[str]): 직종소분류명 (checkpoint 저장 경로에 사용)
        training_size (int): Train set 크기 (기본값: 5000)
        do_refutation (bool): Refutation 실행 여부 (기본값: False)
    
    Returns:
        Dict[str, Any]: 분석 결과 딕셔너리
    """
    from . import utils
    try:
        step_times = {}
        step_start = time.time()
        
        if experiment_id:
            print(f"\n{'='*80}")
            print(f"실험 ID: {experiment_id}")
            print(f"그래프: {Path(graph_file).name}")
            print(f"Treatment: {treatment}, Outcome: {outcome}")
            print(f"Estimator: {estimator}")
            print(f"{'='*80}\n")
        
        # 1. 그래프 로드
        print("1️⃣ 인과 그래프 로드 중...")
        step_start = time.time()
        causal_graph = utils.create_causal_graph(graph_file)
        step_times['그래프 로드'] = time.time() - step_start
        
        # 2. 데이터 필터링
        print("2️⃣ 그래프 변수에 맞게 데이터 필터링 중...")
        step_start = time.time()
        
        graph_variables = set(causal_graph.nodes())
        data_variables = set(merged_df_clean.columns)
        essential_vars = {treatment, outcome, "JHNT_CTN", "JHNT_MBN"}
        stratification_vars = {"HOPE_JSCD1_NAME"}
        vars_to_keep = (graph_variables | essential_vars | stratification_vars) & data_variables
        df_for_analysis = merged_df_clean[list(vars_to_keep)].copy()
        
        missing_vars = [var for var in [treatment, outcome] if var not in df_for_analysis.columns]
        if missing_vars:
            raise ValueError(f"필수 변수가 데이터에 없습니다: {missing_vars}")
        
        step_times['데이터 필터링'] = time.time() - step_start
        
        # 3. Train/Test Split (고정 개수 샘플링)
        print(f"3️⃣ Train/Test Split 중...")
        step_start = time.time()
        
        total_size = len(df_for_analysis)
        outcome_data = df_for_analysis[outcome]
        is_binary = outcome_data.nunique() <= 2 and outcome_data.dtype in ['int64', 'int32', 'bool']
        
        # 데이터가 training_size보다 작거나 같은 경우 8:2 비율로 split
        if total_size <= training_size:
            print(f"⚠️ 전체 데이터({total_size}건)가 training_size({training_size}건)보다 작거나 같습니다. 8:2 비율로 split합니다.")
            if is_binary:
                # Binary outcome인 경우 stratify 사용
                df_train, df_test = train_test_split(
                    df_for_analysis,
                    test_size=0.2,
                    random_state=42,
                    stratify=outcome_data
                )
            else:
                # 연속형 outcome인 경우 stratify 없이 split
                df_train, df_test = train_test_split(
                    df_for_analysis,
                    test_size=0.2,
                    random_state=42
                )
        else:
            # training_size만큼 샘플링하여 train set 생성, 나머지는 test set
            print(f"📊 Train: {training_size}개, Test: 나머지 ({total_size - training_size}개)")
            if is_binary:
                # Binary outcome인 경우 stratify 사용
                df_train, df_test = train_test_split(
                    df_for_analysis,
                    train_size=training_size,
                    random_state=42,
                    stratify=outcome_data
                )
            else:
                # 연속형 outcome인 경우 stratify 없이 샘플링
                df_train, df_test = train_test_split(
                    df_for_analysis,
                    train_size=training_size,
                    random_state=42
                )
        
        print(f"✅ Train set: {len(df_train)}건, Test set: {len(df_test)}건")
        step_times['Train/Test Split'] = time.time() - step_start
        
        # 3-1. 컬럼별 타입 체크 (int/str 혼합 감지)
        print("🔍 컬럼별 타입 체크 중...")
        for col in df_train.columns:
            if df_train[col].dtype == 'object':
                non_null = df_train[col].dropna()
                if len(non_null) > 0:
                    types = set(type(v).__name__ for v in non_null)
                    if len(types) > 1:
                        print(f"⚠️ 컬럼 '{col}'에 타입 혼합 감지: {types}")
                        if logger:
                            logger.warning(f"컬럼 '{col}'에 타입 혼합 감지: {types}")
        
        # 3-2. Categorical 변수 Ordinal Encoding (TabPFN용)
        ordinal_encoder = None
        categorical_columns = []
        if estimator == 'tabpfn':
            print("🔢 Categorical 변수 Ordinal Encoding 중...")
            step_start = time.time()
            
            # Categorical 변수 찾기 (Treatment/Outcome 제외)
            categorical_columns = [
                col for col in df_train.select_dtypes(include=['object', 'string', 'category']).columns
                if col not in [treatment, outcome]
            ]
            
            if categorical_columns:
                print(f"   발견된 categorical 변수: {categorical_columns}")
                
                # OrdinalEncoder 생성
                ordinal_encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    dtype=np.int64
                )
                
                # 전처리 함수: NaN 처리 및 문자열 변환
                def preprocess_for_encoding(df, cols):
                    df_processed = df.copy()
                    for col in cols:
                        if col in df_processed.columns:
                            df_processed[col] = df_processed[col].fillna('__nan__').astype(str)
                    return df_processed
                
                # Train/Test 데이터 전처리 및 인코딩
                df_train_processed = preprocess_for_encoding(df_train, categorical_columns)
                df_test_processed = preprocess_for_encoding(df_test, categorical_columns)
                
                # 존재하는 컬럼만 인코딩
                existing_cols = [col for col in categorical_columns if col in df_train_processed.columns]
                if not existing_cols:
                    print("   ⚠️ categorical 컬럼이 데이터에 없습니다.")
                    ordinal_encoder = None
                    categorical_columns = []
                else:
                    df_train_encoded = df_train.copy()
                    df_train_encoded[existing_cols] = ordinal_encoder.fit_transform(df_train_processed[existing_cols])
                    
                    df_test_encoded = df_test.copy()
                    test_existing_cols = [col for col in existing_cols if col in df_test_processed.columns]
                    if len(test_existing_cols) < len(existing_cols):
                        missing_cols = [col for col in existing_cols if col not in test_existing_cols]
                        print(f"   ⚠️ Test 데이터에 일부 categorical 컬럼이 없습니다: {missing_cols}")
                    df_test_encoded[test_existing_cols] = ordinal_encoder.transform(df_test_processed[test_existing_cols])
                
                    # Unknown categories 로깅
                    for col in existing_cols:
                        if col in df_test.columns and col in df_train.columns:
                            unknown_cats = set(df_test[col].dropna().unique()) - set(df_train[col].dropna().unique())
                            if unknown_cats:
                                unknown_count = df_test[col].isin(unknown_cats).sum()
                                print(f"   ⚠️ '{col}': {unknown_count}개 unknown categories → -1로 인코딩됨")
                    
                    df_train, df_test = df_train_encoded, df_test_encoded
                    categorical_columns = existing_cols
                    print(f"✅ Ordinal Encoding 완료: {len(categorical_columns)}개 변수")
            else:
                print("   Categorical 변수가 없습니다.")
            
            step_times['Ordinal Encoding'] = time.time() - step_start
        
        # 4. 인과모델 생성
        print("4️⃣ 인과모델 생성 중...")
        step_start = time.time()
        model = CausalModel(
            data=df_train,
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph
        )
        step_times['인과모델 생성'] = time.time() - step_start
        
        # 5. 인과효과 식별
        print("5️⃣ 인과효과 식별 중...")
        step_start = time.time()
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        step_times['인과효과 식별'] = time.time() - step_start
        
        # 6. 인과효과 추정
        print("6️⃣ 인과효과 추정 중...")
        step_start = time.time()
        estimate = estimate_causal_effect(
            model,
            identified_estimand,
            estimator,
            logger,
            tabpfn_config=tabpfn_config
        )
        
        # OrdinalEncoder를 estimate 객체에 저장 (예측 시 사용)
        if ordinal_encoder is not None:
            estimate._ordinal_encoder = ordinal_encoder
            estimate._categorical_columns = categorical_columns
            print(f"💾 OrdinalEncoder를 estimate 객체에 저장: {len(categorical_columns)}개 변수")
        
        step_times['인과효과 추정'] = time.time() - step_start
        
        # 6-1. Checkpoint 저장
        checkpoint_path = None
        if experiment_id:
            try:
                script_dir = Path(__file__).parent.parent
                checkpoint_dir = script_dir / "data" / "checkpoint"
                
                if job_category:
                    job_category_safe = str(job_category).replace("/", "_").replace("\\", "_").replace(" ", "_")
                    checkpoint_dir = checkpoint_dir / job_category_safe
                
                graph_name = Path(graph_file).stem if graph_file else None
                checkpoint_path = save_checkpoint(
                    estimate,
                    checkpoint_dir,
                    experiment_id,
                    graph_name=graph_name,
                    logger=logger
                )
            except Exception as e:
                if logger:
                    logger.warning(f"Checkpoint 저장 실패 (계속 진행): {e}")
                print(f"⚠️ Checkpoint 저장 실패 (계속 진행): {e}")
        
        # 6-2. Refutation (선택 사항)
        validation_results = {}
        if do_refutation:
            print("🛡️ Refutation 테스트 실행 중...")
            step_start = time.time()
            validation_results = run_validation_tests(model, identified_estimand, estimate, logger=logger)
            step_times['Refutation'] = time.time() - step_start
        
        # 7. 예측
        print("7️⃣ 예측 중...")
        step_start = time.time()
        essential_vars_for_pred = {treatment, outcome}
        if outcome in df_test.columns:
            df_test = df_test.copy()
            df_test[f"{outcome}_actual"] = df_test[outcome].copy()
        
        df_test_clean = utils.clean_dataframe_for_causal_model(
            df_test,
            required_vars=list(essential_vars_for_pred) + [f"{outcome}_actual"] if f"{outcome}_actual" in df_test.columns else list(essential_vars_for_pred),
            logger=logger
        )
        # TabPFN 배치 크기 설정 (config에서 가져오기, 기본값: 64)
        metrics, df_with_predictions = predict_conditional_expectation(
            estimate, df_test_clean, logger=logger
        )
        step_times['예측'] = time.time() - step_start
        
        # 직종소분류 정보 추가 (있는 경우)
        if job_category is not None:
            df_with_predictions['job_category'] = job_category
        
        # 예측 결과 저장
        if experiment_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{experiment_id}_{timestamp}.xlsx"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.xlsx"
        
        step_start = time.time()
        excel_path = utils.save_predictions_to_excel(df_with_predictions, filename=filename, logger=logger)
        step_times['예측 결과 저장'] = time.time() - step_start
        
        # 8. 결과 출력
        print("\n" + "="*60)
        print("📊 추정 결과 요약")
        print("="*60)
        print(f"  ATE (Average Treatment Effect): {estimate.value:.6f}")
        
        if validation_results:
            print(f"\n🔬 Refutation 결과 요약:")
            for test_name, res in validation_results.items():
                if res:
                    # status 판단 (run_validation_tests의 로직 참고)
                    if test_name == 'placebo':
                        status = "통과" if abs(res.new_effect - res.estimated_effect) < 0.01 else "실패"
                    elif test_name == 'unobserved':
                        change_rate = abs(res.new_effect - res.estimated_effect) / abs(res.estimated_effect) if abs(res.estimated_effect) > 0 else float('inf')
                        status = "강건함" if change_rate < 0.2 else "민감함"
                    elif test_name == 'subset':
                        change_rate = abs(res.new_effect - res.estimated_effect) / abs(res.estimated_effect) if abs(res.estimated_effect) > 0 else float('inf')
                        status = "통과" if change_rate < 0.1 else "실패"
                    elif test_name == 'dummy':
                        status = "통과" if abs(res.new_effect) < 0.01 else "실패"
                    else:
                        status = "완료"
                    print(f"  - {test_name}: {status} (New Effect: {res.new_effect:.6f})")
        
        if metrics:
            print(f"\n📈 예측 성능:")
            if metrics.get('accuracy') is not None:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if metrics.get('f1_score') is not None:
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
            if metrics.get('auc') is not None:
                print(f"  AUC: {metrics['auc']:.4f}")
        print("="*60)
        
        if not do_refutation:
            print("ℹ️  민감도 분석/Refutation 테스트는 별도로 실행하거나 config에서 활성화하세요.")
            print("="*60 + "\n")
        
        # 9. TabPFN 메모리 정리 (분석 완료 후)
        if estimator == 'tabpfn':
            cleanup_tabpfn_memory(estimate, device_id=0, logger=logger)
        
        total_time = sum(step_times.values())
        step_times['전체'] = total_time
        
        print(f"\n✅ 분석 완료! (총 소요 시간: {total_time:.2f}초)")
        
        res_dict = {
            "status": "success",
            "estimate": estimate,
            "validation_results": validation_results,
            "metrics": metrics,
            "excel_path": excel_path,
            "checkpoint_path": checkpoint_path,
            "step_times": step_times,
            "train_size": len(df_train),
            "test_size": len(df_test)
        }
        
        # CSV 로깅을 위해 Refutation 결과를 평탄화하여 추가
        if validation_results:
            for test_name, res in validation_results.items():
                if res:
                    p_val = calculate_refutation_pvalue(res, test_name, logger)
                    res_dict[f"{test_name}_pvalue"] = p_val
                    
                    if test_name == 'placebo':
                        res_dict['placebo_passed'] = abs(res.new_effect - res.estimated_effect) < 0.01
                    elif test_name == 'unobserved':
                        change_rate = abs(res.new_effect - res.estimated_effect) / abs(res.estimated_effect) if abs(res.estimated_effect) > 0 else float('inf')
                        res_dict['unobserved_passed'] = change_rate < 0.2
                    elif test_name == 'subset':
                        change_rate = abs(res.new_effect - res.estimated_effect) / abs(res.estimated_effect) if abs(res.estimated_effect) > 0 else float('inf')
                        res_dict['subset_passed'] = change_rate < 0.1
                    elif test_name == 'dummy':
                        res_dict['dummy_passed'] = abs(res.new_effect) < 0.01
        
        return res_dict
        
    except Exception as e:
        if logger:
            logger.error(f"분석 중 오류 발생: {e}")
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_single_experiment(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    experiment_id: str,
    logger: Optional[logging.Logger] = None,
    split_by_job_category: bool = True,
    training_size: int = 5000,
    tabpfn_config: Optional[Dict[str, Any]] = None,
    do_refutation: bool = False
) -> Dict[str, Any]:
    """
    단일 실험을 실행합니다
    
    Args:
        merged_df_clean (pd.DataFrame): 전처리 및 정리된 데이터프레임
        graph_file (str): 그래프 파일 경로
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        experiment_id (str): 실험 ID
        logger (Optional[logging.Logger]): 로거 객체
        split_by_job_category (bool): 직종소분류별로 분리하여 실험 실행 여부
        training_size (int): Train set 크기 (기본값: 5000)
        do_refutation (bool): Refutation 실행 여부 (기본값: False)
    
    Returns:
        Dict[str, Any]: 실험 결과 딕셔너리
    """
    from . import utils
    start_time = datetime.now()
    try:
        # 직종소분류별로 분리하여 실험 실행
        if split_by_job_category and "HOPE_JSCD1_NAME" in merged_df_clean.columns:
            job_categories = merged_df_clean["HOPE_JSCD1_NAME"].dropna().unique()
            print(f"📊 직종소분류별 실험 실행: {len(job_categories)}개 직종소분류")
            
            all_results = []
            all_predictions = []
            all_metrics = []
            job_category_list = []  # 직종소분류 리스트 저장 (성공한 것만)
            
            for job_category in job_categories:
                job_df = merged_df_clean[merged_df_clean["HOPE_JSCD1_NAME"] == job_category].copy()
                
                if len(job_df) < 10:
                    if logger:
                        logger.warning(f"직종소분류 '{job_category}' 데이터가 너무 적어 건너뜁니다: {len(job_df)}건")
                    print(f"⚠️ 직종소분류 '{job_category}' 데이터가 너무 적어 건너뜁니다: {len(job_df)}건")
                    continue
                
                job_category_safe = str(job_category).replace("/", "_").replace("\\", "_").replace(" ", "_")
                job_experiment_id = f"{experiment_id}_{job_category_safe}"
                
                print(f"\n  🔹 직종소분류: {job_category} ({len(job_df)}건)")
                
                try:
                    job_result = run_analysis_without_preprocessing(
                        merged_df_clean=job_df,
                        graph_file=graph_file,
                        treatment=treatment,
                        outcome=outcome,
                        estimator=estimator,
                        logger=logger,
                        experiment_id=job_experiment_id,
                        job_category=job_category,
                        training_size=training_size,
                        tabpfn_config=tabpfn_config,
                        do_refutation=do_refutation
                    )
                    
                    all_results.append(job_result)
                    job_category_list.append(job_category)  # 성공한 직종소분류만 저장
                    
                    if job_result.get("excel_path"):
                        try:
                            pred_df = pd.read_excel(job_result["excel_path"])
                            all_predictions.append(pred_df)
                        except:
                            pass
                    
                    if job_result.get("metrics"):
                        all_metrics.append(job_result["metrics"])
                    
                    # TabPFN 사용 시 각 실험 후 GPU 메모리 정리 (CUDA 0번)
                    if estimator == 'tabpfn' and job_result.get('estimate'):
                        cleanup_tabpfn_memory(job_result['estimate'], device_id=0, logger=logger)
                        
                except Exception as e:
                    # 실패 시에도 GPU 메모리 정리 (CUDA 0번)
                    if estimator == 'tabpfn':
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.set_device(0)  # CUDA 0번으로 설정
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        except:
                            pass
                    
                    if logger:
                        logger.error(f"직종소분류 '{job_category}' 실험 실패: {e}")
                    print(f"  ❌ 직종소분류 '{job_category}' 실험 실패: {e}")
                    continue
            
            if not all_results:
                raise ValueError("모든 직종소분류 실험이 실패했습니다.")
            
            # 예측 결과 합치기 및 평가지표 계산
            if all_predictions:
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                
                # 각 직종소분류별 평가지표 수집
                job_category_metrics_list = []
                for idx, job_result in enumerate(all_results):
                    job_category = job_category_list[idx] if idx < len(job_category_list) else None
                    job_metrics = job_result.get("metrics", {})
                    
                    if job_category and job_metrics:
                        job_metric_row = {
                            "job_category": job_category,
                            "train_size": job_result.get("train_size", 0),
                            "test_size": job_result.get("test_size", 0),
                            "accuracy": job_metrics.get("accuracy"),
                            "f1_score": job_metrics.get("f1_score"),
                            "auc": job_metrics.get("auc")
                        }
                        job_category_metrics_list.append(job_metric_row)
                
                # 전체 데이터 평가지표 계산
                combined_metrics = {'accuracy': None, 'f1_score': None, 'auc': None}
                if all_metrics:
                    actual_outcome_col = f"{outcome}_actual"
                    if actual_outcome_col in combined_predictions.columns and outcome in combined_predictions.columns:
                        actual_y = combined_predictions[actual_outcome_col]
                        predicted_y = combined_predictions[outcome]
                        prob_col = f"{outcome}_prob"
                        prob_y = combined_predictions[prob_col] if prob_col in combined_predictions.columns else None
                        
                        combined_metrics = utils.calculate_metrics(actual_y, predicted_y, prob_y=prob_y, logger=logger)
                
                # 전체 평가지표를 리스트에 추가
                if combined_metrics:
                    overall_metric_row = {
                        "job_category": "전체",
                        "train_size": sum([r.get("train_size", 0) for r in all_results]),
                        "test_size": sum([r.get("test_size", 0) for r in all_results]),
                        "accuracy": combined_metrics.get("accuracy"),
                        "f1_score": combined_metrics.get("f1_score"),
                        "auc": combined_metrics.get("auc")
                    }
                    job_category_metrics_list.append(overall_metric_row)
                
                # 평가지표를 DataFrame으로 변환하여 CSV 저장
                if job_category_metrics_list:
                    metrics_df = pd.DataFrame(job_category_metrics_list)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    script_dir = Path(__file__).parent.parent
                    output_dir = script_dir / "log"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    metrics_csv_path = output_dir / f"metrics_{experiment_id}_{timestamp}.csv"
                    metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
                    if logger:
                        logger.info(f"✅ 평가지표 CSV 저장 완료: {metrics_csv_path}")
                    print(f"✅ 평가지표 CSV 저장 완료: {metrics_csv_path}")
                
                # 예측 결과 Excel 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_path = utils.save_predictions_to_excel(
                    combined_predictions, 
                    filename=f"predictions_{experiment_id}_combined_{timestamp}.xlsx",
                    logger=logger
                )
            else:
                combined_metrics = {}
                excel_path = None
            
            base_result = all_results[0]
            result = {
                "status": "success",
                "estimate": base_result.get("estimate"),
                "validation_results": base_result.get("validation_results", {}),
                "sensitivity_df": base_result.get("sensitivity_df"),
                "metrics": combined_metrics,
                "excel_path": excel_path,
                "step_times": base_result.get("step_times", {}),
                "train_size": sum([r.get("train_size", 0) for r in all_results]),
                "test_size": sum([r.get("test_size", 0) for r in all_results]),
                "job_category_results": all_results,
                "num_job_categories": len(all_results)
            }
        else:
            result = run_analysis_without_preprocessing(
                merged_df_clean=merged_df_clean,
                graph_file=graph_file,
                treatment=treatment,
                outcome=outcome,
                estimator=estimator,
                logger=logger,
                experiment_id=experiment_id,
                training_size=training_size,
                tabpfn_config=tabpfn_config,
                do_refutation=do_refutation
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        metrics = result.get("metrics", {})
        estimate = result.get("estimate")
        
        ate_value = None
        if estimate and hasattr(estimate, 'value'):
            ate_value = estimate.value
        
        return_dict = {
            "experiment_id": experiment_id,
            "status": "success",
            "duration_seconds": duration,
            "graph": graph_file,
            "graph_name": Path(graph_file).stem,
            "treatment": treatment,
            "outcome": outcome,
            "estimator": estimator,
            "ate_value": ate_value,
            "metrics": metrics,
            "accuracy": metrics.get("accuracy") if metrics else None,
            "f1_score": metrics.get("f1_score") if metrics else None,
            "auc": metrics.get("auc") if metrics else None,
            "excel_path": result.get("excel_path"),
            "train_size": result.get("train_size"),
            "test_size": result.get("test_size"),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        # Refutation 결과 추가
        for key in ['placebo_passed', 'placebo_pvalue', 'unobserved_passed', 'unobserved_pvalue', 
                    'subset_passed', 'subset_pvalue', 'dummy_passed', 'dummy_pvalue']:
            if key in result:
                return_dict[key] = result[key]
                
        return return_dict
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "experiment_id": experiment_id,
            "status": "failed",
            "duration_seconds": duration,
            "graph": graph_file,
            "treatment": treatment,
            "outcome": outcome,
            "estimator": estimator,
            "error": str(e),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }


def run_inference(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    checkpoint_dir: Path,
    treatment: str,
    outcome: str,
    estimator: str,
    logger: Optional[logging.Logger] = None,
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inference 모드: checkpoint에서 모델을 로드하여 예측만 수행하는 함수
    
    Args:
        merged_df_clean (pd.DataFrame): 전처리 및 정리된 데이터프레임
        graph_file (str): 그래프 파일 경로
        checkpoint_dir (Path): checkpoint 디렉토리 경로
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        logger (Optional[logging.Logger]): 로거 객체
        experiment_id (Optional[str]): 실험 ID (선택적)
    
    Returns:
        Dict[str, Any]: 예측 결과 딕셔너리
    """
    from . import utils
    try:
        step_times = {}
        step_start = time.time()
        
        if experiment_id:
            print(f"\n{'='*80}")
            print(f"Inference 모드 - 실험 ID: {experiment_id}")
            print(f"그래프: {Path(graph_file).name}")
            print(f"Treatment: {treatment}, Outcome: {outcome}, Estimator: {estimator}")
            print(f"{'='*80}\n")
        
        graph_name = Path(graph_file).stem
        
        # 직종소분류별로 분리하여 예측
        if "HOPE_JSCD1_NAME" in merged_df_clean.columns:
            job_categories = merged_df_clean["HOPE_JSCD1_NAME"].dropna().unique()
            print(f"📊 직종소분류별 Inference 실행: {len(job_categories)}개 직종소분류")
            
            all_predictions = []
            all_metrics = []
            
            for job_category in job_categories:
                job_df = merged_df_clean[merged_df_clean["HOPE_JSCD1_NAME"] == job_category].copy()
                
                if len(job_df) == 0:
                    continue
                
                job_category_safe = str(job_category).replace("/", "_").replace("\\", "_").replace(" ", "_")
                job_checkpoint_dir = checkpoint_dir / job_category_safe
                
                print(f"\n  🔹 직종소분류: {job_category} ({len(job_df)}건)")
                
                checkpoint_file = find_checkpoint(
                    job_checkpoint_dir,
                    graph_name,
                    treatment,
                    outcome,
                    estimator,
                    logger
                )
                
                if not checkpoint_file:
                    print(f"  ⚠️ Checkpoint를 찾을 수 없어 건너뜁니다: {job_category}")
                    continue
                
                try:
                    estimate = load_checkpoint(checkpoint_file, logger)
                    
                    essential_vars = {treatment, outcome, "JHNT_CTN", "JHNT_MBN"}
                    data_variables = set(job_df.columns)
                    # causal_graph = utils.create_causal_graph(graph_file)
                    # graph_vars = set(causal_graph.nodes())
                    vars_to_keep = essential_vars | data_variables
                    
                    missing_vars = [var for var in [treatment, outcome] if var not in job_df.columns]
                    if missing_vars:
                        print(f"  ⚠️ 필수 변수가 없어 건너뜁니다: {missing_vars}")
                        continue
                    
                    df_for_prediction = job_df[list(vars_to_keep)].copy()
                    
                    if outcome in df_for_prediction.columns:
                        df_for_prediction[f"{outcome}_actual"] = df_for_prediction[outcome].copy()
                    
                    df_pred_clean = utils.clean_dataframe_for_causal_model(
                        df_for_prediction,
                        required_vars=list(essential_vars) + [f"{outcome}_actual"] if f"{outcome}_actual" in df_for_prediction.columns else list(essential_vars),
                        logger=logger
                    )
                    metrics, df_with_predictions = predict_conditional_expectation(
                        estimate, df_pred_clean, logger=logger
                    )
                    
                    all_predictions.append(df_with_predictions)
                    if metrics:
                        all_metrics.append(metrics)
                    
                    print(f"  ✅ 예측 완료: {len(df_with_predictions)}건")
                    
                except Exception as e:
                    print(f"  ❌ 직종소분류 '{job_category}' 예측 실패: {e}")
                    if logger:
                        logger.error(f"직종소분류 '{job_category}' 예측 실패: {e}")
                    continue
            
            if not all_predictions:
                raise ValueError("모든 직종소분류 예측이 실패했습니다.")
            
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # 통합 메트릭 계산
            combined_metrics = {'accuracy': None, 'f1_score': None, 'auc': None}
            actual_outcome_col = f"{outcome}_actual"
            if actual_outcome_col in combined_predictions.columns and outcome in combined_predictions.columns:
                actual_y = combined_predictions[actual_outcome_col]
                predicted_y = combined_predictions[outcome]
                prob_col = f"{outcome}_prob"
                prob_y = combined_predictions[prob_col] if prob_col in combined_predictions.columns else None
                
                combined_metrics = utils.calculate_metrics(actual_y, predicted_y, prob_y=prob_y, logger=logger)
            
            # 예측 결과 저장
            step_start = time.time()
            if experiment_id:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions_inference_{experiment_id}_combined_{timestamp}.xlsx"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions_inference_combined_{timestamp}.xlsx"
            
            excel_path = utils.save_predictions_to_excel(combined_predictions, filename=filename, logger=logger)
            step_times['예측 결과 저장'] = time.time() - step_start
            
        else:
            raise ValueError("HOPE_JSCD1_NAME 변수가 데이터에 없습니다. 직종소분류별 분리가 불가능합니다.")
        
        total_time = sum(step_times.values())
        step_times['전체'] = total_time
        
        print(f"\n✅ Inference 완료! (총 소요 시간: {total_time:.2f}초)")
        if combined_metrics:
            print(f"   Accuracy: {combined_metrics.get('accuracy', 'N/A')}")
            print(f"   F1 Score: {combined_metrics.get('f1_score', 'N/A')}")
            print(f"   AUC: {combined_metrics.get('auc', 'N/A')}")
        
        return {
            "status": "success",
            "metrics": combined_metrics,
            "excel_path": excel_path,
            "step_times": step_times,
            "data_size": len(combined_predictions)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Inference 중 오류 발생: {e}")
        print(f"❌ Inference 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise
