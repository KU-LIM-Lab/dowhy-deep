import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

# DoWhy의 CausalModel과 Estimator Import
from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator
from dowhy.causal_estimator import CausalEstimate

# validation_pipeline에서 DAG 파서를 사용해야 하므로 경로를 추가해야 할 수 있습니다.
# 현재 환경에서는 이미 추가되어 있다고 가정합니다.
from kubig_experiments.src.dag_parser import extract_roles_general, dot_to_nx

# Top 5 DAG 정보를 로드하고 처리할 경로
DAG_DIR = Path("./kubig_experiments/dags/")
OUTCOME_NAME = "ACQ_180_YN"

def predict_conditional_expectation_tabpfn(
    model: CausalModel, 
    data_df: pd.DataFrame, 
    logger: logging.LoggerAdapter, 
    treatment_value: int = None
) -> Tuple[pd.DataFrame, float, CausalEstimate]:
    """
    TabPFN Estimator를 사용하여 E(Y|A, X) 조건부 기대값 예측을 수행합니다.
    
    TabpfnEstimator는 predict/interventional_outcomes 메서드를 직접 노출하지 않으므로,
    CausalModel.estimate_effect를 통해 CausalEstimate 객체를 생성한 후,
    내부의 TabpfnEstimator 객체를 사용하여 예측합니다.
    
    Args:
        model: DoWhy CausalModel 객체.
        data_df: 예측할 데이터프레임.
        logger: 로거 객체.
        treatment_value: 처치 값 (0: Control, 1: Treatment). None이면 data_df의 A 값 사용.
        
    Returns:
        tuple: (accuracy, result_df, estimate)
             - accuracy: 정확도 (이진 분류) 또는 -1.0 (계산 불가)
             - result_df: ACQ_180_YN 열에 예측값이 채워진 데이터프레임
             - estimate: 계산된 CausalEstimate 객체
    """
    
    logger.info(f"E(Y|A, X) 예측 시작: {len(data_df)}개")
    if treatment_value is not None:
        logger.info(f"처치 값: {treatment_value}")

    accuracy = -1.0
    result_df = data_df.copy()
    
    try:
        # 1. 식별 (Identification)
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        if identified is None:
            logger.error("Causal estimand 식별 실패. 예측을 건너뜁니다.")
            return accuracy, result_df, None
        
        # 2. TabPFN Estimator를 사용하여 효과 추정 (이 과정에서 Estimator가 학습됨)
        # 이 추정 과정에서 TabpfnEstimator 객체가 CausalEstimate 내부에 생성 및 저장됩니다.
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.tabpfn",
            method_params={"estimator": TabpfnEstimator, "n_estimators": 8},
        )

        if estimate is None or not hasattr(estimate, 'estimator'):
             logger.error("TabPFN CausalEstimate 객체 또는 Estimator 생성 실패.")
             return accuracy, result_df, None

        # 3. TabPFN Estimator 객체 가져오기
        estimator = estimate.estimator 

        # 4. 예측 수행
        if treatment_value is not None:
            # interventional_outcomes는 TabPFNEstimator.interventional_outcomes를 호출
            predictions = estimator.interventional_outcomes(data_df, treatment_value)
        else:
            # predict는 TabPFNEstimator.predict를 호출 (실제 data_df의 A 값 사용)
            predictions = estimator.predict(data_df)
        
        predictions_series = pd.Series(predictions, index=data_df.index)
        
        # 결과 데이터프레임에 예측값 채우기
        result_df[OUTCOME_NAME + "_PRED"] = predictions_series # 예측값을 새 컬럼에 저장
        
        # 실제 Y 값과 비교하여 정확도 계산 (이진 분류)
        if OUTCOME_NAME in data_df.columns:
            actual_y = data_df[OUTCOME_NAME]
            # TabPFN은 확률을 반환하므로 0.5 기준으로 이진 분류
            predicted_classes = (predictions_series > 0.5).astype(int)
            accuracy = (predicted_classes == actual_y).mean()
            logger.info(f"예측 완료: 정확도={accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            logger.warning(f"실제 Y 값({OUTCOME_NAME}) 컬럼 부재. 정확도 계산 불가.")
            logger.info(f"예측 완료: 평균={predictions_series.mean():.6f}")
            
        return accuracy, result_df, estimate
        
    except Exception as e:
        logger.error(f"TabPFN 예측 실패: {e}", exc_info=True)
        return -1.0, data_df, None


def run_prediction_analysis(
    top_dags_info: Dict[int, str], 
    final_df: pd.DataFrame, 
    results_dir: Path, 
    logger: logging.LoggerAdapter
) -> Dict[int, float]:
    """
    Top 5 Robust DAGs에 대해 TabPFN 기반의 예측(E[Y|A, X])을 수행하고 결과를 저장합니다.
    """
    logger.info("=" * 80)
    logger.info("Starting Conditional Expectation Prediction for Top Robust DAGs...")
    
    prediction_results = {}
    
    # DAG 인덱스 목록 추출
    dag_indices = [int(idx) for idx in top_dags_info.keys()]
    
    if not dag_indices:
        logger.warning("Top DAG 정보가 비어있습니다. 예측을 건너뜁니다.")
        return prediction_results

    for dag_idx in dag_indices:
        logger.info("-" * 50)
        logger.info(f"Processing Prediction for DAG index: {dag_idx}")
        
        dag_file = DAG_DIR / f"dag_{dag_idx}.txt"
        
        if not dag_file.exists():
            logger.warning(f"DAG file not found for index {dag_idx}. Skipping.")
            continue
            
        graph_txt = dag_file.read_text(encoding="utf-8")
        
        try:
            # 1. DAG 역할 추출
            roles = extract_roles_general(graph_txt, outcome=OUTCOME_NAME)
            dag_treatment = roles["treatment"]
            
            if dag_treatment not in final_df.columns:
                logger.error(f"Treatment '{dag_treatment}' not found in DataFrame. Skipping.")
                continue

            # 2. CausalModel 생성
            nx_graph = dot_to_nx(graph_txt)
            model = CausalModel(
                data=final_df,
                treatment=dag_treatment,
                outcome=OUTCOME_NAME,
                graph=nx_graph,
            )
            
            # 3. 예측 수행 (treatment_value=None: 실제 처치값(A)을 기반으로 예측)
            accuracy, result_df, estimate = predict_conditional_expectation_tabpfn(
                model=model,
                data_df=final_df,
                logger=logger,
                treatment_value=None # 실제 A 값 기반 예측
            )
            
            if accuracy != -1.0:
                prediction_results[dag_idx] = accuracy
            
            # 4. 결과 저장
            output_file = results_dir / f"prediction_dag_{dag_idx}.csv"
            # 원래 컬럼과 예측 컬럼만 포함
            columns_to_save = [col for col in final_df.columns] + [OUTCOME_NAME + "_PRED"]
            result_df[columns_to_save].to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"Prediction results saved to: {output_file.name}")
            
        except Exception as e:
            logger.error(f"Prediction analysis for DAG {dag_idx} failed: {e}", exc_info=True)
            
    logger.info("Prediction analysis complete.")
    return prediction_results

if __name__ == "__main__":
    # 이 부분은 테스트/디버깅 용도로만 사용됨
    print("This file contains prediction functions and should be imported.")