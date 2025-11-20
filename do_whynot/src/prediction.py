# do_whynot/src/prediction.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator
from do_whynot.src.dag_parser import extract_roles_general 
from do_whynot.config import DAG_DIR, EXCLUDE_COLS, PREFIX_COLS

from tqdm.auto import tqdm

def _build_dowhy_model(
    data_df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None,
    mediators: Optional[List[str]] = None,
) -> CausalModel:
    return CausalModel(
        data=data_df,
        treatment=[treatment],
        outcome=outcome,
        common_causes=(confounders or []),
        instruments=None,
        effect_modifiers=None,
        mediators=(mediators or []),
    )


def predict_for_one_dag(
    data_df: pd.DataFrame,
    roles: Dict[str, Any],
    outcome_name: str,
    logger: logging.Logger,
    treatment_value: Optional[Any] = None, 
    control_value: Optional[Any] = None,
) -> pd.Series:
    """
    TabpfnEstimator 내부의 `_build_model()` + `predict_fn()`을 사용해
    해당 DAG 구조에서의 Y 예측 확률(또는 예측값) Series를 반환.
    """
    treatment = roles.get("treatment")
    confounders = roles.get("confounders", [])
    mediators = roles.get("mediators", [])

    if treatment is None:
        raise ValueError("roles['treatment'] is required.")

    model = _build_dowhy_model(
        data_df=data_df,
        treatment=treatment,
        outcome=outcome_name,
        confounders=confounders,
        mediators=mediators,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    if identified is None:
        raise RuntimeError("Failed to identify estimand for the given DAG roles.")

    estimator = TabpfnEstimator(
        identified_estimand=identified,
        data=data_df,
        treatment=treatment,
        outcome=outcome_name,
        test_significance=False,
        method_params={"n_estimators": 8, "model_type": "auto"}, 
        control_value=control_value,    
        treatment_value=treatment_value,
    )

    # 필요하면 안전장치로 confounder/mediator 이름 주입
    if not hasattr(estimator, "_observed_common_causes_names"):
        estimator._observed_common_causes_names = confounders or []
    if not hasattr(estimator, "_observed_mediator_names"):
        estimator._observed_mediator_names = mediators or []
    if not hasattr(estimator, "_effect_modifier_names"):
        estimator._effect_modifier_names = []

    features, wrapper = estimator._build_model(data_df)
    preds = estimator.predict_fn(data_df, wrapper, features)

    return pd.Series(np.asarray(preds), index=data_df.index)



def run_prediction_pipeline(
    final_merged_df: pd.DataFrame,
    top_5_dags_info: List[Dict[str, Any]],
    outcome_name: str,
    data_output_dir: Path,
    logger: logging.Logger,
    batch_id: int,
) -> pd.DataFrame:

    # 1) 데이터 전처리
    df_copy = final_merged_df.copy()
    object_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    
    excluded_cols = EXCLUDE_COLS.copy()
    for prefix in PREFIX_COLS:
        prefix_cols = [c for c in df_copy.columns if c.startswith(prefix)]
        excluded_cols.extend(prefix_cols)

    excluded_cols = list(set(excluded_cols))
    cols_to_convert = [c for c in object_cols if c not in excluded_cols]

    for col in cols_to_convert:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('int')
        except Exception as e:
            logger.warning(f"[predict] Failed to convert object column '{col}' to int: {e}")
    
    df_copy['JHNT_MBN'] = df_copy['JHNT_MBN'].astype(str)
    pred_result_df = df_copy[['JHNT_MBN']].copy()

    # 2) 각 DAG별 roles 추출 및 예측 수행
    for info in tqdm(top_5_dags_info, desc=f"Batch {batch_id+1} Prediction on Top DAGs", leave=False):
        roles = {}

        dag_idx = int(info.get("dag_idx", -1))
        dag_treatment = info.get("treatment_column")
        colname = f"{outcome_name}_PRED_dag{dag_idx if dag_idx != -1 else 'X'}"

        baseline_val = info.get("multi_class_baseline")
        treatment_val = info.get("multi_class_treatment_value")

        if pd.isna(baseline_val):
            baseline_val = None
        if pd.isna(treatment_val):
            treatment_val = None
        
        if baseline_val is not None or treatment_val is not None:
             logger.info(f"[predict] DAG {dag_idx}: Applying Multi-Class values. Control={baseline_val}, Treatment={treatment_val}")
        else:
             logger.info(f"[predict] DAG {dag_idx}: No Multi-Class values found. Using default CausalEstimator values (Control=0, Treatment=1) for ATE calculation.")

        if not isinstance(dag_treatment, str):
            logger.error(f"[predict] DAG {dag_idx} failed: 'treatment_column' is missing or invalid in top_5_dags_info.")
            continue

        try:
            dag_path = DAG_DIR / f"dag_{dag_idx}.txt"
            if dag_path.exists():
                parsed = extract_roles_general(dag_path.read_text(encoding="utf-8"), outcome=outcome_name)
            else:
                logger.error(f"[predict] DAG {dag_idx} failed: DAG file not found at {dag_path}.")
                continue
        except Exception as e:
            logger.error(f"[predict] DAG {dag_idx} roles parse failed: {e}")
            continue

        roles = {
            "treatment": dag_treatment, # top_5_dags_info에서 가져온 treatment 사용
            "mediators": parsed.get("mediators", []),
            "confounders": parsed.get("confounders", []),
        }

        if roles["treatment"] not in df_copy.columns:
            logger.error(f"[predict] DAG {dag_idx} failed: treatment '{roles['treatment']}' not in DataFrame columns.")
            continue

        try:
            preds = predict_for_one_dag(df_copy, roles, outcome_name, logger, control_value=baseline_val, treatment_value=treatment_val)
            pred_result_df[colname] = preds
            logger.info(f"[predict] DAG {dag_idx}: Prediction column '{colname}' added to batch result.")
        except Exception as e:
            logger.error(f"[predict] DAG {dag_idx} failed: {e}", exc_info=True)
    preds_dir = data_output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = preds_dir / f"batch_preds_{batch_id+1:02d}.csv" 
    pred_result_df.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"[predict] Batch prediction result saved to: {save_path.name}")

    return pred_result_df