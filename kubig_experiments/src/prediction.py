# kubig_experiments/src/prediction.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator
from kubig_experiments.src.dag_parser import extract_roles_general 
from kubig_experiments.config import DAG_DIR, DAG_INDICES, IS_TEST_MODE, DAG_INDICES_TEST

# ---------------------------------------------------------------------
# 0) 유틸
# ---------------------------------------------------------------------

def _numeric_like_series(s: pd.Series) -> bool:
    """Series가 전부 결측 또는 정수형 문자열로만 구성되어 있으면 True."""
    if s.dtype.kind != "O":  
        return False
    non_na = s.dropna().astype(str)
    if non_na.empty:
        return True
    # 정수 문자열(- 부호 포함) 판별
    return non_na.str.fullmatch(r"-?\d+").all()


def restore_numeric_str_to_int(
    df: pd.DataFrame,
    logger: logging.Logger,
    excluded_cols: Optional[List[str]] = None,
    excluded_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    label-encoding 이후 object로 남은 열들 중 '숫자 문자열'이면 int로 복원.
    다만 excluded 컬럼/접두어는 제외.
    """
    excluded_cols = list(excluded_cols or [])
    excluded_prefixes = list(excluded_prefixes or [])

    keep_mask = np.full(df.shape[1], True, dtype=bool)
    cols = list(df.columns)

    # (1) 접두어 제외
    for i, c in enumerate(cols):
        if any(c.startswith(p) for p in excluded_prefixes):
            keep_mask[i] = False
    # (2) 명시 제외
    for i, c in enumerate(cols):
        if c in excluded_cols:
            keep_mask[i] = False

    converted = []
    for i, c in enumerate(cols):
        if not keep_mask[i]:
            continue
        s = df[c]
        if _numeric_like_series(s):
            try:
                df[c] = s.astype("Int64")  # 결측 허용 정수
                converted.append(c)
            except Exception:
                pass

    if converted:
        logger.info(f"[restore] converted to int: {len(converted)} cols -> {converted[:8]}{'...' if len(converted)>8 else ''}")
    else:
        logger.info("[restore] no convertible string-int columns found.")
    return df


def collect_pred_files(data_output_dir: Path, is_test_mode: bool, batch_size: int) -> List[Path]:
    """
    validation_pipeline 규칙:
      - TEST: preds_test.csv 1개
      - PROD: preds_1.csv, preds_2.csv, ... 배치 수만큼
    """
    if is_test_mode:
        f = data_output_dir / "preds_test.csv"
        return [f] if f.exists() else []
    # 배치 파일은 개수 예측이 필요할 수 있으므로 glob + 정렬
    files = sorted(data_output_dir.glob("preds_*.csv"))
    return files


def load_and_merge_llm_preds(
    base_df: pd.DataFrame,
    pred_files: List[Path],
    logger: logging.Logger,
    key_col: str = "JHNT_MBN",
    pred_col: str = "SELF_INTRO_CONT_LABEL",
) -> pd.DataFrame:
    """
    여러 배치의 LLM 예측 csv를 하나로 합쳐 base_df에 merge.
    dtype 보존을 위해 key_col은 문자열로 통일.
    """
    if not pred_files:
        logger.warning("[merge] no preds files found. returning base_df unchanged.")
        return base_df

    merged_pred = []
    for p in pred_files:
        try:
            dfp = pd.read_csv(p, encoding="utf-8", dtype={key_col: str})
            if pred_col not in dfp.columns:
                logger.warning(f"[merge] {p.name} has no column '{pred_col}'. Skipping.")
                continue
            dfp[key_col] = dfp[key_col].astype(str)
            merged_pred.append(dfp[[key_col, pred_col]])
        except Exception as e:
            logger.error(f"[merge] failed to read {p.name}: {e}")

    if not merged_pred:
        logger.warning("[merge] no usable preds file. returning base_df unchanged.")
        return base_df

    pred_all = pd.concat(merged_pred, ignore_index=True).drop_duplicates(subset=[key_col])
    out = base_df.copy()
    out[key_col] = out[key_col].astype(str)
    pred_all[key_col] = pred_all[key_col].astype(str)
    out = out.merge(pred_all, on=key_col, how="left")
    logger.info(f"[merge] merged LLM preds -> shape {out.shape}")
    return out

def _normalize_top_dags(top_dags_info: Any, final_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    analyze_results에서 나온 top_dags_info를 DAG 인덱스와 Treatment 변수로 정규화합니다.
    """
    import os
    # DAG 파일에서 treatment 컬럼을 찾는 유틸리티
    def _fill_from_dag(dag_idx: int) -> Optional[str]:
        dag_file = DAG_DIR / f"dag_{dag_idx}.txt"
        if not dag_file.exists():
            return None
        try:
            graph_txt = dag_file.read_text(encoding="utf-8")
            # DAG 파일의 역할을 추출합니다.
            roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
            treatment = roles.get("treatment")
            # DataFrame에 해당 컬럼이 있는지 최종 확인
            return treatment if treatment in final_df.columns else None
        except Exception:
            return None

    # test mode일 경우 DAG_INDICES_TEST를 사용하여 검증
    valid_dag_indices = DAG_INDICES_TEST if IS_TEST_MODE and os.environ.get("FORCE_TEST_DAGS") else DAG_INDICES
    
    normalized = []

    # case A: [dag_idx, treatment] 형태의 리스트
    if isinstance(top_dags_info, list) and all(
        isinstance(item, dict) and "dag_idx" in item and "treatment" in item
        for item in top_dags_info
    ):
        # 이미 정규화된 형태이므로, 컬럼만 검증합니다.
        for item in top_dags_info:
            if item["treatment"] in final_df.columns:
                normalized.append(item)
        if normalized:
            return normalized

    # case B: {dag_idx: treatment_name} 또는 {dag_idx: ATE_value} 형태의 딕셔너리
    if isinstance(top_dags_info, dict):
        for k, v in top_dags_info.items():
            try:
                dag_idx = int(k)
            except Exception:
                continue

            # 딕셔너리 값이 treatment name인 경우 또는 ATE 값인 경우
            tr = None
            if isinstance(v, str) and v in final_df.columns:
                tr = v
            else: # ATE 값 등이 들어있는 경우, DAG 파일에서 처리 변수를 추출
                tr = _fill_from_dag(dag_idx)
            
            if tr:
                normalized.append({"dag_idx": dag_idx, "treatment": tr})
        
        # 정규화된 결과가 없으면, Top-1 DAG (있는 경우)로 대체
        if not normalized and valid_dag_indices:
             tr = _fill_from_dag(valid_dag_indices[0])
             if tr:
                return [ {"dag_idx": valid_dag_indices[0], "treatment": tr}]
        
        if normalized:
            return normalized

    # 최후의 보루: DAG 1번의 Treatment로 대체
    tr1 = _fill_from_dag(1)
    if tr1:
        return [{"dag_idx": 1, "treatment": tr1}]
    
    return [] # 어떤 것도 찾지 못함

# ---------------------------------------------------------------------
# 1) TabPFN 예측
# ---------------------------------------------------------------------

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

    # DoWhy 스타일로 CausalModel 구성
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
        method_params={"n_estimators": 8, "model_type": "auto"},  # 필요시 조정
    )

    # 필요하면 안전장치로 confounder/mediator 이름 주입
    if not hasattr(estimator, "_observed_common_causes_names"):
        estimator._observed_common_causes_names = confounders or []
    if not hasattr(estimator, "_observed_mediator_names"):
        estimator._observed_mediator_names = mediators or []
    if not hasattr(estimator, "_effect_modifier_names"):
        estimator._effect_modifier_names = []

    # 핵심: estimator 내부 모델을 구성하고, predict_fn으로 바로 예측
    features, wrapper = estimator._build_model(data_df)
    preds = estimator.predict_fn(data_df, wrapper, features)

    # preds는 ndarray 형태일 수 있음 → pandas Series로 변환
    return pd.Series(np.asarray(preds), index=data_df.index)



# ---------------------------------------------------------------------
# 2) 최상위 API
# ---------------------------------------------------------------------

def run_prediction_pipeline(
    final_df: pd.DataFrame,
    top_5_dags_info: List[Dict[str, Any]],
    outcome_name: str,
    data_output_dir: Path,
    logger: logging.Logger,
    is_test_mode: bool = False,
    batch_size: int = 0,
) -> Dict[str, Any]:
    """
    반환값:
      {
        "merged_input_shape": (..),
        "pred_files": [...],
        "per_dag_outputs": [
            {"dag_idx": int, "csv": Path, "col": "Y_PRED_dagX"}
        ]
      }
    """
    # 1) preds 파일 수집 & 병합
    pred_files = collect_pred_files(data_output_dir, is_test_mode, batch_size)
    work_df = load_and_merge_llm_preds(final_df, pred_files, logger)

    # 2) 숫자 문자열 → int 복원
    base_excluded = ["SELF_INTRO_CONT", "JHNT_MBN", "JHNT_CTN", "JHCR_DE", "CLOS_YM"]
    prefix_excluded = ["CLOS_YM", "JHCR_DE"]
    work_df = restore_numeric_str_to_int(
        work_df, logger,
        excluded_cols=base_excluded,
        excluded_prefixes=prefix_excluded
    )

    # DAG 파일 경로
    dags_dir = Path("./kubig_experiments/dags")

    # 3) 각 DAG별 예측 실행
    outputs = []
    for info in top_5_dags_info:
        dag_idx = int(info.get("dag_idx", -1))
        colname = f"{outcome_name}_PRED_dag{dag_idx if dag_idx != -1 else 'X'}"

        # -------- roles 해석/보강 시작 --------
        raw_roles = info.get("roles", {}) if isinstance(info.get("roles"), dict) else {}
        roles = {
            "treatment": raw_roles.get("treatment"),
            "mediators": raw_roles.get("mediators", []),
            "confounders": raw_roles.get("confounders", []),
        }

        # (1) info에 treatment 키가 있으면 먼저 사용
        if not roles.get("treatment") and isinstance(info.get("treatment"), str):
            roles["treatment"] = info["treatment"]

        # (2) 그래도 없으면 DAG 텍스트/파일에서 추출
        if not roles.get("treatment"):
            dot_text = info.get("dot_text")
            try:
                if isinstance(dot_text, str) and dot_text.strip():
                    parsed = extract_roles_general(dot_text, outcome=outcome_name)
                else:
                    dag_path = dags_dir / f"dag_{dag_idx}.txt"
                    if dag_path.exists():
                        parsed = extract_roles_general(dag_path.read_text(encoding="utf-8"), outcome=outcome_name)
                    else:
                        parsed = {}
            except Exception as e:
                logger.error(f"[predict] DAG {dag_idx} roles parse failed: {e}")
                parsed = {}

            if isinstance(parsed, dict):
                roles["treatment"] = roles.get("treatment") or parsed.get("treatment")
                roles["mediators"] = roles.get("mediators", []) or parsed.get("mediators", [])
                roles["confounders"] = roles.get("confounders", []) or parsed.get("confounders", [])

        # (3) 최종 안전장치: treatment 존재 + 실제 컬럼 검증
        if not roles.get("treatment"):
            logger.error(f"[predict] DAG {dag_idx} failed: roles['treatment'] is required.")
            continue
        if roles["treatment"] not in work_df.columns:
            logger.error(f"[predict] DAG {dag_idx} failed: treatment '{roles['treatment']}' not in DataFrame columns.")
            continue
        # -------- roles 해석/보강 끝 --------

        try:
            preds = predict_for_one_dag(work_df, roles, outcome_name, logger)
            out_df = work_df.copy()
            out_df[colname] = preds

            save_path = data_output_dir / f"prediction_dag_{dag_idx if dag_idx != -1 else 'X'}.csv"
            out_df[[*work_df.columns, colname]].to_csv(save_path, index=False, encoding="utf-8")

            logger.info(f"[predict] DAG {dag_idx}: saved -> {save_path.name}")
            outputs.append({"dag_idx": dag_idx, "csv": save_path, "col": colname})
        except Exception as e:
            logger.error(f"[predict] DAG {dag_idx} failed: {e}", exc_info=True)

    return {
        "merged_input_shape": tuple(work_df.shape),
        "pred_files": pred_files,
        "per_dag_outputs": outputs,
    }
