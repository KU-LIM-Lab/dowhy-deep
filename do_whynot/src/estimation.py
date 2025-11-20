import pandas as pd
import numpy as np
import logging

from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

from do_whynot.src.dag_parser import extract_roles_general, dot_to_nx
from do_whynot.config import DAG_DIR, EXCLUDE_COLS, PREFIX_COLS, MULTICLASS_THRESHOLD, MULTICLASS_PASS

from tqdm.auto import tqdm

def get_treatment_type(df: pd.DataFrame, treatment_col: str) -> str:
    """treatment 컬럼의 타입을 binary, multi-class, continuous 중 하나로 결정합니다."""
    if treatment_col not in df.columns:
        return "unknown"

    s = df[treatment_col].dropna()
    s_list = [v for v in s if v != -1 and v != '-1']
    unique_count = s_list.nunique()

    if unique_count <= 1:
        return "unknown"

    # 1. object 타입이면서 unique count 확인 
    if s.dtype == object:
        if unique_count == 2:
            return "binary"
        elif unique_count >= 3:
            return "multi-class"
            
    # 2. 숫자형 타입 확인
    if unique_count == 2:
        return "binary"
    elif unique_count < MULTICLASS_THRESHOLD: # 카디널리티가 낮은 숫자형은 multi-class로 간주
        return "multi-class"
    
    # 3. 그 외 (주로 카디널리티가 높은 숫자형 또는 기타)
    return "continuous"


def estimate_tabpfn_ate_multi(model, identified, data, treatment_col, logger, 
                              fixed_baseline=None, fixed_treatment_value=None):
    """
    TabPFN 기반 ATE 추정 (multi-category 전용)

    - fixed_baseline/fixed_treatment_value가 제공되면 해당 값으로 ATE 추정
    - 제공되지 않으면, 모든 레벨에 대해 ATE를 계산하고 '절댓값'이 가장 큰 조합을 선택.

    return:
        best_ate (float) or None,
        best_level (selected treatment value) or None,
        baseline_value (selected baseline value) or None,
        ate_dict (dict[level] = ate) or None,
        best_estimate (CausalEstimate) or None  # refutation용
    """

    # treatment 컬럼 존재 여부 확인
    if treatment_col not in data.columns:
        logger.warning("[TabPFN/Multi] treatment column '%s' not in data.", treatment_col)
        return None, None, None, None, None

    # 실제 treatment level들
    levels = pd.Series(data[treatment_col].dropna().unique()).tolist()
    levels = [lvl for lvl in levels if lvl != -1 and lvl != '-1']
    
    if len(levels) < 2:
        logger.info("[TabPFN/Multi] '%s': less than 2 levels, skip", treatment_col)
        return None, None, None, None, None

    # 고정된 값이 있는 경우 (2번째 배치 이후)
    if fixed_baseline is not None and fixed_treatment_value is not None:
        if fixed_baseline not in levels or fixed_treatment_value not in levels:
            logger.warning("[TabPFN/Multi] Fixed baseline or treatment value not in current batch levels. Skipping.")
            return None, None, None, None, None
            
        baseline = fixed_baseline
        lvl = fixed_treatment_value
        logger.info("[TabPFN/Multi] Fixed Estimation: treatment='%s', level=%r vs baseline=%r", treatment_col, lvl, baseline)
        
        try:
            est = model.estimate_effect(
                identified,
                method_name="backdoor.tabpfn",
                method_params={
                    "estimator": TabpfnEstimator,
                    "n_estimators": 8,
                },
                treatment_value=lvl,
                control_value=baseline,
            )

            val = getattr(est, "value", None)
            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                val = float(val)
                logger.info("[TabPFN/Multi] Fixed ATE('%s'): level=%r vs baseline=%r -> %.6f", treatment_col, lvl, baseline, val)
                # 고정된 값일 경우 ate_dict, est_dict는 단일 값만 가집니다.
                return val, lvl, baseline, {lvl: val}, est
            else:
                logger.warning("[TabPFN/Multi] Non-numeric ATE for fixed level=%r vs baseline=%r: %r", lvl, baseline, val)
                return None, None, None, None, None
        
        except Exception as e:
            logger.warning("[TabPFN/Multi] Failed to estimate fixed ATE('%s') for level=%r vs baseline=%r: %s", treatment_col, lvl, baseline, e)
            return None, None, None, None, None
            
    # 첫 번째 배치: baseline 선택 로직 및 전체 ATE 계산
    if any(l == 0 for l in levels):
        baseline = 0
    else:
        try:
            baseline = min(levels)
        except TypeError:
            # 숫자/문자 섞인 경우 문자열 기준으로 가장 앞
            baseline = sorted(levels, key=lambda x: str(x))[0]

    logger.info("[TabPFN/Multi] First Batch Estimation: treatment '%s': levels=%s, baseline=%r", treatment_col, levels, baseline)

    ate_dict = {}
    est_dict = {}

    # baseline vs. 나머지 (레벨 개수 - 1 만큼 ATE 계산)
    for lvl in levels:
        if lvl == baseline:
            continue
        try:
            est = model.estimate_effect(
                identified,
                method_name="backdoor.tabpfn",
                method_params={
                    "estimator": TabpfnEstimator,
                    "n_estimators": 8
                },
                treatment_value=lvl,
                control_value=baseline,
            )

            val = getattr(est, "value", None)
            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                val = float(val)
                ate_dict[lvl] = val
                est_dict[lvl] = est
                logger.info("[TabPFN/Multi] ATE('%s'): level=%r vs baseline=%r -> %.6f", treatment_col, lvl, baseline, val,)
            
            else:
                logger.warning("[TabPFN/Multi] Non-numeric ATE for level=%r vs baseline=%r: %r", lvl, baseline, val)
        
        except Exception as e:
            logger.warning("[TabPFN/Multi] Failed to estimate ATE('%s') for level=%r vs baseline=%r: %s", treatment_col, lvl, baseline, e)

    if not ate_dict:
        logger.warning(
            "[TabPFN/Multi] No valid ATE values for treatment '%s'.", treatment_col
        )
        return None, None, None, None, None

    # 한 DAG 내에서 '절댓값'이 가장 큰 ATE 선택 
    best_level, best_ate = max(ate_dict.items(), key=lambda kv: abs(kv[1]))

    selected_treatment_value = best_level
    selected_baseline = baseline

    # ate가 음수일 경우 baseline과 treatment swap
    if best_ate < 0:
        best_ate = abs(best_ate)
        
        selected_treatment_value = baseline
        selected_baseline = best_level

        logger.info(
            "[TabPFN/Multi] ATE was negative (%.6f). Flipped treatment/baseline: New Treatment=%r, New Baseline=%r, New ATE=%.6f", 
            ate_dict[best_level], selected_treatment_value, selected_baseline, best_ate
        )
    else:
        logger.info("[TabPFN/Multi] ATE is non-negative (%.6f). Retain original treatment/baseline: Treatment=%r, Baseline=%r",
                    best_ate, selected_treatment_value, selected_baseline)
        
    logger.info("[TabPFN/Multi] Selected best ATE (max absolute) for '%s': level=%r, ate=%.6f", treatment_col, best_level, best_ate)

    best_est = est_dict[best_level]

    return float(best_ate), selected_treatment_value, selected_baseline, ate_dict, best_est


def estimate_tabpfn_ate_binary_continuous(model, identified, dag_treatment, logger):
    """
    TabPFN 기반 ATE 추정 (binary + continuous 전용)
    """
    try:
        est_tabpfn = model.estimate_effect(
            identified,
            method_name="backdoor.tabpfn",
            method_params={"estimator": TabpfnEstimator, "n_estimators": 8},
        )
    except Exception as e:
        msg = f"[skip] TabPFN (B/C) estimation failed with exception: {e}"
        logger.error(f"[%s] %s", dag_treatment, msg)
        return None, None

    # 추정 결과 검증
    if est_tabpfn is None or est_tabpfn.value is None or not (
        isinstance(est_tabpfn.value, (float, np.floating)) and np.isfinite(est_tabpfn.value)
    ):
        msg = f"[skip] TabPFN (B/C) estimation returned invalid value: {getattr(est_tabpfn, 'value', None)}"
        logger.info(f"[%s] %s", dag_treatment, msg)
        return None, None
    
    return float(est_tabpfn.value), est_tabpfn


def run_tabpfn_estimation(model: CausalModel, identified, df_copy: pd.DataFrame, 
                          dag_file_name: str, dag_treatment: str, treatment_type: str, logger: logging.LoggerAdapter,
                          fixed_baseline=None, fixed_treatment_value=None) -> dict:
    """
    TabPFN ATE 추정 및 Refutation을 수행하고 결과를 반환합니다.
    (validate_tabpfn_estimator에서 분리된 ATE Estimation 및 Refutation 전용 함수)
    """
    results = {
        "tabpfn_ate": None,
        "multi_class_baseline": None,
        "multi_class_treatment_value": None,
        "placebo_p_value": None,
        "random_cc_p_value": None,
        "is_successful": False,
        "skip_reason": None,
    }
    
    est_tabpfn = None
    best_ate = None

    if treatment_type == "binary":
        # Binary/Continuous
        best_ate, est_tabpfn = estimate_tabpfn_ate_binary_continuous(
            model=model, identified=identified, dag_treatment=dag_treatment, logger=logger
        )
        if best_ate is None:
            results["skip_reason"] = "TabPFN (B/C) invalid value or failed"
            return results
        
        results["tabpfn_ate"] = best_ate
        logger.info("[%s] [%s] TabPFN ATE: %.6f", dag_file_name, dag_treatment, best_ate)


    elif treatment_type == "multi-class":
        # Multi-Class
        logger.info(f"[{dag_file_name}] Running Multi-Class ATE Estimation for '{dag_treatment}'.")
        
        best_ate, best_level, baseline, ate_dict, est_tabpfn = estimate_tabpfn_ate_multi(
            model=model, 
            identified=identified, 
            data=df_copy, # 변환된 df_copy 사용
            treatment_col=dag_treatment, 
            logger=logger,
            fixed_baseline=fixed_baseline,
            fixed_treatment_value=fixed_treatment_value
        )
        
        if best_ate is None:
            results["skip_reason"] = "TabPFN multi-class estimation failed or returned no valid ATE."
            return results
            
        results["tabpfn_ate"] = best_ate
        results["multi_class_baseline"] = baseline
        results["multi_class_treatment_value"] = best_level
        
        logger.info("[%s] [%s] Multi-Class TabPFN ATE (Level: %r, Base: %r): %.6f", 
                    dag_file_name, dag_treatment, best_level, baseline, best_ate)
    
    else:
        results["skip_reason"] = "Unknown treatment type in estimation phase"
        return results

    # --- 반박 (Refutation) ---
    # refuters = {
    #     "placebo_treatment_refuter": "placebo_p_value",
    #     "random_common_cause": "random_cc_p_value"
    # }
    
    # for ref_name, result_key in tqdm(refuters.items(), desc=f"Refutation ({dag_file_name})", leave=False):
    #     try:
    #         # est_tabpfn이 유효한 경우에만 Refutation 수행
    #         if est_tabpfn is None:
    #              logger.error("[%s] Skipping Refutation: est_tabpfn is None.", dag_file_name)
    #              results[result_key] = "EST_NONE"
    #              continue
                 
    #         refutation = model.refute_estimate(identified, est_tabpfn, method_name=ref_name, 
    #                                            n_jobs=-1, num_simulations=50)
    #         logger.info("[%s] Refutation (%s): %s", dag_file_name, ref_name, refutation)

    #         if refutation is None:
    #             logger.error("[%s] Refutation failed for %s", dag_file_name, ref_name)
    #         else:
    #             # --- 문자열 파싱을 통한 p-value 추출 로직 ---
    #             p_value_float = None
    #             refutation_str = str(refutation)
                
    #             if 'p value:' in refutation_str:
    #                 p_str = refutation_str.split('p value:')[-1].strip()
    #                 if p_str:
    #                     p_str = p_str.split()[0]
                    
    #                 try:
    #                     p_value_float = float(p_str)
    #                 except (ValueError, TypeError):
    #                     pass
                
    #             if p_value_float is not None and np.isfinite(p_value_float):
    #                 results[result_key] = p_value_float
                
    #     except Exception as e:
    #         logger.error("[%s] Refutation (%s) failed with exception: %s", dag_file_name, ref_name, e)
    #         results[result_key] = "ERROR"
            
    results["is_successful"] = (best_ate is not None)
    return results


def validate_tabpfn_estimator(dag_idx: int, logger: logging.LoggerAdapter,
                              df: pd.DataFrame, treatment_type: str) -> dict: 
    """
    단일 DAG에 대해 Causal Estimation을 위한 전처리 및 CausalModel 초기화/식별을 수행합니다.
    (ATE 추정 및 Refutation은 run_tabpfn_estimation에서 처리됨)
    """
    results = {
        "dag_idx": dag_idx,
        "dag_file": f"dag_{dag_idx}.txt",
        "treatment": None,
        "lr_ate": None,
        "tabpfn_ate": None,
        "multi_class_baseline": None, # 멀티클래스 Baseline
        "multi_class_treatment_value": None, # 멀티클래스 최적 Treatment Value
        "placebo_p_value": None,
        "random_cc_p_value": None,
        "is_successful": False,
        "skip_reason": None,
        "model": None, # CausalModel 객체 저장
        "identified": None, # IdentifiedEstimand 객체 저장
        "df_processed": None, # 전처리된 DataFrame 저장
    }

    # DAG 파일 로딩
    dag_file = DAG_DIR / f"dag_{dag_idx}.txt"
    dag_file_name = dag_file.name
    if not dag_file.exists():
        logger.warning("[%s] DAG file not found, skipping: %s", dag_file_name, str(dag_file))
        results["skip_reason"] = "DAG file not found"
        return results # 파일이 없으면 스킵

    graph_txt = dag_file.read_text(encoding="utf-8")

    roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
    dag_treatment = roles["treatment"]
    results["treatment"] = dag_treatment

    if dag_treatment not in df.columns:
        msg = f"[skip] treatment '{dag_treatment}' not found in dataframe."
        logger.info("[%s] %s", dag_file_name, msg)
        results["skip_reason"] = "treatment not in dataframe"
        return results 
    
    if treatment_type == "unknown":
        msg = f"[skip] Unknown treatment type for '{dag_treatment}'. Skipping."
        logger.info("[%s] %s", dag_file_name, msg)
        results["skip_reason"] = "Unknown treatment type"
        return results
    
    if treatment_type == "continuous":
        msg = f"[skip] Continuous type for '{dag_treatment}'. Skipping."
        logger.info("[%s] %s", dag_file_name, msg)
        results["skip_reason"] = "Continuous treatment type"
        return results

    logger.info(
        "[%s] Roles | X=%s | M=%s | C=%s (Type: %s)",
        dag_file_name, dag_treatment, roles["mediators"], roles["confounders"], treatment_type
    )
    
    if treatment_type == "multi-class":
        label_count = df[dag_treatment].nunique()
        
        # config의 MULTICLASS_PASS가 True이고, label_count가 임계값을 초과할 경우
        if MULTICLASS_PASS and label_count > MULTICLASS_THRESHOLD:
            msg = (f"[skip] Multi-class treatment '{dag_treatment}' has too many labels ({label_count} > {MULTICLASS_THRESHOLD}). "
                   f"Skipping due to MULTICLASS_PASS=True.")
            logger.warning("[%s] %s", dag_file_name, msg)
            results["skip_reason"] = f"Label count too high ({label_count})"
            return results

    # 데이터 복사 및 Object 컬럼 숫자형 변환 (전처리)
    df_copy = df.copy()
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
            logger.debug(f"[{dag_file_name}] Converted object column '{col}' to Int64.")
        except Exception as e:
            logger.warning(f"[{dag_file_name}] Failed to convert object column '{col}' to Int64: {e}")
    
    # 그래프 생성 및 노드 이름 조정
    nx_graph = dot_to_nx(graph_txt)
    
    # DoWhy CausalModel 생성
    model = CausalModel(
        data=df_copy,
        treatment=dag_treatment,
        outcome="ACQ_180_YN",
        graph=nx_graph,
    )

    # 식별
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    if identified is None:
        msg = "[skip] No valid identified estimand (식별 실패)."
        logger.info("[%s] %s", dag_file_name, msg)
        results["skip_reason"] = "No valid identified estimand"
        return results
    
    # Linear Regression (Baseline) 추정
    try:
        method_lr = "backdoor.linear_regression"
        est_lr = model.estimate_effect(identified, method_name=method_lr, test_significance=True)
        
        if est_lr.value is None or not (isinstance(est_lr.value, (float, np.floating)) and np.isfinite(est_lr.value)):
            msg = f"[skip] Baseline (LR) estimation returned invalid value: {est_lr.value}"
            logger.warning("[%s] %s", dag_file_name, msg)
            est_lr_value = "INVALID"
        else:
            est_lr_value = float(est_lr.value)

        results["lr_ate"] = est_lr_value
        logger.info("[%s] [%s] Baseline(Linear Regression) ATE: %s", dag_file_name, dag_treatment, est_lr_value)
        
    except Exception as e:
        logger.error(f"[%s] Baseline (LR) estimation failed with exception: %s", dag_file_name, e)
        results["lr_ate"] = "ERROR"
        
    # 다음 단계(run_tabpfn_estimation)를 위해 필요한 객체 저장
    results["model"] = model
    results["identified"] = identified
    results["df_processed"] = df_copy
    results["is_successful"] = True 
    return results