import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import warnings
import uuid
from datetime import datetime
import argparse
import pytz

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

from kubig_experiments.src.preprocessor import build_pipeline_wide, postprocess
from kubig_experiments.src.dag_parser import extract_roles_general, dot_to_nx
from kubig_experiments.src.inference_top1 import llm_inference
from kubig_experiments.src.interpretator import load_and_consolidate_data, analyze_results
from kubig_experiments.src.eda import perform_eda

RESULTS_DIR = None
DATA_OUTPUT_DIR = project_root / "kubig_experiments" / "data" / "output"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class KSTFormatter(logging.Formatter):
    """UTC 타임스탬프를 KST (Asia/Seoul)로 변환하는 포맷터"""
    def converter(self, timestamp):
        # timestamp는 UTC 시간을 기준으로 합니다.
        KST = pytz.timezone('Asia/Seoul')
        dt = datetime.fromtimestamp(timestamp, pytz.utc)
        return dt.astimezone(KST).timetuple()

def setup_logger():
    """테스트 로깅 설정을 초기화하고 LoggerAdapter 객체를 반환합니다. """
    # 1) warnings 전부 무시 + 브릿지 차단
    warnings.filterwarnings("ignore")
    logging.captureWarnings(False)

    # 2) 외부 로거 경고 숨기기
    for name in [
        "py.warnings",
        "dowhy",
        "dowhy.causal_model",
        "dowhy.causal_identifier",
        "dowhy.causal_estimator",
    ]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)

    # 3) 고유 run_id (timestamp + uuid8)
    KST = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(KST)

    run_ts = now_kst.strftime("%Y%m%d-%H%M%S")
    run_uid = uuid.uuid4().hex[:8]
    run_id = f"{run_ts}_{run_uid}"


    # 4) 로그 디렉토리: 현재 파일 위치 기준으로 새 폴더 생성 후 로그 파일 저장
    global RESULTS_DIR
    log_base_dir = Path(__file__).resolve().parent / "logs"
    
    RESULTS_DIR = log_base_dir / run_id
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_path = RESULTS_DIR / f"run_log_{run_id}.log"

    # 5) 로거 + 핸들러
    base_logger = logging.getLogger("kubig.validation.tabpfn")
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    if not base_logger.handlers:
        fmt = KSTFormatter(
            "%(asctime)s | %(levelname)s | run=%(run_id)s | %(message)s"
        )

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        base_logger.addHandler(fh)
        base_logger.addHandler(ch)

    # 6) run_id를 필드로 주입하는 어댑터
    logger = logging.LoggerAdapter(base_logger, extra={"run_id": run_id})
    logger.info("Logging initialized. File: %s", str(log_path))
    return logger

def get_treatment_type(df: pd.DataFrame, treatment_col: str) -> str:
    """treatment 컬럼의 타입을 binary, multi-class, continuous 중 하나로 결정합니다."""
    if treatment_col not in df.columns:
        return "unknown"

    s = df[treatment_col].dropna()
    unique_count = s.nunique()

    if unique_count <= 1:
        return "unknown"

    # 1. object 타입이면서 unique count 확인 (사용자 요청 반영)
    if s.dtype == object:
        if unique_count == 2:
            return "binary"
        elif unique_count >= 3:
            return "multi-class"
            
    # 2. 숫자형 타입 확인
    if unique_count == 2:
        return "binary"
    elif unique_count < 20 and pd.api.types.is_numeric_dtype(s): # 카디널리티가 낮은 숫자형은 multi-class로 간주
        return "multi-class"
    
    # 3. 그 외 (주로 카디널리티가 높은 숫자형 또는 기타)
    return "continuous"


def estimate_tabpfn_ate_multi(model, identified, data, treatment_col, logger):
    """
    TabPFN 기반 ATE 추정 (binary + multi-category 공통 처리)

    - treatment_col의 unique level 확인
    - baseline:
        - 0이 있으면 0
        - 아니면 numeric이면 min(levels)
        - 혼합/문자면 문자열 기준 가장 앞
    - baseline vs 각 level에 대해 TabpfnEstimator로 ATE 추정
    - 유효한 값들 중 최댓값 선택

    return:
        best_ate (float) or None,
        best_level or None,
        ate_dict (dict[level] = ate) or None,
        best_estimate (CausalEstimate) or None  # refutation용
    """

    # treatment 컬럼 존재 여부 확인
    if treatment_col not in data.columns:
        logger.warning("[TabPFN] treatment column '%s' not in data.", treatment_col)
        return None, None, None, None

    # 실제 treatment level들
    levels = pd.Series(data[treatment_col].dropna().unique()).tolist()
    if len(levels) < 2:
        logger.info("[TabPFN] '%s': less than 2 levels, skip", treatment_col)
        return None, None, None, None

    # baseline 선택 로직
    if any(l == 0 for l in levels):
        baseline = 0
    else:
        try:
            baseline = min(levels)
        except TypeError:
            # 숫자/문자 섞인 경우 문자열 기준으로 가장 앞
            baseline = sorted(levels, key=lambda x: str(x))[0]

    logger.info("[TabPFN] treatment '%s': levels=%s, baseline=%r", treatment_col, levels, baseline)

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
                    "n_estimators": 8,
                    "treatment_value": lvl,
                    "control_value": baseline,
                },
            )

            val = getattr(est, "value", None)
            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                val = float(val)
                ate_dict[lvl] = val
                est_dict[lvl] = est
                logger.info("[TabPFN] ATE('%s'): level=%r vs baseline=%r -> %.6f", treatment_col, lvl, baseline, val,)
            
            else:
                logger.warning("[TabPFN] Non-numeric ATE for level=%r vs baseline=%r: %r", lvl, baseline, val)
        
        except Exception as e:
            logger.warning("[TabPFN] Failed to estimate ATE('%s') for level=%r vs baseline=%r: %s", treatment_col, lvl, baseline, e)

    if not ate_dict:
        logger.warning(
            "[TabPFN] No valid ATE values for treatment '%s'.", treatment_col
        )
        return None, None, None, None

    # 한 DAG 내에서 가장 큰 ATE 선택
    best_level, best_ate = max(ate_dict.items(), key=lambda kv: kv[1])

    logger.info("[TabPFN] Selected best ATE for '%s': level=%r, ate=%.6f", treatment_col, best_level, best_ate)

    best_est = est_dict[best_level]
    return float(best_ate), best_level, ate_dict, best_est


def validate_tabpfn_estimator(dag_idx: int, logger: logging.LoggerAdapter,
                              df: pd.DataFrame, treatment_type: str) -> dict: 
    """
    단일 DAG에 대해 Causal Estimation (TabPFN) 및 Refutation을 수행합니다.
    """
    results = {
        "dag_idx": dag_idx,
        "dag_file": f"dag_{dag_idx}.txt",
        "treatment": None,
        "lr_ate": None,
        "tabpfn_ate": None,
        "placebo_p_value": None,
        "random_cc_p_value": None,
        "is_successful": False,
        "skip_reason": None,
    }

    # DAG 파일 로딩
    dag_dir = Path("./kubig_experiments/dags/")
    dag_file = dag_dir / f"dag_{dag_idx}.txt"
    if not dag_file.exists():
        logger.warning("[%s] DAG file not found, skipping: %s", dag_file.name, str(dag_file))
        results["skip_reason"] = "DAG file not found"
        return results # 파일이 없으면 스킵

    graph_txt = dag_file.read_text(encoding="utf-8")

    roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
    dag_treatment = roles["treatment"]
    results["treatment"] = dag_treatment

    if dag_treatment not in df.columns:
        msg = f"[skip] treatment '{dag_treatment}' not found in dataframe."
        logger.info("[%s] %s", dag_file.name, msg)
        results["skip_reason"] = "treatment not in dataframe"
        return results 
    
    if treatment_type == "unknown":
        msg = f"[skip] Unknown treatment type for '{dag_treatment}'. Skipping."
        logger.info("[%s] %s", dag_file.name, msg)
        results["skip_reason"] = "Unknown treatment type"
        return results

    logger.info(
        "[%s] Roles | X=%s | M=%s | C=%s (Type: %s)",
        dag_file.name, dag_treatment, roles["mediators"], roles["confounders"], treatment_type
    )
    
    df_copy = df.copy()
    object_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    
    EXCLUDE_COLS = ["JHNT_MBN", "JHNT_CTN", "SELF_INTRO_CONT"] 
    cols_to_convert = [c for c in object_cols if c not in EXCLUDE_COLS]

    for col in cols_to_convert:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Int64')
            logger.debug(f"[{dag_file.name}] Converted object column '{col}' to Int64.")
        except Exception as e:
            logger.warning(f"[{dag_file.name}] Failed to convert object column '{col}' to Int64: {e}")
            
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
        logger.info("[%s] %s", dag_file.name, msg)
        results["skip_reason"] = "No valid identified estimand"
        return results
    
    # Linear Regression (Baseline) 추정
    try:
        method_lr = "backdoor.linear_regression"
        est_lr = model.estimate_effect(identified, method_name=method_lr, test_significance=True)
        
        if est_lr.value is None or not (isinstance(est_lr.value, (float, np.floating)) and np.isfinite(est_lr.value)):
            msg = f"[skip] Baseline (LR) estimation returned invalid value: {est_lr.value}"
            logger.warning("[%s] %s", dag_file.name, msg)
            est_lr_value = "INVALID"
        else:
            est_lr_value = float(est_lr.value)

        results["lr_ate"] = est_lr_value
        logger.info("[%s] [%s] Baseline(Linear Regression) ATE: %s", dag_file.name, dag_treatment, est_lr_value)
        
    except Exception as e:
        logger.error(f"[%s] Baseline (LR) estimation failed with exception: %s", dag_file.name, e)
        results["lr_ate"] = "ERROR"
        
    est_tabpfn = None # TabPFN 추정 결과를 저장할 변수 초기화

    # TabPFN 추정 (treatment_type에 따라 분기)
    if treatment_type in ["binary", "continuous"]:
        # Binary/Continuous (기존 로직: 단일 ATE 추정)
        try:
            est_tabpfn = model.estimate_effect(
                identified,
                method_name="backdoor.tabpfn",
                method_params={"estimator": TabpfnEstimator, "n_estimators": 8},
            )
        except Exception as e:
            msg = f"[skip] TabPFN estimation failed with exception: {e}"
            logger.error("[%s] %s", dag_file.name, msg)
            results["skip_reason"] = "TabPFN estimation failed"
            return results

        # 추정 결과 검증
        if est_tabpfn is None or est_tabpfn.value is None or not (
            isinstance(est_tabpfn.value, (float, np.floating)) and np.isfinite(est_tabpfn.value)
        ):
            msg = f"[skip] TabPFN estimation returned invalid value: {getattr(est_tabpfn, 'value', None)}"
            logger.info("[%s] %s", dag_file.name, msg)
            results["skip_reason"] = "TabPFN invalid value"
            return results
        
        results["tabpfn_ate"] = float(est_tabpfn.value)
        logger.info("[%s] [%s] TabPFN ATE: %s", dag_file.name, dag_treatment, est_tabpfn.value)

    elif treatment_type == "multi-class":
        # Multi-Class (요구사항 3: estimate_tabpfn_ate_multi 사용)
        logger.info(f"[{dag_file.name}] Running Multi-Class ATE Estimation for '{dag_treatment}'.")
        
        best_ate, best_level, ate_dict, est_tabpfn_best = estimate_tabpfn_ate_multi(
            model=model, 
            identified=identified, 
            data=df_copy, # 변환된 df_copy 사용
            treatment_col=dag_treatment, 
            logger=logger
        )
        
        if best_ate is None:
            msg = "[skip] Multi-Class TabPFN estimation failed or returned no valid ATE."
            logger.info("[%s] %s", dag_file.name, msg)
            results["skip_reason"] = "TabPFN multi-class estimation failed"
            return results
            
        results["tabpfn_ate"] = best_ate
        # Refutation을 위해 est_tabpfn을 최적의 추정 객체로 업데이트
        est_tabpfn = est_tabpfn_best 
        logger.info("[%s] [%s] Multi-Class TabPFN ATE (Best: %r): %s", dag_file.name, dag_treatment, best_level, best_ate)
    
    else:
        # 'unknown' 타입은 이미 위에서 걸러짐
        return results

    # 반박 (Refutation)
    refuters = {
        "placebo_treatment_refuter": "placebo_p_value",
        "random_common_cause": "random_cc_p_value"
    }
    
    for ref_name, result_key in refuters.items():
        try:
            refutation = model.refute_estimate(identified, est_tabpfn, method_name=ref_name)
            logger.info("[%s] Refutation (%s): %s", dag_file.name, ref_name, refutation)

            if refutation is None:
                logger.error("[%s] Refutation failed for %s", dag_file.name, ref_name)
            else:
                # --- 문자열 파싱을 통한 p-value 추출 로직 ---
                p_value_float = None
                refutation_str = str(refutation)
                
                if 'p value:' in refutation_str:
                    p_str = refutation_str.split('p value:')[-1].strip()
                    if p_str:
                        p_str = p_str.split()[0]
                    
                    try:
                        p_value_float = float(p_str)
                    except (ValueError, TypeError):
                        pass
                
                if p_value_float is not None and np.isfinite(p_value_float):
                    results[result_key] = p_value_float
                
        except Exception as e:
            logger.error("[%s] Refutation (%s) failed with exception: %s", dag_file.name, ref_name, e)
            results[result_key] = "ERROR"
            
    results["is_successful"] = True
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Causal Validation Pipeline.")
    parser.add_argument('-test', action='store_true', help='Enable test mode, sampling 1000 rows.')
    args = parser.parse_args()
    
    IS_TEST_MODE = args.test
    TEST_SAMPLE_SIZE = 100

    BATCH_SIZE = 50 
    
    # 로거 초기화
    main_logger = setup_logger()
    main_logger.info("Starting Data Preprocessing Pipeline from imported functions.")

    # --- 1. 전처리 실행 (데이터 전체에 대해 단 1회 실행) ---
    try:
        # 1) 와이드 포맷 조립 (JSON 파싱 및 CSV 병합)
        intermediate_df = build_pipeline_wide(main_logger)
        main_logger.info(f"Wide pipeline complete. Intermediate shape: {intermediate_df.shape}")
        
        # 2) 후처리 (이진 매핑, 날짜 차이, 결측 컬럼 제거)
        final_df = postprocess(intermediate_df, main_logger, DATA_OUTPUT_DIR) 
        main_logger.info(f"Preprocessing complete. Final DataFrame shape: {final_df.shape}")
        
    except Exception as e:
        main_logger.error(f"[Fatal] Preprocessing failed during execution: {e}")
        sys.exit(1)

    preprocessed_path = DATA_OUTPUT_DIR / "preprocessed_df.csv"
    final_df.to_csv(preprocessed_path, index=False, encoding="utf-8")
    main_logger.info(f"[OK] Preprocessed data saved to: {preprocessed_path.name}")

    perform_eda(final_df, DATA_OUTPUT_DIR, main_logger)
    
    # --- 2. 배치 분할 및 반복 실행 ---
    if IS_TEST_MODE:
        main_logger.info("Test mode enabled: sampling %d rows for quick validation.", TEST_SAMPLE_SIZE)
        final_df = final_df.sample(n=TEST_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        BATCH_SIZE = TEST_SAMPLE_SIZE

    if not IS_TEST_MODE:
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        main_logger.info("Data shuffled successfully before batching. Final DataFrame shape: %s", final_df.shape)

    total_rows = len(final_df)
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    if IS_TEST_MODE:
        dag_indices = [1, 10]  # 테스트 모드에서는 소수의 DAG만 사용
    else:
        dag_indices = range(1, 43)

    main_logger.info("-" * 50)
    main_logger.info("Starting batch validation runs.")
    main_logger.info(f"Total rows: {total_rows}. Batch size: {BATCH_SIZE}. Total batches: {num_batches}.")

    all_results_df = pd.DataFrame()

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        
        batch_df = final_df.iloc[start_idx:end_idx].copy()
        
        main_logger.info("=" * 70)
        main_logger.info(f"BATCH {i+1}/{num_batches}: Processing rows {start_idx} to {end_idx-1} (Size: {len(batch_df)})")
        main_logger.info("=" * 70)
        
        preds_dir = DATA_OUTPUT_DIR

        if IS_TEST_MODE:
            preds_file = preds_dir / f"preds_test.csv"
        else:
            preds_file = preds_dir / f"preds_{i+1}.csv"
        
        if not preds_file.exists():
            main_logger.info(f"Starting LLM Inference for BATCH {i+1}...")
            llm_preds_df = llm_inference(batch_df, main_logger, i, IS_TEST_MODE, DATA_OUTPUT_DIR) 
            main_logger.info(f"LLM Inference for BATCH {i+1} complete. Predictions saved.")
        else:
            main_logger.info(f"Loading existing LLM predictions from {preds_file.name} for BATCH {i+1}...")
            # JHNT_MBN을 문자열로 로드하여 정밀도 보존
            llm_preds_df = pd.read_csv(preds_file, encoding="utf-8", dtype={'JHNT_MBN': str})
        
        # JHNT_MBN을 str로 통일하여 병합
        batch_df['JHNT_MBN'] = batch_df['JHNT_MBN'].astype(str)
        llm_preds_df['JHNT_MBN'] = llm_preds_df['JHNT_MBN'].astype(str)

        batch_df = pd.merge(
            batch_df, 
            llm_preds_df[['JHNT_MBN', 'SELF_INTRO_CONT_LABEL']], 
            on='JHNT_MBN', 
            how='left'
        )
        main_logger.info(f"Merged LLM predictions into batch dataframe. New shape: {batch_df.shape}")

        batch_results = []

        for dag_idx in dag_indices:
            
            dag_dir = Path("./kubig_experiments/dags/")
            dag_file = dag_dir / f"dag_{dag_idx}.txt"
            if not dag_file.exists():
                continue
            graph_txt = dag_file.read_text(encoding="utf-8")
            roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
            dag_treatment = roles["treatment"]
            
            treatment_type = get_treatment_type(batch_df, dag_treatment)

            main_logger.info("-" * 50)
            main_logger.info(f"[Batch {i+1}] Processing DAG index: {dag_idx} (Treatment: {dag_treatment}, Type: {treatment_type})")
            
            result_dict = validate_tabpfn_estimator(dag_idx, main_logger, batch_df, treatment_type) 
            
            if result_dict:
                result_dict["batch_id"] = i + 1 
                result_dict["batch_size"] = len(batch_df) 
                
                batch_results.append(result_dict)
        
        if batch_results:
            batch_results_df = pd.DataFrame(batch_results)
            all_results_df = pd.concat([all_results_df, batch_results_df], ignore_index=True) # 전체 결과에 누적
            
            # 배치별 파일 저장 (CSV)
            batch_result_file = RESULTS_DIR / f"batch_results_{i+1:02d}.csv"
            batch_results_df.to_csv(batch_result_file, index=False, encoding="utf-8")
            main_logger.info(f" BATCH {i+1} results saved to: {batch_result_file.name}")
    
    final_result_file = RESULTS_DIR / "all_validation_results.csv"
    all_results_df.to_csv(final_result_file, index=False, encoding="utf-8")
    main_logger.info("All validation results saved to: %s", final_result_file.name)

    main_logger.info("-" * 50)
    main_logger.info("Validation runs complete.")
    
    # --- 3. 최종 해석 로직 ---
    main_logger.info("Starting Causal Interpretation Analysis...")
    
    df_consolidated = load_and_consolidate_data(RESULTS_DIR, main_logger)
    top_dags_info = analyze_results(df_consolidated, main_logger)
    
    main_logger.info("Interpretation analysis complete.")

if __name__ == "__main__":
    main()