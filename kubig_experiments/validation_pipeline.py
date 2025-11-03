import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import warnings
import uuid
from datetime import datetime
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

from kubig_experiments.src.preprocessor import build_pipeline_wide, postprocess
from kubig_experiments.src.dag_parser import extract_roles_general, dot_to_nx

RESULTS_DIR = None

def setup_logger():
    """테스트 로깅 설정을 초기화하고 LoggerAdapter 객체를 반환합니다. (원래의 test_logger fixture)"""
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
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
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
        fmt = logging.Formatter(
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


def validate_tabpfn_estimator(dag_idx: int, logger: logging.LoggerAdapter,
                              df: pd.DataFrame) -> dict:
    """
    단일 DAG에 대해 Causal Estimation (TabPFN) 및 Refutation을 수행합니다.
    - DAG에서 Treatment/Confounder/Mediator 자동 추출
    - 비교: TabPFN 추정기
    - Validation: Placebo treatment, Random common cause
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
    dag_dir = Path("./kubig_experiments/dags/output/")
    dag_file = dag_dir / f"dag_{dag_idx}.txt"
    if not dag_file.exists():
        logger.warning("[%s] DAG file not found, skipping: %s", dag_file.name, str(dag_file))
        results["skip_reason"] = "DAG file not found"
        return # 파일이 없으면 스킵

    graph_txt = dag_file.read_text(encoding="utf-8")

    # 역할 추출 (kubig_experiments.src.dag_parser의 extract_roles_general 함수 사용)
    roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
    dag_treatment = roles["treatment"]
    results["treatment"] = dag_treatment

    if dag_treatment not in df.columns:
        msg = f"[skip] treatment '{dag_treatment}' not found in dataframe."
        logger.info("[%s] %s", dag_file.name, msg)
        results["skip_reason"] = "treatment not in dataframe"
        return # pytest.skip 대신 return 사용

    logger.info(
        "[%s] Roles | X=%s | M=%s | C=%s",
        dag_file.name, dag_treatment, roles["mediators"], roles["confounders"]
    )

    # 그래프 생성 및 노드 이름 조정
    nx_graph = dot_to_nx(graph_txt)
    
    # DoWhy CausalModel 생성
    model = CausalModel(
        data=df,
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
        return 
    
    # Linear Regression (Baseline) 추정
    try:
        method_lr = "backdoor.linear_regression"
        est_lr = model.estimate_effect(identified, method_name=method_lr, test_significance=True)
        
        # Linear Regression 결과 검증
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

    # TabPFN 추정
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
        return

    # 추정 결과 검증
    if est_tabpfn is None or est_tabpfn.value is None or not (
        isinstance(est_tabpfn.value, (float, np.floating)) and np.isfinite(est_tabpfn.value)
    ):
        msg = f"[skip] TabPFN estimation returned invalid value: {getattr(est_tabpfn, 'value', None)}"
        logger.info("[%s] %s", dag_file.name, msg)
        results["skip_reason"] = "TabPFN invalid value"
        return
    
    results["tabpfn_ate"] = float(est_tabpfn.value)
    logger.info("[%s] [%s] TabPFN ATE: %s", dag_file.name, dag_treatment, est_tabpfn.value)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Causal Validation Pipeline.")
    parser.add_argument('-test', action='store_true', help='Enable test mode, sampling 1000 rows.')
    args = parser.parse_args()
    
    IS_TEST_MODE = args.test
    TEST_SAMPLE_SIZE = 100

    BATCH_SIZE = 10000 
    
    # 로거 초기화
    main_logger = setup_logger()
    main_logger.info("Starting Data Preprocessing Pipeline from imported functions.")

    # --- 1. 전처리 실행 (데이터 전체에 대해 단 1회 실행) ---
    try:
        # 1) 와이드 포맷 조립 (JSON 파싱 및 CSV 병합)
        intermediate_df = build_pipeline_wide(main_logger)
        main_logger.info(f"Wide pipeline complete. Intermediate shape: {intermediate_df.shape}")
        
        # 2) 후처리 (이진 매핑, 날짜 차이, 결측 컬럼 제거)
        final_df = postprocess(intermediate_df, main_logger) 
        main_logger.info(f"Preprocessing complete. Final DataFrame shape: {final_df.shape}")
        
    except Exception as e:
        main_logger.error(f"[Fatal] Preprocessing failed during execution: {e}")
        sys.exit(1)
    
    # --- 2. 배치 분할 및 반복 실행 ---
    if IS_TEST_MODE:
        main_logger.info("Test mode enabled: sampling %d rows for quick validation.", TEST_SAMPLE_SIZE)
        final_df = final_df.sample(n=TEST_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        BATCH_SIZE = TEST_SAMPLE_SIZE

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

        batch_results = []

        for dag_idx in dag_indices:
            main_logger.info("-" * 50)
            main_logger.info(f"[Batch {i+1}] Processing DAG index: {dag_idx}")
            result_dict = validate_tabpfn_estimator(dag_idx, main_logger, batch_df)
            
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