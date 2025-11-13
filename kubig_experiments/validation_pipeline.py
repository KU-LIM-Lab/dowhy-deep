import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import warnings
import uuid
from datetime import datetime
import pytz

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from kubig_experiments.src.preprocessor import build_pipeline_wide, postprocess
from kubig_experiments.src.dag_parser import extract_roles_general, dot_to_nx
from kubig_experiments.src.inference_top1 import llm_inference
from kubig_experiments.src.interpretator import load_and_consolidate_data, analyze_results
from kubig_experiments.src.eda import perform_eda
from kubig_experiments.src.estimation import get_treatment_type, validate_tabpfn_estimator, run_tabpfn_estimation

from config import IS_TEST_MODE, TEST_SAMPLE_SIZE, BATCH_SIZE, DATA_OUTPUT_DIR, DAG_INDICES_TEST, DAG_INDICES, DAG_DIR

RESULTS_DIR = None
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


def main():

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

    if not IS_TEST_MODE:
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        main_logger.info("Data shuffled successfully before batching. Final DataFrame shape: %s", final_df.shape)

    total_rows = len(final_df)
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    if IS_TEST_MODE:
        dag_indices = DAG_INDICES_TEST  
    else:
        dag_indices = DAG_INDICES

    main_logger.info("-" * 50)
    main_logger.info("Starting batch validation runs.")
    main_logger.info(f"Total rows: {total_rows}. Batch size: {BATCH_SIZE}. Total batches: {num_batches}.")

    all_results_df = pd.DataFrame()
    multi_class_fixed_params = {}

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        
        batch_df = final_df.iloc[start_idx:end_idx].copy()
        
        main_logger.info("=" * 70)
        main_logger.info(f"BATCH {i+1}/{num_batches}: Processing rows {start_idx} to {end_idx-1} (Size: {len(batch_df)})")
        main_logger.info("=" * 70)
                
        main_logger.info(f"Starting LLM Inference for BATCH {i+1}...")
        llm_preds_df = llm_inference(batch_df, main_logger, i, IS_TEST_MODE, DATA_OUTPUT_DIR) 
        main_logger.info(f"LLM Inference for BATCH {i+1} complete. Predictions saved.")
        
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
            
            dag_file = DAG_DIR / f"dag_{dag_idx}.txt"
            if not dag_file.exists():
                continue
            graph_txt = dag_file.read_text(encoding="utf-8")
            roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
            dag_treatment = roles["treatment"]
            
            treatment_type = get_treatment_type(batch_df, dag_treatment)

            main_logger.info("-" * 50)
            main_logger.info(f"[Batch {i+1}] Processing DAG index: {dag_idx} (Treatment: {dag_treatment}, Type: {treatment_type})")
            
            initial_results = validate_tabpfn_estimator(dag_idx, main_logger, batch_df, treatment_type) 
            
            if not initial_results.get("is_successful"):
                main_logger.warning(f"[Batch {i+1}] Initial validation failed. Reason: {initial_results.get('skip_reason')}")
                result_dict = {k: v for k, v in initial_results.items() if k not in ["model", "identified", "df_processed"]}
                if result_dict.get("skip_reason") == "Unknown treatment type":
                     pass
                elif result_dict.get("skip_reason") is None:
                    result_dict["skip_reason"] = "Initial validation/identification failure"
                
            else:
                fixed_params = multi_class_fixed_params.get(dag_idx)
                
                if treatment_type == "multi-class":
                    # 첫 번째 배치일 때만 고정값을 정하고, 이후 배치부터는 고정값을 사용
                    fixed_baseline = fixed_params.get("baseline") if fixed_params else None
                    fixed_treatment_value = fixed_params.get("treatment_value") if fixed_params else None
                else:
                    fixed_baseline = None
                    fixed_treatment_value = None

                # run_tabpfn_estimation 호출
                estimation_results = run_tabpfn_estimation(
                    model=initial_results["model"],
                    identified=initial_results["identified"],
                    df_copy=initial_results["df_processed"],
                    dag_file_name=initial_results["dag_file"],
                    dag_treatment=dag_treatment,
                    treatment_type=treatment_type,
                    logger=main_logger,
                    fixed_baseline=fixed_baseline,
                    fixed_treatment_value=fixed_treatment_value
                )
                
                # 최종 결과 딕셔너리 조합
                result_dict = {
                    "dag_idx": dag_idx,
                    "dag_file": initial_results["dag_file"],
                    "treatment": dag_treatment,
                    "lr_ate": initial_results["lr_ate"], 
                    **estimation_results
                }

                if treatment_type == "multi-class" and i == 0 and estimation_results["is_successful"]:
                    multi_class_fixed_params[dag_idx] = {
                        "baseline": estimation_results["multi_class_baseline"],
                        "treatment_value": estimation_results["multi_class_treatment_value"]
                    }
                    main_logger.info(f"[Batch {i+1}] Multi-Class fixed parameters set for DAG {dag_idx}: Base={estimation_results['multi_class_baseline']}, Tx={estimation_results['multi_class_treatment_value']}")

            if result_dict:
                result_dict["batch_id"] = i + 1 
                result_dict["batch_size"] = len(batch_df) 
                
                batch_results.append(result_dict)
        
        if batch_results:
            batch_results_df = pd.DataFrame(batch_results)
            all_results_df = pd.concat([all_results_df, batch_results_df], ignore_index=True) # 전체 결과에 누적
            
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