import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import warnings
import uuid
from datetime import datetime
import pytz
import csv
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from do_whynot.src.preprocessor_3 import build_pipeline_wide, postprocess
from do_whynot.src.dag_parser import extract_roles_general
from do_whynot.src.inference_top1 import llm_inference
from do_whynot.src.interpretator import load_and_consolidate_batch_results, analyze_results
from do_whynot.src.eda import perform_eda
from do_whynot.src.estimation import get_treatment_type, validate_tabpfn_estimator, run_tabpfn_estimation
from do_whynot.src.prediction import run_prediction_pipeline

from do_whynot.config import IS_TEST_MODE, TEST_SAMPLE_SIZE, BATCH_SIZE, DATA_OUTPUT_DIR, DAG_INDICES_TEST, DAG_INDICES, DAG_DIR, RAW_CSV, EXCLUDE_COLS

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

# =========================
# 6) 후처리 (옵션)
# =========================
def postprocess_base(df: pd.DataFrame, logger: logging.LoggerAdapter, data_output_dir) -> pd.DataFrame:
    logger.info("Starting postprocessing: Binary mapping and date calculation.")
    
    # ---- (1) 바이너리 매핑 ----
    bin_map = {"예":1, "아니오":0, "아니요":0, "필요":1, "불필요":0, "Y": 1, "Yes": 1, "y":1, "yes":1,  "N": 0, "No": 0, "n":0}
    mapped_cols = [] 
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(bin_map)
            if df[col].dtype != object:
                mapped_cols.append(col)
                
    if mapped_cols:
        logger.info(f"Successfully applied Binary Mapping to {len(mapped_cols)} columns: {', '.join(mapped_cols)}") 
    else:
        logger.info("No object columns were successfully converted by binary mapping.")

    # ---- (2) 날짜 차이 계산  ----
    date_diff_cols = [] 
    if "JHCR_DE" in df.columns:
        anchor = pd.to_datetime(df["JHCR_DE"], errors="coerce")
        date_cols = [c for c in df.columns if any(x in c.upper() for x in ["DE","DT","DATE","BGDE","ENDE","STDT","ENDT"]) and "MDTN" not in c.upper()]
        
        for col in date_cols:
            if col == "JHCR_DE":
                continue
            
            vals = pd.to_datetime(df[col], errors="coerce")
            
            if not anchor.isna().all() and not vals.isna().all():
                diff = (vals - anchor).dt.days
                df[col] = diff.abs()
                date_diff_cols.append(col)
            else:
                 logger.warning(f"Skipped date diff for column '{col}' due to all-NaN anchor or all-NaN target date values.")

        if date_diff_cols:
            logger.info(f"Calculated Date Difference (days from JHCR_DE) for {len(date_diff_cols)} columns: {', '.join(date_diff_cols)}")
        else:
            logger.info("No date columns were processed for date difference calculation.")
    else:
        logger.warning("Anchor column 'JHCR_DE' not found. Skipping date difference calculation.")

    # ---- (3) 모든 값이 결측인 컬럼 제거 ----
    original_cols = df.shape[1]
    cols_to_drop = df.columns[df.isnull().all()].tolist()
    
    df = df.dropna(axis=1, how="all")
    dropped_cols_count = original_cols - df.shape[1]
    
    if dropped_cols_count > 0:
        logger.info(f"Dropped {dropped_cols_count} columns that were entirely missing values: {', '.join(cols_to_drop)}")
    
    # ---- (4) label encoding ----
    clos_ym_prefix_cols = [c for c in df.columns if c.startswith('CLOS_YM')]
    jhcr_de_prefix_cols = [c for c in df.columns if c.startswith('JHCR_DE')]
    excluded_cols = list(set(EXCLUDE_COLS + clos_ym_prefix_cols + jhcr_de_prefix_cols))

    cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c not in excluded_cols]

    encoding_map = {}
    for c in cat_cols:
        cat_dtype = pd.Categorical(df[c], categories=sorted(df[c].dropna().unique()))
        
        mapping = {label: code for code, label in enumerate(cat_dtype.categories)}
        encoding_map[c] = mapping
        
        df[c] = cat_dtype.codes

    df = df.assign(**{c: df[c].astype('int64') for c in df.select_dtypes(include=['bool']).columns})
    df[cat_cols] = df[cat_cols].astype('str')

    # 로깅 추가
    if cat_cols:
        logger.info(f"Applied Label Encoding to {len(cat_cols)} columns:")
        logger.info(f"    Encoded Columns: {cat_cols}") 
        logger.info(f"    Excluded Columns: {', '.join(excluded_cols)}")

        map_path = data_output_dir / "label_encoding_map.json"
        try:
            with map_path.open("w", encoding="utf-8") as f:
                json.dump(encoding_map, f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"Label Encoding Map saved to: {map_path.name}")
        except Exception as e:
            logger.error(f"Failed to save Label Encoding Map to {map_path.name}: {e}")
    else:
        logger.info(f"No categorical columns (excluding {', '.join(excluded_cols)}) found for Label Encoding.")
    
    logger.info("Postprocessing complete.")
    return df

def main():

    # 로거 초기화
    main_logger = setup_logger()
    main_logger.info("Starting Data Preprocessing Pipeline from imported functions.")

    # --- 1. 전처리 실행 (데이터 전체에 대해 단 1회 실행) ---
    try:
        dtype_map = {}
        for k in ["JHNT_MBN", "JHNT_CTN"]:
            dtype_map[k] = str

        base = pd.read_csv(RAW_CSV, encoding="utf-8", dtype=dtype_map)
        final_df = postprocess_base(base, main_logger, DATA_OUTPUT_DIR) 
        main_logger.info(f"Preprocessing complete. Final DataFrame shape: {final_df.shape}")
        
    except Exception as e:
        main_logger.error(f"[Fatal] Preprocessing failed during execution: {e}")
        sys.exit(1)

    preprocessed_path = DATA_OUTPUT_DIR / "preprocessed_df_base.csv"
    final_df.to_csv(preprocessed_path, index=False, encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar='\\')
    main_logger.info(f"[OK] Preprocessed data saved to: {preprocessed_path.name}")

    try:
        perform_eda(final_df, DATA_OUTPUT_DIR, main_logger)
    except Exception as e:
        main_logger.error(f"[skip] Skip EDA due to the error: {e}")

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

    multi_class_fixed_params = {}

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        
        batch_df = final_df.iloc[start_idx:end_idx].copy()
        
        main_logger.info("=" * 70)
        main_logger.info(f"BATCH {i+1}/{num_batches}: Processing rows {start_idx} to {end_idx-1} (Size: {len(batch_df)})")
        main_logger.info("=" * 70)
        
        batch_results = []

        batch_result_folder = RESULTS_DIR / "validations"
        batch_result_folder.mkdir(parents=True, exist_ok=True)

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
            
            batch_result_file = batch_result_folder / f"batch_results_{i+1:02d}.csv"
            batch_results_df.to_csv(batch_result_file, index=False, encoding="utf-8")
            main_logger.info(f" BATCH {i+1} results saved to: {batch_result_file.name}")
    
    final_result_file = RESULTS_DIR / "all_validation_results.csv"
    all_results_df.to_csv(final_result_file, index=False, encoding="utf-8")
    main_logger.info("All validation results saved to: %s", final_result_file.name)

    main_logger.info("-" * 50)
    main_logger.info("Validation runs complete.")
    
    # --- 3. 최종 해석 로직 ---
    main_logger.info("Starting Causal Interpretation Analysis...")
    
    df_consolidated = load_and_consolidate_batch_results(RESULTS_DIR, main_logger)
    top_5_dags_info = analyze_results(df_consolidated, main_logger)
    main_logger.info("Interpretation analysis complete.")

    # --- 4. 최종 예측 로직 ---
    main_logger.info("Starting Prediction Pipeline on Top DAGs...")

    all_final_preds = pd.DataFrame()
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        
        batch_final_df = final_df.iloc[start_idx:end_idx].copy()

        pred_result = run_prediction_pipeline(
            final_merged_df=batch_final_df,
            top_5_dags_info=top_5_dags_info,
            outcome_name="ACQ_180_YN",
            data_output_dir=RESULTS_DIR,
            logger=main_logger,
            batch_id=i,
        )
        
        all_final_preds = pd.concat([all_final_preds, pred_result], ignore_index=True)
        main_logger.info(f"Batch{i+1} prediction results accumulated. Current total rows: {len(all_final_preds)}")
    
    main_logger.info("Merging all predictions with the final preprocessed DataFrame...")
    
    final_df['JHNT_MBN'] = final_df['JHNT_MBN'].astype(str)
    all_final_preds['JHNT_MBN'] = all_final_preds['JHNT_MBN'].astype(str)
    
    final_merged_with_all_preds = pd.merge(final_df, all_final_preds, on='JHNT_MBN', how='left')
    
    all_final_preds_file = RESULTS_DIR / "final_df_all_predictions.csv"
    final_merged_with_all_preds.to_csv(all_final_preds_file, index=False, encoding="utf-8")
    
    main_logger.info("Final DataFrame shape: %s", final_merged_with_all_preds.shape)
    main_logger.info("All final predictions merged and saved to: %s", all_final_preds_file.name)
    main_logger.info("Prediction complete.")

if __name__ == "__main__":
    main()