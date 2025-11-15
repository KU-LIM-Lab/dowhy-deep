import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import glob
import sys
import os
import logging
from do_whynot.config import P_VALUE_THRESHOLD


def load_and_consolidate_batch_results(results_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """지정된 디렉토리에서 모든 배치 결과 CSV 파일을 로드하고 통합합니다."""
    
    # glob을 사용하여 디렉토리 내의 모든 배치 결과 파일 찾기
    search_pattern = str(results_dir / "validations" / "batch_results_*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        logger.info(f"[ERROR] Found no files matching '{search_pattern}'. Please check the directory path.")
        sys.exit(1)
        
    logger.info(f"Found {len(all_files)} result files. Consolidating...")
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            logger.info(f"[WARNING] Skipping file {f} due to error: {e}")
            
    if not df_list:
        logger.info("[ERROR] No valid dataframes to process.")
        sys.exit(1)

    # 모든 DataFrame을 하나로 통합
    consolidated_df = pd.concat(df_list, ignore_index=True)
    
    # 데이터 타입 정리 (혹시 모를 오류 대비)
    numeric_cols = ['lr_ate', 'tabpfn_ate', 'placebo_p_value', 'random_cc_p_value']
    for col in numeric_cols:
        consolidated_df[col] = pd.to_numeric(consolidated_df[col], errors='coerce')
        
    logger.info(f"Consolidation complete. Total runs: {len(consolidated_df)}.")
    return consolidated_df


def analyze_results(df: pd.DataFrame, logger: logging.Logger) -> dict:
    """
    통합된 결과를 바탕으로 주요 통계 및 해석을 수행합니다.
    Top 5 긍정적 ATE를 가진 DAG의 index와 평균 TabPFN ATE를 딕셔너리로 반환합니다.
    """
    
    # 기본 반환 값 초기화
    top_5_positive_dags = {}
    report_lines = []

    report_lines.append("\n" + "="*80)
    report_lines.append("                 Causal Validation Interpretation Report                ")
    report_lines.append("="*80)

    # --- 1. Run & Success Overview ---
    report_lines.append("\n[1. Run & Success Overview]")
    total_runs = len(df)
    success_runs = df['is_successful'].sum()
    success_rate = success_runs / total_runs * 100
    
    report_lines.append(f"Total DAG-Treatment Runs: {total_runs}")
    report_lines.append(f"Successful Estimation Runs (TabPFN ATE Available): {success_runs} ({success_rate:.2f}%)")
    
    skipped_df = df[~df['is_successful']]
    if not skipped_df.empty:
        skip_counts = skipped_df['skip_reason'].value_counts(dropna=False)
        report_lines.append("\nTop Skip Reasons:")
        report_lines.append(skip_counts.to_string()) 
    
    df_success = df[df['is_successful']].copy()
    if df_success.empty:
        report_lines.append("\n[NOTE] No successful runs to perform further analysis.")
        logger.info('\n'.join(report_lines))
        return top_5_positive_dags

    # --- 2. ATE Comparison ---
    report_lines.append("\n" + "-"*80)
    report_lines.append("[2. ATE Comparison (TabPFN vs. Linear Regression Baseline)]")
    
    # ATE 요약 통계
    ate_summary = df_success[['lr_ate', 'tabpfn_ate']].agg(['mean', 'median', 'std']).T
    report_lines.append("\nSummary Statistics of ATEs:")
    report_lines.append(ate_summary.to_string(float_format="%.6f")) # to_string() 추가
    
    # 상관관계
    correlation = df_success['lr_ate'].corr(df_success['tabpfn_ate'])
    report_lines.append(f"\nCorrelation between LR ATE and TabPFN ATE: {correlation:.4f}")
    
    # ATE 차이 분석
    df_success['ate_abs_diff'] = np.abs(df_success['tabpfn_ate'] - df_success['lr_ate'])
    
    report_lines.append(f"Mean Absolute Difference in ATE: {df_success['ate_abs_diff'].mean():.6f}")
    report_lines.append(f"Median Absolute Difference in ATE: {df_success['ate_abs_diff'].median():.6f}")


    # --- 3. Refutation Robustness ---
    report_lines.append("\n" + "-"*80)
    report_lines.append(f"[3. Refutation Robustness (Criterion: P-value > {P_VALUE_THRESHOLD})]")
    
    # 반박 성공률 계산 (p > 0.05)
    placebo_pass = (df_success['placebo_p_value'] > P_VALUE_THRESHOLD).sum()
    random_cc_pass = (df_success['random_cc_p_value'] > P_VALUE_THRESHOLD).sum()
    
    report_lines.append(f"Placebo Refutation Pass Rate: {placebo_pass / success_runs * 100:.2f}% ({placebo_pass}/{success_runs})")
    report_lines.append(f"Random Common Cause Pass Rate: {random_cc_pass / success_runs * 100:.2f}% ({random_cc_pass}/{success_runs})")

    # 모든 반박을 통과한 안정적인 추정 (Robust Runs) 필터링
    df_robust = df_success[
        (df_success['placebo_p_value'] > P_VALUE_THRESHOLD) & 
        (df_success['random_cc_p_value'] > P_VALUE_THRESHOLD)
    ].copy()
    
    robust_rate = len(df_robust) / success_runs * 100 if success_runs > 0 else 0
    report_lines.append(f"\nTotal Runs Robust to ALL Refutations: {len(df_robust)} ({robust_rate:.2f}%)")

    # 추가 분석을 위해 robust runs가 없으면 함수 종료
    if df_robust.empty:
        # 이 시점까지 모인 내용을 출력하고 종료
        logger.info('\n'.join(report_lines)) 
        return top_5_positive_dags


    # --- 4. Key Findings ---
    report_lines.append("\n" + "-"*80)
    report_lines.append("[4. Key Findings (Aggregated by Robust DAG-Treatment Runs)]")
    
    # 4-1. Robust Runs의 DAG별 TabPFN ATE 평균 계산
    dag_ate_summary = df_robust.groupby(['dag_idx', 'treatment'])['tabpfn_ate'].mean().reset_index(name='mean_tabpfn_ate')
    
    # 4-2. 가장 강력한 인과 효과 (TabPFN ATE 평균 기준)
    top_5_positive_dag = dag_ate_summary.sort_values(by='mean_tabpfn_ate', ascending=False).head(5)
    top_5_negative_dag = dag_ate_summary.sort_values(by='mean_tabpfn_ate', ascending=True).head(5)

    report_lines.append("\nTop 5 DAGs with Largest POSITIVE Mean TabPFN ATE (Robust Runs):")
    report_lines.append(top_5_positive_dag.to_string(index=False, float_format="%.6f"))

    report_lines.append("\nTop 5 DAGs with Largest NEGATIVE Mean TabPFN ATE (Robust Runs):")
    report_lines.append(top_5_negative_dag.to_string(index=False, float_format="%.6f"))

    # 반환할 딕셔너리 생성
    top_5_positive_dags_list = top_5_positive_dag.rename(columns={'mean_tabpfn_ate': 'ate_mean', 'treatment': 'treatment_column'}).to_dict('records')

    report_lines.append("\n" + "="*80)
    report_lines.append("Report complete.")
    
    logger.info('\n'.join(report_lines))
    
    return top_5_positive_dags_list