import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import glob
import sys
import os

# 유의성 기준 (인과성 반박이 실패하지 않았다고 간주하는 P-value의 하한선)
P_VALUE_THRESHOLD = 0.05

def load_and_consolidate_data(results_dir: Path) -> pd.DataFrame:
    """지정된 디렉토리에서 모든 배치 결과 CSV 파일을 로드하고 통합합니다."""
    
    # glob을 사용하여 디렉토리 내의 모든 배치 결과 파일 찾기
    search_pattern = str(results_dir / "batch_results_*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"[ERROR] Found no files matching '{search_pattern}'. Please check the directory path.")
        sys.exit(1)
        
    print(f"Found {len(all_files)} result files. Consolidating...")
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"[WARNING] Skipping file {f} due to error: {e}")
            
    if not df_list:
        print("[ERROR] No valid dataframes to process.")
        sys.exit(1)

    # 모든 DataFrame을 하나로 통합
    consolidated_df = pd.concat(df_list, ignore_index=True)
    
    # 데이터 타입 정리 (혹시 모를 오류 대비)
    numeric_cols = ['lr_ate', 'tabpfn_ate', 'placebo_p_value', 'random_cc_p_value']
    for col in numeric_cols:
        consolidated_df[col] = pd.to_numeric(consolidated_df[col], errors='coerce')
        
    print(f"Consolidation complete. Total runs: {len(consolidated_df)}.")
    return consolidated_df


def analyze_results(df: pd.DataFrame):
    """통합된 결과를 바탕으로 주요 통계 및 해석을 수행합니다."""
    
    print("\n" + "="*80)
    print("                 Causal Validation Interpretation Report                ")
    print("="*80)

    # --- 1. Run & Success Overview ---
    print("\n[1. Run & Success Overview]")
    total_runs = len(df)
    success_runs = df['is_successful'].sum()
    success_rate = success_runs / total_runs * 100
    
    print(f"Total DAG-Treatment Runs: {total_runs}")
    print(f"Successful Estimation Runs (TabPFN ATE Available): {success_runs} ({success_rate:.2f}%)")
    
    skipped_df = df[~df['is_successful']]
    if not skipped_df.empty:
        skip_counts = skipped_df['skip_reason'].value_counts(dropna=False)
        print("\nTop Skip Reasons:")
        print(skip_counts.to_string())
    
    df_success = df[df['is_successful']].copy()
    if df_success.empty:
        print("\n[NOTE] No successful runs to perform further analysis.")
        return

    # --- 2. ATE Comparison ---
    print("\n" + "-"*80)
    print("[2. ATE Comparison (TabPFN vs. Linear Regression Baseline)]")
    
    # ATE 요약 통계
    ate_summary = df_success[['lr_ate', 'tabpfn_ate']].agg(['mean', 'median', 'std']).T
    print("\nSummary Statistics of ATEs:")
    print(ate_summary.to_string(float_format="%.6f"))
    
    # 상관관계
    correlation = df_success['lr_ate'].corr(df_success['tabpfn_ate'])
    print(f"\nCorrelation between LR ATE and TabPFN ATE: {correlation:.4f}")
    
    # ATE 차이 분석
    df_success['ate_abs_diff'] = np.abs(df_success['tabpfn_ate'] - df_success['lr_ate'])
    
    print(f"Mean Absolute Difference in ATE: {df_success['ate_abs_diff'].mean():.6f}")
    print(f"Median Absolute Difference in ATE: {df_success['ate_abs_diff'].median():.6f}")


    # --- 3. Refutation Robustness ---
    print("\n" + "-"*80)
    print(f"[3. Refutation Robustness (Criterion: P-value > {P_VALUE_THRESHOLD})]")
    
    # 반박 성공률 계산 (p > 0.05)
    placebo_pass = (df_success['placebo_p_value'] > P_VALUE_THRESHOLD).sum()
    random_cc_pass = (df_success['random_cc_p_value'] > P_VALUE_THRESHOLD).sum()
    
    print(f"Placebo Refutation Pass Rate: {placebo_pass / success_runs * 100:.2f}% ({placebo_pass}/{success_runs})")
    print(f"Random Common Cause Pass Rate: {random_cc_pass / success_runs * 100:.2f}% ({random_cc_pass}/{success_runs})")

    # 모든 반박을 통과한 안정적인 추정
    df_robust = df_success[
        (df_success['placebo_p_value'] > P_VALUE_THRESHOLD) & 
        (df_success['random_cc_p_value'] > P_VALUE_THRESHOLD)
    ].copy()
    
    robust_rate = len(df_robust) / success_runs * 100 if success_runs > 0 else 0
    print(f"\nTotal Runs Robust to ALL Refutations: {len(df_robust)} ({robust_rate:.2f}%)")


    # --- 4. Key Findings ---
    print("\n" + "-"*80)
    print("[4. Key Findings]")

    # 4-1. 가장 강력한 인과 효과 (TabPFN ATE 기준)
    top_5_positive = df_robust.sort_values(by='tabpfn_ate', ascending=False).head(5)
    top_5_negative = df_robust.sort_values(by='tabpfn_ate', ascending=True).head(5)

    print("\nTop 5 Treatment Variables with Largest POSITIVE TabPFN ATE (Robust Runs):")
    if not top_5_positive.empty:
        print(top_5_positive[['dag_idx', 'treatment', 'tabpfn_ate', 'lr_ate']].to_string(index=False, float_format="%.6f"))
    else:
        print("No robust runs found for this analysis.")

    print("\nTop 5 Treatment Variables with Largest NEGATIVE TabPFN ATE (Robust Runs):")
    if not top_5_negative.empty:
        print(top_5_negative[['dag_idx', 'treatment', 'tabpfn_ate', 'lr_ate']].to_string(index=False, float_format="%.6f"))
    else:
        print("No robust runs found for this analysis.")
        
    print("\n" + "="*80)
    print("Report complete.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and interpret Causal Validation Pipeline results.")
    parser.add_argument(
        'results_dir', 
        type=str, 
        help="The directory containing batch_results_XX.csv files. (e.g., ./logs/20251103-060636_4c75a150)"
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"[ERROR] Directory not found: {args.results_dir}")
        sys.exit(1)
        
    df_consolidated = load_and_consolidate_data(results_path)
    analyze_results(df_consolidated)