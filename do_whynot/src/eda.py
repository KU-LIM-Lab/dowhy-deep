import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import List

# 변수 분류
def classify_variables(df):
    """
    EDA를 위해 변수들을 Outcome, Categorical, Continuous, Exclude 그룹으로 분류
    """
    # Outcome
    outcome_col = 'ACQ_180_YN'
    
    # 1. Nominal/Low-Card. Categorical (범주형)
    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    
    # Int64 중 범주형으로 간주할 변수 (YN, CD, 간단한 코드)
    yn_cols = [col for col in df.columns if col.endswith('_YN') and col != outcome_col]
    code_cols = ['INFO_OTPB_GRAD_CD', 'EMPL_STLE_CD', 'BFR_OCTR_YN', 'AGE']
    simple_code_cols = ['IDIF_AOFR_YN', 'AFIV_RDJT_PSBL_YN', 'DRV_PSBL_YN', 'SMS_RCYN', 'EMAIL_OTPB_YN', 'MPNO_OTPB_YN']
    
    categorical_cols.extend(yn_cols)
    for col in code_cols + simple_code_cols:
        if col not in categorical_cols:
            categorical_cols.append(col)
            
    # 2. Continuous/Count (연속형/수치형)
    continuous_cols = list(df.select_dtypes(include=['float64']).columns)
    count_amt_cols = [col for col in df.columns if col.endswith(('_AMT', '_CT', '_DYCT', '_RATE', '_NUM', '_NMPR', '_SNSC')) and col not in categorical_cols and col != outcome_col]
    for col in count_amt_cols:
        if col not in continuous_cols:
            continuous_cols.append(col)
            
    # 3. ID/High-Card./Dates (제외 또는 특수 처리)
    exclude_cols = ['JHNT_MBN', 'JHNT_CTN', 'CLOS_YM', 'JHCR_DE', 'JHNT_CLOS_DE', 'HOPE_JSCD1', 'HOPE_JSCD2', 'HOPE_JSCD3', 'HPAR_CD1', 'HPAR_CD2', 'HPAR_CD3', 'MAJR_CD1', 'AREA_CD', 'KECO_CD1', 'LAST_JSCD', 'ETL_DT', 'MAKE_DT', 'ACQ_DT', 'EMPN_DE', 'LAST_FRFT_DE']
    
    # 최종 리스트 정리
    final_categorical = [col for col in categorical_cols if col not in exclude_cols and col != outcome_col and col in df.columns]
    final_continuous  = [col for col in continuous_cols if col not in exclude_cols and col != outcome_col and col in df.columns]
    
    return outcome_col, final_categorical, final_continuous, exclude_cols

# 라벨 인코딩된 범주형(df[cat_cols])의 범주 개수 확인
import json

def _get_postprocess_cat_cols(output_dir: Path, df: pd.DataFrame, outcome_col: str, exclude_cols: list, logger: logging.Logger):
    """
    preprocessor.postprocess()가 저장한 label_encoding_map.json의 키(=cat_cols)를 사용해
    정확히 동일한 범주형 컬럼 집합을 반환. 파일이 없으면 object dtype 기반으로 fallback.
    """
    map_path = Path(output_dir) / "label_encoding_map.json"
    if map_path.exists():
        try:
            with map_path.open("r", encoding="utf-8") as f:
                enc_map = json.load(f)
            # 키만 cat_cols로 사용
            cat_cols = [c for c in enc_map.keys() if c in df.columns and c != outcome_col and c not in exclude_cols]
            if cat_cols:
                logger.info(f"[EDA] Using cat_cols from label_encoding_map.json (n={len(cat_cols)}).")
                return cat_cols
        except Exception as e:
            logger.warning(f"[EDA] Failed to read label_encoding_map.json: {e}")

    # fallback: object dtype
    cat_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != outcome_col and c not in exclude_cols
    ]
    logger.info(f"[EDA] Fallback: inferring cat_cols from dtypes (n={len(cat_cols)}).")
    return cat_cols


def plot_cat_cardinality_simple_strict(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    outcome_col: str,
    exclude_cols: list,
) -> None:
    """
    preprocessor의 cat_cols(= label_encoding_map.json의 키)만 대상으로
    x=컬럼명, y=nunique 의 단일 바플롯 저장.
    - 범주 수 20개 이하: 파란색
    - 범주 수 20개 초과: 회색
    - y=20 위치에 빨간 점선 수평선
    """
    cat_cols = _get_postprocess_cat_cols(output_dir, df, outcome_col, exclude_cols, logger)
    if not cat_cols:
        logger.info("[EDA] No categorical columns to plot (cat_cols empty).")
        return

    # 각 컬럼별 고유값 개수 계산 후 오름차순 정렬
    nunique_s = df[cat_cols].nunique(dropna=True).sort_values(ascending=True)

    colors = ["tab:blue" if v <= 20 else "lightgray" for v in nunique_s.values]

    plt.figure(figsize=(max(10, 0.6 * len(nunique_s)), 6))

    ax = sns.barplot(x=nunique_s.index, y=nunique_s.values, palette=colors)
    ax.set_xlabel("Categorical Columns (postprocess cat_cols)")
    ax.set_ylabel("Number of Unique Categories")
    ax.set_title("Cardinality of Categorical Columns (df[cat_cols])")

    # 4) y=20 기준선 (빨간 점선)
    ax.axhline(y=20, linestyle="--", color="red")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = output_dir / "05_cat_cardinality_simple.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f" -> Saved categorical cardinality plot: {out_path.name}")


# EDA 메인 함수
def perform_eda(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    전처리된 DataFrame에 대해 종합 EDA를 수행하고 결과를 output_dir에 저장
    """
    logger.info("=" * 70)
    logger.info("Starting Comprehensive Exploratory Data Analysis (EDA) for all variables.")
    logger.info("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"EDA results will be saved to: {output_dir.resolve()}")

    # 변수 분류
    Y, T_Z_cat, T_Z_cont, T_Z_exclude = classify_variables(df)
    logger.info(f"Y (Outcome): {Y}")
    logger.info(f"T/Z Categorical Candidates ({len(T_Z_cat)}): {T_Z_cat[:5]}...")
    logger.info(f"T/Z Continuous Candidates ({len(T_Z_cont)}): {T_Z_cont[:5]}...")
    logger.info(f"ID/High-Card. Excluded ({len(T_Z_exclude)}): {T_Z_exclude[:5]}...")

    try:
        plot_cat_cardinality_simple_strict(
            df=df,
            output_dir=output_dir,
            logger=logger,
            outcome_col=Y,
            exclude_cols=T_Z_exclude,
    )
    except Exception as e:
        logger.error(f"Error during label cardinality/distribution EDA: {e}")

    # ----------------------------------------------------------------------
    # 1. Continuous/Count Variables Analysis
    # ----------------------------------------------------------------------
    logger.info("\n[STEP 1] Analyzing Continuous Variable Distributions & Relationship with Y...")
    try:
        n_plots = len(T_Z_cont)
        if n_plots > 0:
            rows = int(n_plots**0.5)
            cols = (n_plots + rows - 1) // rows
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()
            for i, col in enumerate(T_Z_cont):
                sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(col, fontsize=10)
                axes[i].set_xlabel("")
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.suptitle(f'Distribution of {len(T_Z_cont)} Continuous/Count Variables', y=1.02)
            plt.tight_layout()
            dist_path = output_dir / "01_continuous_distribution_all.png"
            fig.savefig(dist_path)
            plt.close(fig)
            logger.info(f" -> Saved Distribution Plot: {dist_path.name}")
    except Exception as e:
        logger.error(f"Error during continuous distribution EDA: {e}")
        
    # 1-2. Outcome Relationship (Box Plot by Y=0/1)
    try:
        n_plots = len(T_Z_cont)
        if n_plots > 0:
            rows = int(n_plots**0.5)
            cols = (n_plots + rows - 1) // rows
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()
            for i, col in enumerate(T_Z_cont):
                sns.boxplot(x=df[Y].astype(str), y=df[col], ax=axes[i], palette="viridis")
                axes[i].set_title(f'{col} by {Y}', fontsize=10)
                axes[i].set_xlabel(f'{Y} (0: No, 1: Yes)')
                axes[i].set_ylabel("")
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.suptitle(f'Conditional Distribution of Continuous/Count Variables by Outcome ({Y})', y=1.02)
            plt.tight_layout()
            cond_path = output_dir / "02_continuous_conditional_by_Y.png"
            fig.savefig(cond_path)
            plt.close(fig)
            logger.info(f" -> Saved Conditional Box Plots: {cond_path.name}")
    except Exception as e:
        logger.error(f"Error during continuous conditional EDA: {e}")
        
    # ----------------------------------------------------------------------
    # 2. Categorical Variables Analysis (기존: Y 조건부 확률)
    # ----------------------------------------------------------------------
    logger.info("\n[STEP 2] Analyzing Categorical Variables & Conditional Probability of Y...")
    try:
        plot_cols = T_Z_cat[:15]
        if len(plot_cols) > 0:
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            axes = axes.flatten()
            for i, col in enumerate(plot_cols):
                top_cats = df[col].value_counts().nlargest(5).index.tolist()
                df_plot = df[df[col].isin(top_cats)].copy()
                y_mean = df_plot.groupby(col)[Y].mean().sort_values(ascending=False)
                sns.barplot(x=y_mean.index, y=y_mean.values, ax=axes[i], palette="cividis")
                axes[i].set_title(col, fontsize=10)
                axes[i].set_xlabel("")
                axes[i].set_ylabel(f'P({Y}=1)')
                axes[i].tick_params(axis='x', rotation=45)
            plt.suptitle(f'Conditional Probability P(Y=1) by Top 15 Categorical Variables', y=1.02)
            plt.tight_layout()
            cat_path = output_dir / "03_categorical_conditional_by_Y.png"
            fig.savefig(cat_path)
            plt.close(fig)
            logger.info(f" -> Saved Conditional Bar Plots: {cat_path.name}")
    except Exception as e:
        logger.error(f"Error during categorical conditional EDA: {e}")
        
    # ----------------------------------------------------------------------
    # 3. Correlation Analysis
    # ----------------------------------------------------------------------
    logger.info("\n[STEP 3] Analyzing Correlation Matrix...")
    try:
        corr_cols = [Y] + T_Z_cont
        corr_df = df[corr_cols]
        corr_matrix = corr_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
        plt.title(f'Correlation Matrix of Continuous Variables and Outcome ({Y})', fontsize=14)
        corr_path = output_dir / "04_correlation_matrix.png"
        plt.savefig(corr_path)
        plt.close()
        logger.info(f" -> Saved Correlation Matrix: {corr_path.name}")
        y_corr = corr_matrix[Y].sort_values(ascending=False).drop(Y)
        logger.info(f"\nCorrelation with {Y}:\n{y_corr.to_string()}")
    except Exception as e:
        logger.error(f"Error during correlation matrix EDA: {e}")

    logger.info("\n Comprehensive Exploratory Data Analysis (EDA) completed successfully.")
    logger.info("=" * 70)

if __name__ == '__main__':
    pass