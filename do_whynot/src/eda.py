import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _get_postprocess_cat_cols(output_dir: Path, df: pd.DataFrame,
                              outcome_col: str, exclude_cols: list,
                              logger: logging.Logger):

    map_path = Path(output_dir) / "label_encoding_map.json"
    if map_path.exists():
        try:
            with map_path.open("r", encoding="utf-8") as f:
                enc_map = json.load(f)
            cat_cols = [
                c for c in enc_map.keys()
                if c in df.columns and c != outcome_col and c not in exclude_cols
            ]
            if cat_cols:
                logger.info(f"[EDA] Using cat_cols from label_encoding_map.json (n={len(cat_cols)}).")
                return cat_cols
        except Exception as e:
            logger.warning(f"[EDA] Failed to read label_encoding_map.json: {e}")

    # fallback
    cat_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != outcome_col and c not in exclude_cols
    ]
    logger.info(f"[EDA] Fallback: inferring cat_cols from dtypes (n={len(cat_cols)}).")
    return cat_cols


def classify_variables(df: pd.DataFrame):
    outcome_col = 'ACQ_180_YN'

    categorical_cols = list(df.select_dtypes(include=['object']).columns)

    yn_cols = [col for col in df.columns if col.endswith('_YN') and col != outcome_col]
    code_cols = ['INFO_OTPB_GRAD_CD', 'EMPL_STLE_CD', 'BFR_OCTR_YN', 'AGE']
    simple_code_cols = [
        'IDIF_AOFR_YN', 'AFIV_RDJT_PSBL_YN', 'DRV_PSBL_YN', 'SMS_RCYN',
        'EMAIL_OTPB_YN', 'MPNO_OTPB_YN'
    ]
    categorical_cols.extend(yn_cols)
    for col in code_cols + simple_code_cols:
        if col not in categorical_cols:
            categorical_cols.append(col)

    continuous_cols = list(df.select_dtypes(include=['float64']).columns)
    count_amt_cols = [
        col for col in df.columns
        if col.endswith(('_AMT', '_CT', '_DYCT', '_RATE', '_NUM', '_NMPR', '_SNSC'))
        and col not in categorical_cols and col != outcome_col
    ]
    for col in count_amt_cols:
        if col not in continuous_cols:
            continuous_cols.append(col)

    exclude_cols = [
        'JHNT_MBN', 'JHNT_CTN', 'CLOS_YM', 'JHCR_DE', 'JHNT_REG_DT',
        'HOPE_JSCD1', 'HOPE_JSCD2', 'HOPE_JSCD3',
        'HPAR_CD1', 'HPAR_CD2', 'HPAR_CD3',
        'MAJR_CD1', 'AREA_CD', 'KECO_CD1', 'LAST_JSCD',
        'ETL_DT', 'MAKE_DT', 'ACQ_DT', 'EMPN_DE', 'LAST_FRFT_DE'
    ]

    final_categorical = [
        col for col in categorical_cols
        if col not in exclude_cols and col != outcome_col and col in df.columns
    ]
    final_continuous = [
        col for col in continuous_cols
        if col not in exclude_cols and col != outcome_col and col in df.columns
    ]

    return outcome_col, final_categorical, final_continuous, exclude_cols

# Categorical cardinality plot
def plot_cat_cardinality_simple_strict(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    outcome_col: str,
    exclude_cols: list,
) -> None:

    cat_cols = _get_postprocess_cat_cols(output_dir, df, outcome_col, exclude_cols, logger)
    if not cat_cols:
        logger.info("[EDA] No categorical columns to plot (cat_cols empty).")
        return

    nunique_s = df[cat_cols].nunique(dropna=True).sort_values(ascending=True)

    logger.info("[EDA] Categorical cardinality per column:")
    for col, n in nunique_s.items():
        logger.info(f"  - {col}: {int(n)} unique categories")

    colors = ["tab:blue" if v <= 20 else "lightgray" for v in nunique_s.values]

    plt.figure(figsize=(max(10, 0.6 * len(nunique_s)), 6))
    ax = sns.barplot(x=nunique_s.index, y=nunique_s.values, palette=colors)
    ax.set_xlabel("Categorical Columns (postprocess cat_cols)")
    ax.set_ylabel("Number of Unique Categories")
    ax.set_title("Cardinality of Categorical Columns (df[cat_cols])")

    ax.axhline(y=20, linestyle="--", color="red")

    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    out_path = output_dir / "05_cat_cardinality_simple.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f" -> Saved categorical cardinality plot: {out_path.name}")


# perform_eda (MAIN)
def perform_eda(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:

    logger.info("\n================== Exploratory Data Analysis (EDA) ==================\n")

    Y, T_Z_cat, T_Z_cont, T_Z_exclude = classify_variables(df)
    df = df.copy()
    df[Y] = pd.to_numeric(df[Y], errors="coerce")

    date_like_cols = [c for c in df.columns if "_ym" in c.lower()]
    if date_like_cols:
        logger.info(f"[EDA] Excluding date-like columns from EDA (contains '_ym'): {date_like_cols}")
        T_Z_cat = [c for c in T_Z_cat if c not in date_like_cols]
        T_Z_cont = [c for c in T_Z_cont if c not in date_like_cols]

    MAX_PLOTS = 15

    # variables by Var(P(Y=1|X))
    # Categorical
    cat_var_scores = {}
    for col in T_Z_cat:
        try:
            y_mean = df.groupby(col)[Y].mean()
            var = y_mean.var()
            cat_var_scores[col] = float(var) if pd.notnull(var) else 0.0
        except:
            cat_var_scores[col] = 0.0

    sorted_cat = sorted(cat_var_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info("[EDA] Top categorical by Var(P(Y=1)|X):")
    for col, score in sorted_cat[:MAX_PLOTS]:
        logger.info(f"  - {col}: var={score:.6f}")

    exclude_cat = {"resume_title", "basic_resume_yn"}
    ranked_cat_for_cond = [c for c, _ in sorted_cat if c not in exclude_cat]
    top_cat_for_cond = ranked_cat_for_cond[:MAX_PLOTS]

    # Continuous
    cont_var_scores = {}
    for col in T_Z_cont:
        try:
            s = df[col]
            valid = s.dropna()
            if valid.nunique() < 2:
                cont_var_scores[col] = 0.0
                continue

            q = min(10, valid.nunique())
            binned = pd.qcut(valid, q=q, duplicates="drop")

            tmp = pd.DataFrame({"bin": binned, Y: df.loc[valid.index, Y]})
            var = tmp.groupby("bin")[Y].mean().var()
            cont_var_scores[col] = float(var) if pd.notnull(var) else 0.0
        except:
            cont_var_scores[col] = 0.0

    sorted_cont = sorted(cont_var_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info("[EDA] Top continuous by Var(P(Y=1)|binned X):")
    for col, score in sorted_cont[:MAX_PLOTS]:
        logger.info(f"  - {col}: var={score:.6f}")

    top_cont_for_plots = [c for c, _ in sorted_cont[:MAX_PLOTS]]

    # Continuous plots
    logger.info("\n[STEP 1] Continuous Distributions & Conditional Boxplots")

    try:
        plot_cols = top_cont_for_plots
        n = len(plot_cols)
        if n > 0:
            rows = int(n**0.5)
            cols = (n + rows - 1) // rows
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()
            for i, col in enumerate(plot_cols):
                sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            out = output_dir / "01_continuous_distribution_all.png"
            plt.savefig(out)
            plt.close()
            logger.info(f" -> Saved: {out.name}")
    except Exception as e:
        logger.error(f"Continuous distribution error: {e}")

    try:
        plot_cols = top_cont_for_plots
        n = len(plot_cols)
        if n > 0:
            rows = int(n**0.5)
            cols = (n + rows - 1) // rows
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()
            for i, col in enumerate(plot_cols):
                sns.boxplot(x=df[Y].astype(str), y=df[col], ax=axes[i], palette="viridis")
                axes[i].set_title(f"{col} by {Y}")
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            out = output_dir / "02_continuous_conditional_by_Y.png"
            plt.savefig(out)
            plt.close()
            logger.info(f" -> Saved: {out.name}")
    except Exception as e:
        logger.error(f"Conditional continuous error: {e}")

    # Categorical Conditional P(Y=1)
    logger.info("\n[STEP 2] Categorical Variables - Conditional P(Y=1)")

    try:
        plot_cols = top_cat_for_cond
        n = len(plot_cols)
        if n > 0:
            rows, cols = 3, 5
            fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
            axes = axes.flatten()
            for i, col in enumerate(plot_cols):
                top_cats = df[col].value_counts().nlargest(5).index.tolist()
                df_plot = df[df[col].isin(top_cats)].copy()
                y_mean = df_plot.groupby(col)[Y].mean().sort_values(ascending=False)
                sns.barplot(x=y_mean.index, y=y_mean.values, ax=axes[i], palette="cividis")
                axes[i].set_title(col)
                axes[i].tick_params(axis='x', rotation=45)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            out = output_dir / "03_categorical_conditional_by_Y.png"
            plt.savefig(out)
            plt.close()
            logger.info(f" -> Saved: {out.name}")
    except Exception as e:
        logger.error(f"Categorical conditional error: {e}")

    # Correlation Matrix
    logger.info("\n[STEP 3] Correlation Matrix")

    try:
        corr_cols = [Y] + top_cont_for_plots
        corr_df = df[corr_cols].fillna(0)
        corr = corr_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        out = output_dir / "04_correlation_matrix.png"
        plt.savefig(out)
        plt.close()
        logger.info(f" -> Saved: {out.name}")
    except Exception as e:
        logger.error(f"Correlation matrix error: {e}")

    # Categorical Cardinality
    logger.info("\n[STEP 4] Categorical Cardinality")
    plot_cat_cardinality_simple_strict(df, output_dir, logger, Y, T_Z_exclude)

    logger.info("\n================== EDA Completed Successfully ==================\n")