import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path


def main():
    """
    seis_data.csv의 HOPE_JSCD1 분포를 단순 히스토그램으로 그려
    job_subcategory_analysis 폴더에 저장하고, 빈도 구간별 개수를 JSON으로 저장합니다.
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent  # laborlab_2/
    seis_csv = base_dir / "data" / "seis_data" / "seis_data.csv"
    # merged_csv = base_dir / "log" / "merged_df.csv"

    if not seis_csv.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {seis_csv}")

    df = pd.read_csv(seis_csv)
    if "HOPE_JSCD1" not in df.columns:
        raise KeyError("HOPE_JSCD1 컬럼이 존재하지 않습니다.")
    counts = df["HOPE_JSCD1"].value_counts(dropna=False).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("HOPE_JSCD1 분포")
    ax.set_xlabel("HOPE_JSCD1")
    ax.set_ylabel("Count")
    plt.xticks(rotation=90, ha="center", fontsize=8)
    plt.tight_layout()

    hist_path = script_dir / "hope_jscd1_hist.png"
    fig.savefig(hist_path, dpi=150)
    print(f"✅ 히스토그램 저장 완료: {hist_path}")

    # 빈도 구간별 요약
    bin_counts = {
        "0": int((counts == 0).sum()),
        "0~5000": int(((counts > 0) & (counts <= 5000)).sum()),
        "5000~10000": int(((counts > 5000) & (counts <= 10000)).sum()),
        "10000~": int((counts > 10000).sum()),
    }
    json_path = script_dir / "hope_jscd1_bins.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bin_counts, f, ensure_ascii=False, indent=2)
    print(f"✅ 구간별 요약 저장 완료: {json_path}")


if __name__ == "__main__":
    main()
