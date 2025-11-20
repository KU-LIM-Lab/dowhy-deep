import argparse
import pandas as pd
from pathlib import Path
from do_whynot.config import DATA_OUTPUT_DIR

def main():
    input_path = DATA_OUTPUT_DIR / "intermediate_preprocessed_df.csv"
    output_path = DATA_OUTPUT_DIR / "intermediate_preprocessed_df_head_1000.csv"
    rows = 1000

    # Output 파일 지정 없으면 자동 생성

    print(f"📥 Reading: {input_path}")
    print(f"📤 Saving first {rows} rows to: {output_path}")

    # CSV 읽기
    df = pd.read_csv(input_path)

    # head 저장
    df.head(rows).to_csv(output_path, index=False)

    print("✅ Done!")

if __name__ == "__main__":
    main()