import pandas as pd
from pathlib import Path
import sys

def main():
    project_root=Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    intermediate_path = data_dir / "intermediate_preprocessed_df.csv"
    preprocessed_path = data_dir / "preprocessed_df.csv"
    
    if intermediate_path.exists():
        target_path = intermediate_path
        print(f"[INFO] Using file: {intermediate_path.name}")
    
    elif preprocessed_path.exists():
        target_path = preprocessed_path
        print(f"[INFO] Using file: {preprocessed_path.name}")

    else:
        print("[ERROR] Neither CSV file was found in data directory")
        print(f"Checked : {intermediate_path} and {preprocessed_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(target_path)
        print(f"[INFO] Loaded DataFrmae shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        sys.exit(1)
    
    sample_size = 10000
    if len(df) < sample_size:
        print(f"[WARN] Data has only {len(df)} rows. Using entire dataset.")
        df_sample = df.copy()
    else:
        df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled 10000 rows")

    sample_file = data_dir / "intermediate_sample_1000_rows.csv"
    try:
        df_sample.to_csv(sample_file, index=False, encoding='utf-8')
        print(f"[OK] Sample saved to: {sample_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")

if __name__ == "__main__":
    main()