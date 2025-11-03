#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' CLI 사용
python inference_top1.py \
  --model_dir ./model_cls10 \
  --input_csv ./data/data_preprocessed_revised.csv \
  --uuid_col JHNT_MBN \
  --text_col SELF_INTRO_CONT \
  --output_csv ./outputs/preds_top1.csv \
  --batch_size 32 \
  --max_length 384
'''
import os
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

# install requirements.txt
REQ_FILE = "requirements.txt"
if os.path.exists(REQ_FILE):
    print(f"[INFO] Installing packages from {REQ_FILE} ...")
    subprocess.run(["pip", "install", "-r", REQ_FILE, "--quiet"], check=False)
else:
    print("[INFO] No requirements.txt found — skipping package installation.")

# label index
IDX2LABEL = {
    0: "협업/팀워크",
    1: "커뮤니케이션/소통",
    2: "문제해결/개선",
    3: "적응력/유연성",
    4: "리더십/주도성",
    5: "성실성/책임감",
    6: "학습의지/자기계발",
    7: "고객지향/서비스마인드",
    8: "시간관리/규율준수",
    9: "직무동기/조직몰입",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Trained model directory (e.g., ./model_cls10)")
    ap.add_argument("--input_csv", default="data/data_preprocessed_revised.csv")
    ap.add_argument("--text_col", default="SELF_INTRO_CONT")
    ap.add_argument("--uuid_col", default="JHNT_MBN")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--output_csv", default="preds_top1.csv")  # 최종: [JHNT_MBN, pred_label]
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ----- model/tokenizer/config  -----
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    cfg = AutoConfig.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_config(cfg)

    # .safetensors or .bin 
    safepath = os.path.join(args.model_dir, "model.safetensors")
    binpath  = os.path.join(args.model_dir, "pytorch_model.bin")
    if os.path.exists(safepath):
        from safetensors.torch import load_file
        state_dict = load_file(safepath)
        print(f"[INFO] Loaded: {safepath}")
    elif os.path.exists(binpath):
        state_dict = torch.load(binpath, map_location="cpu")
        print(f"[INFO] Loaded: {binpath}")
    else:
        raise FileNotFoundError("No model weights found (model.safetensors / pytorch_model.bin)")

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # ----- input (csv) -----
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns or args.uuid_col not in df.columns:
        raise ValueError(f"Columns {args.text_col} and {args.uuid_col} must exist in input CSV")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    uuids = df[args.uuid_col].astype(str).tolist()

    # ----- inference (top-1 index return) -----
    top1_indices = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="[Inference]"):
        batch = texts[i:i + args.batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits.detach().cpu().numpy()
            pred_idx = np.argmax(logits, axis=1)
        top1_indices.append(pred_idx)

    top1 = np.concatenate(top1_indices, axis=0).astype(int)

    # ----- save -----
    out_df = pd.DataFrame({
        args.uuid_col: uuids,
        "pred_label": top1,  # index (0~9)
    })

    # 저장 경로는 실행 위치 기준 (model_dir 안이 아니라 현재 작업 디렉토리)
    out_path = args.output_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved predictions → {out_path}")
    print(out_df.head())

if __name__ == "__main__":
    main()
