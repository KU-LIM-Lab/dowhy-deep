import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm
from pathlib import Path
import logging

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

BATCH_SIZE = 32
MAX_LENGTH = 384

logger = logging.getLogger(__name__)

# Ignore TensorFlow and Abseil logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['ABSL_VERBOSITY'] = '-1'

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

def llm_inference(df: pd.DataFrame, logger: logging.LoggerAdapter,
                  batch_id: int, is_test_mode: bool, output_dir) -> pd.DataFrame:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Device: {device}")

    # ----- model/tokenizer/config  -----
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_config(cfg)

    # .safetensors or .bin 
    safepath = os.path.join(MODEL_DIR, "model.safetensors")
    binpath  = os.path.join(MODEL_DIR, "pytorch_model.bin")
    if os.path.exists(safepath):
        from safetensors.torch import load_file
        state_dict = load_file(safepath)
        logger.info(f"[INFO] Loaded: {safepath}")
    elif os.path.exists(binpath):
        state_dict = torch.load(binpath, map_location="cpu")
        logger.info(f"[INFO] Loaded: {binpath}")
    else:
        raise FileNotFoundError("No model weights found (model.safetensors / pytorch_model.bin)")

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    if "SELF_INTRO_CONT" not in df.columns or "JHNT_MBN" not in df.columns:
        raise ValueError(f"Columns {"SELF_INTRO_CONT"} and {"JHNT_MBN"} must exist in input CSV")

    texts = df["SELF_INTRO_CONT"].astype(str).fillna("").tolist()
    uuids = df["JHNT_MBN"].astype(str).tolist()

    # ----- inference (top-1 index return) -----
    top1_indices = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="[Inference]"):
        batch = texts[i:i + BATCH_SIZE]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits.detach().cpu().numpy()
            pred_idx = np.argmax(logits, axis=1)
        top1_indices.append(pred_idx)

    top1 = np.concatenate(top1_indices, axis=0).astype(int)

    # ----- save -----
    out_df = pd.DataFrame({
        "JHNT_MBN": uuids,
        "SELF_INTRO_CONT_LABEL": top1,  # index (0~9)
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_test_mode:
        out_file_name = "preds_test.csv"
    else:
        out_file_name = f"preds_{batch_id+1}.csv"

    out_path = output_dir / out_file_name

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"[OK] Saved predictions → {out_path}")

    return out_df