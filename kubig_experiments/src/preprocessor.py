# coding: utf-8
import json, math
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# =========================
# 경로 설정
# =========================
ROOT = Path("data")
RAW_CSV = ROOT / "synthetic_data_raw.csv"

RESUME_DIR   = ROOT / "RESUME_JSON/ver1"
COVER_DIR    = ROOT / "COVERLETTERS_JSON/ver1"
TRAINING_DIR = ROOT / "TRAININGS_JSON"
LICENSE_DIR  = ROOT / "LICENSES_JSON"

OUT_CSV = ROOT / "data_preprocessed.csv"

# =========================
# 유틸
# =========================
def _read_json_safe(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _collect_json(d: Path) -> List[Path]:
    return sorted([p for p in d.rglob("*.json") if p.is_file()]) if d.exists() else []

def _none(x):
    # 결측은 None으로 통일
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x)
    return None if s == "" else x

def _expand_list_columns(df_lists: pd.DataFrame, key_col: str, value_cols: List[str]) -> pd.DataFrame:
    """
    df_lists: 각 셀에 list가 들어있는 DataFrame (key 단위 한 행)
    value_cols: list가 들어있는 컬럼들(=key 제외 모든 컬럼)
    규칙: 소스 전체에서 가장 긴 길이를 K로 잡고, 변수명_1..변수명_K 로 확장. 부족분은 NaN.
    """
    if df_lists.empty:
        return df_lists

    # key별 길이(컬럼별 리스트 길이가 다를 수 있어 방어적으로 계산)
    def _row_max_len(row):
        lens = []
        for c in value_cols:
            v = row[c]
            lens.append(len(v) if isinstance(v, list) else 0)
        return max(lens) if lens else 0

    per_row_max = df_lists.apply(_row_max_len, axis=1)
    K = int(per_row_max.max() if len(per_row_max) else 0)

    if K == 0:
        # 값이 하나도 없으면 key만 반환
        return df_lists[[key_col]].copy()

    out = df_lists[[key_col]].copy()

    for c in value_cols:
        # 각 셀(list)을 길이 K로 패딩
        padded = df_lists[c].apply(
            lambda lst: (lst if isinstance(lst, list) else []) + [np.nan] * (K - (len(lst) if isinstance(lst, list) else 0))
        )
        # 리스트 → 여러 컬럼
        expanded = pd.DataFrame(padded.tolist(), columns=[f"{c}_{i}" for i in range(1, K+1)], index=df_lists.index)
        out = pd.concat([out, expanded], axis=1)

    return out

# =========================
# 1) 이력서 파서 (항목 단위 → key별 리스트 → _1.._K 확장)
#    CSV.JHNT_MBN ↔ JSON.SEEK_CUST_NO
# =========================
RESUME_COLS = [
    "SEEK_CUST_NO","TMPL_SEQNO","RESUME_TITLE","BASIC_RESUME_YN",
    "RESUME_ITEM_CLCD","DS_RESUME_ITEM_CLCD","SEQNO",
    "RESUME_ITEM_1_CD","DS_RESUME_ITEM_1_CD",
    "RESUME_ITEM_2_CD","DS_RESUME_ITEM_2_CD",
    "RESUME_ITEM_3_CD","DS_RESUME_ITEM_3_CD",
    "RESUME_ITEM_4_CD","DS_RESUME_ITEM_4_CD",
    "RESUME_ITEM_1_NM","RESUME_ITEM_2_NM","RESUME_ITEM_3_NM",
    "RESUME_ITEM_1_VAL","HIST_STDT","HIST_ENDT",
    "HIST_ITEM_GRDCD","DS_HIST_ITEM_GRDCD","ETC_ITEM_CONT",
    "CAREER_MMCNT","RESUME_ITEM_5_CD","RESUME_ITEM_5_NM",
    "RESUME_ITEM_6_CD","RESUME_ITEM_6_NM",
    "COMP_NM_OPEN_YN","JSCD","SLRY_AMT","EMMD_OCCP_CL_YEAR"
]

def parse_resume_to_lists(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
        except Exception:
            continue
        seek = str(data.get("SEEK_CUST_NO",""))
        for t in data.get("CONTENTS", []):
            # 대표 이력서만 선택
            if str(t.get("BASIC_RESUME_YN","")) != "Y":
                continue

            base = {
                "SEEK_CUST_NO": seek,
                "TMPL_SEQNO": _none(t.get("TMPL_SEQNO", None)),
                "RESUME_TITLE": _none(t.get("RESUME_TITLE", None)),
                "BASIC_RESUME_YN": _none(t.get("BASIC_RESUME_YN", None)),
            }
            contents = t.get("RESUME_CONTENTS", [])
            if contents:
                for it in contents:
                    row = dict(base)
                    for k in RESUME_COLS:
                        if k in row:  # 상단 4개는 항목 개수만큼 반복됨
                            continue
                        row[k] = _none(it.get(k, None))
                    rows.append(row)
            else:
                # 항목이 없으면 상단만 1건으로
                row = dict(base)
                for k in RESUME_COLS:
                    if k in row: continue
                    row[k] = None
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=RESUME_COLS)

    df_long = pd.DataFrame(rows, columns=RESUME_COLS)

    grouped = df_long.groupby("SEEK_CUST_NO", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in RESUME_COLS if c != "SEEK_CUST_NO"})
    if "SEEK_CUST_NO" not in df_lists.columns:
        df_lists.insert(0, "SEEK_CUST_NO", grouped["SEEK_CUST_NO"].first().values)

    value_cols = [c for c in df_lists.columns if c != "SEEK_CUST_NO"]
    df_expanded = _expand_list_columns(df_lists, "SEEK_CUST_NO", value_cols)
    return df_expanded

    df_long = pd.DataFrame(rows, columns=RESUME_COLS)

    # key별로 모든 컬럼 리스트화
    grouped = df_long.groupby("SEEK_CUST_NO", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in RESUME_COLS if c != "SEEK_CUST_NO"})
    # pandas 버전에 따라 이미 포함돼 있을 수 있음 → 없을 때만 삽입
    if "SEEK_CUST_NO" not in df_lists.columns:
        df_lists.insert(0, "SEEK_CUST_NO", grouped["SEEK_CUST_NO"].first().values)

    # 리스트 → _1.._K 확장
    value_cols = [c for c in df_lists.columns if c != "SEEK_CUST_NO"]
    df_expanded = _expand_list_columns(df_lists, "SEEK_CUST_NO", value_cols)
    return df_expanded

# =========================
# 2) 자기소개서 파서
#    CSV.JHNT_MBN ↔ JSON.SEEK_CUST_NO
# =========================
COVER_COLS = [
    "SEEK_CUST_NO","SFID_NO","BASS_SFID_YN",
    "SFID_IEM_SN","SFID_SJNM","SFID_IEM_SECD","DS_SFID_IEM_SECD","SFID_IEM_SJNM","SELF_INTRO_CONT"
]

def parse_cover_to_lists(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
        except Exception:
            continue
        seek = str(data.get("SEEK_CUST_NO",""))
        for s in data.get("CONTENTS", []):
            sfid_no = _none(s.get("SFID_NO", None))
            bass    = _none(s.get("BASS_SFID_YN", None))
            for it in s.get("COVERLETER_CONTENTS", []):  # 오타 그대로
                rows.append({
                    "SEEK_CUST_NO": seek,
                    "SFID_NO": sfid_no,
                    "BASS_SFID_YN": bass,
                    "SFID_IEM_SN": _none(it.get("SFID_IEM_SN", None)),
                    "SFID_SJNM": _none(it.get("SFID_SJNM", None)),
                    "SFID_IEM_SECD": _none(it.get("SFID_IEM_SECD", None)),
                    "DS_SFID_IEM_SECD": _none(it.get("DS_SFID_IEM_SECD", None)),
                    "SFID_IEM_SJNM": _none(it.get("SFID_IEM_SJNM", None)),
                    "SELF_INTRO_CONT": _none(it.get("SELF_INTRO_CONT", None)),
                })
    if not rows:
        return pd.DataFrame(columns=COVER_COLS)

    df_long = pd.DataFrame(rows, columns=COVER_COLS)
    grouped = df_long.groupby("SEEK_CUST_NO", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in COVER_COLS if c != "SEEK_CUST_NO"})
    if "SEEK_CUST_NO" not in df_lists.columns:
        df_lists.insert(0, "SEEK_CUST_NO", grouped["SEEK_CUST_NO"].first().values)

    value_cols = [c for c in df_lists.columns if c != "SEEK_CUST_NO"]
    df_expanded = _expand_list_columns(df_lists, "SEEK_CUST_NO", value_cols)
    return df_expanded

# =========================
# 3) 직업훈련 파서
#    CSV.JHNT_CTN ↔ JSON.JHNT_CTN (JHCR_DE 상단, SORTN_SN → SORT_SN)
# =========================
TRAINING_BASE_COLS = [
    "CLOS_YM","JHNT_CTN","JHCR_DE","CRSE_ID","TGCR_TME",
    "TRNG_CRSN","TRNG_BGDE","TRNG_ENDE","TRNG_JSCD","KECO_CD","SORT_SN","ETL_DT",
]

def parse_train_to_lists(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
        except Exception:
            continue
        clos = _none(data.get("CLOS_YM", None))
        jhnt = _none(data.get("JHNT_CTN", None))
        jhcr_de = _none(data.get("JHCR_DE", None))
        contents = data.get("CONTENTS", [])
        if contents:
            for it in contents:
                sort_sn = it.get("SORT_SN", it.get("SORTN_SN", None))
                rows.append({
                    "CLOS_YM": clos, "JHNT_CTN": jhnt, "JHCR_DE": jhcr_de,
                    "CRSE_ID": _none(it.get("CRSE_ID", None)),
                    "TGCR_TME": _none(it.get("TGCR_TME", None)),
                    "TRNG_CRSN": _none(it.get("TRNG_CRSN", None)),
                    "TRNG_BGDE": _none(it.get("TRNG_BGDE", None)),
                    "TRNG_ENDE": _none(it.get("TRNG_ENDE", None)),
                    "TRNG_JSCD": _none(it.get("TRNG_JSCD", None)),
                    "KECO_CD": _none(it.get("KECO_CD", None)),
                    "SORT_SN": _none(sort_sn),
                    "ETL_DT": _none(it.get("ETL_DT", None)),
                })
        else:
            rows.append({
                "CLOS_YM": clos, "JHNT_CTN": jhnt, "JHCR_DE": jhcr_de,
                "CRSE_ID": None,"TGCR_TME": None,"TRNG_CRSN": None,"TRNG_BGDE": None,
                "TRNG_ENDE": None,"TRNG_JSCD": None,"KECO_CD": None,"SORT_SN": None,"ETL_DT": None,
            })

    if not rows:
        return pd.DataFrame(columns=TRAINING_BASE_COLS)

    df_long = pd.DataFrame(rows, columns=TRAINING_BASE_COLS)
    grouped = df_long.groupby("JHNT_CTN", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in TRAINING_BASE_COLS if c != "JHNT_CTN"})
    if "JHNT_CTN" not in df_lists.columns:
        df_lists.insert(0, "JHNT_CTN", grouped["JHNT_CTN"].first().values)

    value_cols = [c for c in df_lists.columns if c != "JHNT_CTN"]
    df_expanded = _expand_list_columns(df_lists, "JHNT_CTN", value_cols)
    return df_expanded

# =========================
# 4) 자격증 파서
#    CSV.JHNT_CTN ↔ JSON.JHNT_CTN
# =========================
LICENSE_BASE_COLS = ["CLOS_YM","JHNT_CTN","CRQF_CD","QULF_ITNM","QULF_LCNS_LCFN","ETL_DT"]

def parse_license_to_lists(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
        except Exception:
            continue
        clos = _none(data.get("CLOS_YM", None))
        jhnt = _none(data.get("JHNT_CTN", None))
        contents = data.get("CONTENTS", [])
        if contents:
            for it in contents:
                rows.append({
                    "CLOS_YM": clos, "JHNT_CTN": jhnt,
                    "CRQF_CD": _none(it.get("CRQF_CD", None)),
                    "QULF_ITNM": _none(it.get("QULF_ITNM", None)),
                    "QULF_LCNS_LCFN": _none(it.get("QULF_LCNS_LCFN", None)),
                    "ETL_DT": _none(it.get("ETL_DT", None)),
                })
        else:
            rows.append({
                "CLOS_YM": clos, "JHNT_CTN": jhnt,
                "CRQF_CD": None,"QULF_ITNM": None,"QULF_LCNS_LCFN": None,"ETL_DT": None,
            })

    if not rows:
        return pd.DataFrame(columns=LICENSE_BASE_COLS)

    df_long = pd.DataFrame(rows, columns=LICENSE_BASE_COLS)
    grouped = df_long.groupby("JHNT_CTN", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in LICENSE_BASE_COLS if c != "JHNT_CTN"})
    if "JHNT_CTN" not in df_lists.columns:
        df_lists.insert(0, "JHNT_CTN", grouped["JHNT_CTN"].first().values)

    value_cols = [c for c in df_lists.columns if c != "JHNT_CTN"]
    df_expanded = _expand_list_columns(df_lists, "JHNT_CTN", value_cols)
    return df_expanded

# =========================
# 5) 조립: CSV LEFT 기준
# =========================
def build_pipeline_wide() -> pd.DataFrame:
    base = pd.read_csv(RAW_CSV, encoding="utf-8")
    for k in ["JHNT_MBN","JHNT_CTN"]:
        if k in base.columns:
            base[k] = base[k].astype(str)

    resume_df   = parse_resume_to_lists(_collect_json(RESUME_DIR))     # key: SEEK_CUST_NO
    cover_df    = parse_cover_to_lists(_collect_json(COVER_DIR))       # key: SEEK_CUST_NO
    training_df = parse_train_to_lists(_collect_json(TRAINING_DIR))    # key: JHNT_CTN
    license_df  = parse_license_to_lists(_collect_json(LICENSE_DIR))   # key: JHNT_CTN

    out = base.copy()

    # 이력서: base.JHNT_MBN ↔ resume.SEEK_CUST_NO
    if not resume_df.empty and "JHNT_MBN" in out.columns:
        out = out.merge(resume_df, left_on="JHNT_MBN", right_on="SEEK_CUST_NO", how="left")
        out = out.drop(columns=["SEEK_CUST_NO"], errors="ignore")

    # 자기소개서: base.JHNT_MBN ↔ cover.SEEK_CUST_NO
    if not cover_df.empty and "JHNT_MBN" in out.columns:
        # 충돌 방지를 위해 suffix 부여
        out = out.merge(cover_df, left_on="JHNT_MBN", right_on="SEEK_CUST_NO", how="left", suffixes=("", "_cover"))
        out = out.drop(columns=["SEEK_CUST_NO"], errors="ignore")

    # 직업훈련: base.JHNT_CTN ↔ training.JHNT_CTN
    if not training_df.empty and "JHNT_CTN" in out.columns:
        out = out.merge(training_df, on="JHNT_CTN", how="left")

    # 자격증: base.JHNT_CTN ↔ license.JHNT_CTN
    if not license_df.empty and "JHNT_CTN" in out.columns:
        out = out.merge(license_df, on="JHNT_CTN", how="left", suffixes=("", "_license"))

    return out

# =========================
# 6) 후처리 (옵션)
#    - (1) 예/아니오/필요/불필요 → 1/0
#    - (2) 날짜 → 앵커(JHCR_DE) 기준 일수
# =========================
def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    # ---- (1) 바이너리 매핑 ----
    bin_map = {"예":1, "아니오":0, "아니요":0, "필요":1, "불필요":0}
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(bin_map)

    # ---- (2) 날짜 차이 계산 (앵커: JHCR_DE) ----
    if "JHCR_DE" in df.columns:
        anchor = pd.to_datetime(df["JHCR_DE"], errors="coerce")
        date_cols = [c for c in df.columns if any(x in c.upper() for x in ["DE","DT","DATE","BGDE","ENDE","STDT","ENDT"])]
        for col in date_cols:
            if col == "JHCR_DE":
                continue
            vals = pd.to_datetime(df[col], errors="coerce")
            diff = (vals - anchor).dt.days
            df[col] = diff.abs()   # NaN 그대로 보존

    # ---- (3) 모든 값이 결측인 컬럼 제거 ----
    df = df.dropna(axis=1, how="all")

    return df

# =========================
# 실행
# =========================
if __name__ == "__main__":
    final_df = build_pipeline_wide()
    final_df = postprocess(final_df)
    final_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved -> {OUT_CSV}  shape={final_df.shape}")