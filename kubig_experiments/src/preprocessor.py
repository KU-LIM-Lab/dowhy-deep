# coding: utf-8
import json, math
from pathlib import Path
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
import logging

# =========================
# 로거 설정
# =========================
logger = logging.getLogger(__name__)

# =========================
# 경로 설정
# =========================
ROOT = Path(__file__).resolve().parent.parent / "data"

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
    """주어진 디렉토리와 그 하위 디렉토리에서 모든 JSON 파일을 재귀적으로 수집 (rglob 사용)"""
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
# 1) 이력서 파서 (버전별)
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

def _process_resume_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """공통 후처리 로직 (Long 포맷에서 Wide 포맷으로 변환)"""
    if not rows:
        return pd.DataFrame(columns=RESUME_COLS)

    df_long = pd.DataFrame(rows, columns=RESUME_COLS)
    grouped = df_long.groupby("SEEK_CUST_NO", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in RESUME_COLS if c != "SEEK_CUST_NO"})
    if "SEEK_CUST_NO" not in df_lists.columns:
        df_lists.insert(0, "SEEK_CUST_NO", grouped["SEEK_CUST_NO"].first().values)

    value_cols = [c for c in df_lists.columns if c != "SEEK_CUST_NO"]
    return _expand_list_columns(df_lists, "SEEK_CUST_NO", value_cols)


def parse_resume_to_lists_ver1(paths: List[Path]) -> pd.DataFrame:
    """RESUME_JSON/ver1, ver3 파서: CONTENTS/t/RESUME_CONTENTS/it 구조"""
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
            seek = str(data.get("SEEK_CUST_NO", ""))

            for t in data.get("CONTENTS", []):
                if str(t.get("BASIC_RESUME_YN", "")) != "Y": continue

                base = {
                    "SEEK_CUST_NO": seek, "TMPL_SEQNO": _none(t.get("TMPL_SEQNO", None)),
                    "RESUME_TITLE": _none(t.get("RESUME_TITLE", None)),
                    "BASIC_RESUME_YN": _none(t.get("BASIC_RESUME_YN", None)),
                }
                # ver1/ver3의 핵심 특징: 항목들이 RESUME_CONTENTS 리스트 안에 있음
                contents = t.get("RESUME_CONTENTS", [])
                
                if isinstance(contents, list) and len(contents) > 0:
                    for it in contents:
                        row = dict(base)
                        for k in RESUME_COLS:
                            if k in row: continue
                            row[k] = _none(it.get(k, None))
                        rows.append(row)
                else:
                    # 항목이 없으면 상단만 1건으로
                    row = dict(base)
                    for k in RESUME_COLS:
                        if k in row: continue
                        row[k] = None
                    rows.append(row)
        except Exception as e:
            logger.debug(f"Error parsing resume (ver1/3) file {p}: {e}")
            continue
    
    return _process_resume_rows(rows)


def parse_resume_to_lists_ver2(paths: List[Path]) -> pd.DataFrame:
    """RESUME_JSON/ver2 파서: CONTENTS/t에 항목 키들이 평탄하게 존재"""
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
            seek = str(data.get("SEEK_CUST_NO", ""))

            for t in data.get("CONTENTS", []):
                # ver2의 핵심 특징: 항목들이 t 레벨에 평탄하게 존재함
                if str(t.get("BASIC_RESUME_YN", "")) != "Y": continue

                base = {
                    "SEEK_CUST_NO": seek, "TMPL_SEQNO": _none(t.get("TMPL_SEQNO", None)),
                    "RESUME_TITLE": _none(t.get("RESUME_TITLE", None)),
                    "BASIC_RESUME_YN": _none(t.get("BASIC_RESUME_YN", None)),
                }
                
                row = dict(base)
                for k in RESUME_COLS:
                    if k in row: continue
                    # t에서 직접 값을 가져옴
                    row[k] = _none(t.get(k, None))
                rows.append(row)
        except Exception as e:
            logger.debug(f"Error parsing resume (ver2) file {p}: {e}")
            continue

    return _process_resume_rows(rows)


def get_resume_parser(all_paths: List[Path]) -> Callable[[List[Path]], pd.DataFrame]:
    """
    이력서 JSON 경로를 스캔하여 유효한 파서 함수를 반환합니다.
    (RESUME_CONTENTS 키의 존재 여부로 ver1/ver3 vs ver2를 판별)
    """
    if not all_paths:
        logger.warning("No resume JSON files found. Returning empty parser.")
        return lambda paths: pd.DataFrame(columns=RESUME_COLS)

    # 샘플 파일 로드
    sample_path = all_paths[0]
    try:
        data = _read_json_safe(sample_path)
        contents = data.get("CONTENTS", [])
        
        # 대표 템플릿 항목(contents의 첫 번째 항목)의 구조를 확인
        if contents and isinstance(contents, list) and len(contents) > 0:
            first_content = contents[0]
            
            # Ver1/Ver3 구조 판별: RESUME_CONTENTS 키의 존재 여부
            if "RESUME_CONTENTS" in first_content:
                logger.info(f"Resume JSON structure detected as: ver1/ver3 (based on 'RESUME_CONTENTS' key).")
                # ver1과 ver3는 로직이 동일하므로 ver1 파서 사용
                return lambda paths: parse_resume_to_lists_ver1(all_paths)
            else:
                # RESUME_CONTENTS가 없으면 Ver2 구조로 간주
                # Ver2 JSON 파일이 Ver1 JSON처럼 보이도록 (RESUME_CONTENTS를 None으로) 만들 수도 있으나,
                # JSON 구조에 따라 Ver1이 Ver2 데이터를 읽어도 에러가 발생하지 않아 Ver1이 먼저 선택되는 문제를
                # 회피하기 위해, RESUME_CONTENTS가 없으면 Ver2로 명확히 분리합니다.
                logger.info(f"Resume JSON structure detected as: ver2 (based on absence of 'RESUME_CONTENTS' key).")
                return lambda paths: parse_resume_to_lists_ver2(all_paths)

        else:
            logger.warning("Sample resume file is valid but 'CONTENTS' field is empty or invalid list.")
            return lambda paths: pd.DataFrame(columns=RESUME_COLS)

    except Exception as e:
        logger.error(f"Error during resume parser auto-detection for file {sample_path}: {e}")
        # 예외 발생 시, Ver1을 먼저 시도하여 fallback
        logger.warning("Fallback: Attempting to use Ver1 parser due to detection error.")
        return lambda paths: parse_resume_to_lists_ver1(all_paths)


# =========================
# 2) 자기소개서 파서 (버전별)
# =========================
COVER_COLS = [
    "SEEK_CUST_NO","SFID_NO","BASS_SFID_YN",
    "SFID_IEM_SN","SFID_SJNM","SFID_IEM_SECD","DS_SFID_IEM_SECD","SFID_IEM_SJNM","SELF_INTRO_CONT"
]

def _process_cover_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """공통 후처리 로직 (Long 포맷에서 Wide 포맷으로 변환)"""
    if not rows:
        return pd.DataFrame(columns=COVER_COLS)

    df_long = pd.DataFrame(rows, columns=COVER_COLS)
    grouped = df_long.groupby("SEEK_CUST_NO", as_index=False)
    df_lists = grouped.agg({c: (lambda s: list(s)) for c in COVER_COLS if c != "SEEK_CUST_NO"})
    if "SEEK_CUST_NO" not in df_lists.columns:
        df_lists.insert(0, "SEEK_CUST_NO", grouped["SEEK_CUST_NO"].first().values)

    value_cols = [c for c in df_lists.columns if c != "SEEK_CUST_NO"]
    return _expand_list_columns(df_lists, "SEEK_CUST_NO", value_cols)


def parse_cover_to_lists_ver1(paths: List[Path]) -> pd.DataFrame:
    """COVERLETTERS_JSON/ver1 파서: CONTENTS/s/COVERLETER_CONTENTS/it 구조 (오타 포함)"""
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
            seek = str(data.get("SEEK_CUST_NO", ""))
            for s in data.get("CONTENTS", []):
                sfid_no = _none(s.get("SFID_NO", None))
                bass    = _none(s.get("BASS_SFID_YN", None))
                # ver1의 핵심 특징: 항목들이 'COVERLETER_CONTENTS' 리스트 안에 있음 (오타 주의)
                for it in s.get("COVERLETER_CONTENTS", []): 
                    rows.append({
                        "SEEK_CUST_NO": seek, "SFID_NO": sfid_no, "BASS_SFID_YN": bass,
                        "SFID_IEM_SN": _none(it.get("SFID_IEM_SN", None)),
                        "SFID_SJNM": _none(it.get("SFID_SJNM", None)),
                        "SFID_IEM_SECD": _none(it.get("SFID_IEM_SECD", None)),
                        "DS_SFID_IEM_SECD": _none(it.get("DS_SFID_IEM_SECD", None)),
                        "SFID_IEM_SJNM": _none(it.get("SFID_IEM_SJNM", None)),
                        "SELF_INTRO_CONT": _none(it.get("SELF_INTRO_CONT", None)),
                    })
        except Exception as e:
            logger.debug(f"Error parsing cover (ver1) file {p}: {e}")
            continue
    return _process_cover_rows(rows)


def parse_cover_to_lists_ver2(paths: List[Path]) -> pd.DataFrame:
    """COVERLETTERS_JSON/ver2 파서: CONTENTS/s에 항목 키들이 평탄하게 존재"""
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
            seek = str(data.get("SEEK_CUST_NO", ""))
            for s in data.get("CONTENTS", []):
                # ver2의 핵심 특징: 항목 키들이 s 레벨에 평탄하게 존재함
                sfid_no = _none(s.get("SFID_NO", None))
                bass    = _none(s.get("BASS_SFID_YN", None))
                rows.append({
                    "SEEK_CUST_NO": seek, "SFID_NO": sfid_no, "BASS_SFID_YN": bass,
                    "SFID_IEM_SN": _none(s.get("SFID_IEM_SN", None)),
                    "SFID_SJNM": _none(s.get("SFID_SJNM", None)),
                    "SFID_IEM_SECD": _none(s.get("SFID_IEM_SECD", None)),
                    "DS_SFID_IEM_SECD": _none(s.get("DS_SFID_IEM_SECD", None)),
                    "SFID_IEM_SJNM": _none(s.get("SFID_IEM_SJNM", None)),
                    "SELF_INTRO_CONT": _none(s.get("SELF_INTRO_CONT", None)),
                })
        except Exception as e:
            logger.debug(f"Error parsing cover (ver2) file {p}: {e}")
            continue
    return _process_cover_rows(rows)


def get_cover_parser(all_paths: List[Path]) -> Callable[[List[Path]], pd.DataFrame]:
    """
    자기소개서 JSON 경로를 스캔하여 유효한 파서 함수를 반환합니다.
    (COVERLETER_CONTENTS 키의 존재 여부로 ver1 vs ver2를 판별)
    """
    if not all_paths:
        logger.warning("No cover letter JSON files found. Returning empty parser.")
        return lambda paths: pd.DataFrame(columns=COVER_COLS)

    sample_path = all_paths[0]
    try:
        data = _read_json_safe(sample_path)
        contents = data.get("CONTENTS", [])
        
        if contents and isinstance(contents, list) and len(contents) > 0:
            first_content = contents[0]
            
            # Ver1 구조 판별: COVERLETER_CONTENTS 키의 존재 여부
            if "COVERLETER_CONTENTS" in first_content:
                logger.info(f"Cover letter JSON structure detected as: ver1 (based on 'COVERLETER_CONTENTS' key).")
                return lambda paths: parse_cover_to_lists_ver1(all_paths)
            else:
                # COVERLETER_CONTENTS가 없으면 Ver2 구조로 간주
                logger.info(f"Cover letter JSON structure detected as: ver2 (based on absence of 'COVERLETER_CONTENTS' key).")
                return lambda paths: parse_cover_to_lists_ver2(all_paths)
        else:
            logger.warning("Sample cover letter file is valid but 'CONTENTS' field is empty or invalid list.")
            return lambda paths: pd.DataFrame(columns=COVER_COLS)

    except Exception as e:
        logger.error(f"Error during cover parser auto-detection for file {sample_path}: {e}")
        # 예외 발생 시, Ver1을 먼저 시도하여 fallback
        logger.warning("Fallback: Attempting to use Ver1 parser due to detection error.")
        return lambda paths: parse_cover_to_lists_ver1(all_paths)


# =========================
# 3) 직업훈련 파서 (공통)
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
        except Exception as e:
            logger.debug(f"Error parsing training file {p}: {e}")
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
# 4) 자격증 파서 (공통)
# =========================
LICENSE_BASE_COLS = ["CLOS_YM","JHNT_CTN","CRQF_CD","QULF_ITNM","QULF_LCNS_LCFN","ETL_DT"]

def parse_license_to_lists(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            data = _read_json_safe(p)
        except Exception as e:
            logger.debug(f"Error parsing license file {p}: {e}")
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
def build_pipeline_wide(logger: logging.LoggerAdapter) -> pd.DataFrame:
    logger.info("Reading raw CSV data.")
    base = pd.read_csv(RAW_CSV, encoding="utf-8")
    for k in ["JHNT_MBN","JHNT_CTN"]:
        if k in base.columns:
            base[k] = base[k].astype(str)

    # 1. 이력서 데이터 처리
    logger.info(f"Collecting resume JSON files from {RESUME_DIR}.")
    all_resume_paths = _collect_json(RESUME_DIR)
    resume_parser_func = get_resume_parser(all_resume_paths)
    logger.info("Starting resume data parsing.")
    resume_df = resume_parser_func(all_resume_paths) 

    # 2. 자기소개서 데이터 처리
    logger.info(f"Collecting cover letter JSON files from {COVER_DIR}.")
    all_cover_paths = _collect_json(COVER_DIR)
    cover_parser_func = get_cover_parser(all_cover_paths)
    logger.info("Starting cover letter data parsing.")
    cover_df = cover_parser_func(all_cover_paths)

    # 3. 직업훈련 데이터 처리
    logger.info(f"Collecting and parsing training data from {TRAINING_DIR}.")
    training_df = parse_train_to_lists(_collect_json(TRAINING_DIR))

    # 4. 자격증 데이터 처리
    logger.info(f"Collecting and parsing license data from {LICENSE_DIR}.")
    license_df  = parse_license_to_lists(_collect_json(LICENSE_DIR))

    out = base.copy()
    logger.info(f"Base DataFrame shape: {out.shape}")

    # 데이터 병합
    if not resume_df.empty and "JHNT_MBN" in out.columns:
        out = out.merge(resume_df, left_on="JHNT_MBN", right_on="SEEK_CUST_NO", how="left")
        out = out.drop(columns=["SEEK_CUST_NO"], errors="ignore")
        logger.info(f"Merged resume data. Current shape: {out.shape}")

    if not cover_df.empty and "JHNT_MBN" in out.columns:
        out = out.merge(cover_df, left_on="JHNT_MBN", right_on="SEEK_CUST_NO", how="left", suffixes=("", "_cover"))
        out = out.drop(columns=["SEEK_CUST_NO"], errors="ignore")
        logger.info(f"Merged cover letter data. Current shape: {out.shape}")

    if not training_df.empty and "JHNT_CTN" in out.columns:
        out = out.merge(training_df, on="JHNT_CTN", how="left")
        logger.info(f"Merged training data. Current shape: {out.shape}")

    if not license_df.empty and "JHNT_CTN" in out.columns:
        out = out.merge(license_df, on="JHNT_CTN", how="left", suffixes=("", "_license"))
        logger.info(f"Merged license data. Current shape: {out.shape}")

    return out

# =========================
# 6) 후처리 (옵션)
# =========================
def postprocess(df: pd.DataFrame, logger: logging.LoggerAdapter) -> pd.DataFrame:
    logger.info("Starting postprocessing: Binary mapping and date calculation.")
    
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
            df[col] = diff.abs()

    # ---- (3) 모든 값이 결측인 컬럼 제거 ----
    original_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped_cols = original_cols - df.shape[1]
    if dropped_cols > 0:
        logger.info(f"Dropped {dropped_cols} columns that were entirely missing values.")
    
    logger.info("Postprocessing complete.")
    return df