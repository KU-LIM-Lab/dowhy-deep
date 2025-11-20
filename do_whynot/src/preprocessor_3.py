# coding: utf-8
import json, math
from pathlib import Path
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
import logging
import json

from do_whynot.config import RAW_CSV, TOTAL_RESUME_JSON, TOTAL_COVER_JSON, TOTAL_TRAINING_JSON, TOTAL_LICENSE_JSON, EXCLUDE_COLS

logger = logging.getLogger(__name__)


def _read_json_safe(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

# def _collect_json(d: Path) -> List[Path]:
#     """주어진 디렉토리와 그 하위 디렉토리에서 모든 JSON 파일을 재귀적으로 수집 (rglob 사용)"""
#     return sorted([p for p in d.rglob("*.json") if p.is_file()]) if d.exists() else []

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
    리스트 컬럼을 K개(_1.._K)로 확장하지 않고,
    각 셀의 '첫 번째 값'만 남겨 '원래 컬럼명'으로 반환한다.
      - 셀 값이 리스트면: 길이>0이면 lst[0], 비어있으면 NaN
      - 리스트가 아니면: 원래 값을 그대로 사용
    """
    if df_lists.empty:
        return df_lists

    # key 컬럼만 먼저 복사
    out = df_lists[[key_col]].copy() if key_col in df_lists.columns else pd.DataFrame()

    for c in value_cols:
        col = df_lists[c].apply(
            lambda v: (v[0] if isinstance(v, list) and len(v) > 0
                       else (np.nan if isinstance(v, list) else v))
        )
        # 접미사 없이 '원래 컬럼명'으로 저장
        out[c] = col

    return out


# =========================
# 1) 이력서 파서 (버전별)
# =========================
RESUME_COLS = [
    "SEEK_CUST_NO","TEMPL_SEQNO","RESUME_TITLE","BASIC_RESUME_YN",
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


def parse_resume_to_lists_ver1(all_resume_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """RESUME_JSON/ver1, ver3 파서: CONTENTS/t/RESUME_CONTENTS/it 구조"""
    rows = []
    # Note: preprocessor.py에서는 paths를 받지만, 여기서는 이미 로드된 all_resume_data를 받음
    for data in all_resume_data: 
        try:
            seek = str(data.get("SEEK_CUST_NO", ""))

            for t in data.get("CONTENTS", []):
                if str(t.get("BASIC_RESUME_YN", "")) != "Y": continue

                base = {
                    "SEEK_CUST_NO": seek, "TEMPL_SEQNO": _none(t.get("TEMPL_SEQNO", None)),
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
            # 통합 JSON이므로 p 대신 seek_cust_no를 사용
            logger.debug(f"Error parsing resume (ver1/3) for SEEK_CUST_NO {seek}: {e}")
            continue
    
    return _process_resume_rows(rows)


def parse_resume_to_lists_ver2(all_resume_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """RESUME_JSON/ver2 파서: CONTENTS/t에 항목 키들이 평탄하게 존재"""
    rows = []
    # Note: preprocessor.py에서는 paths를 받지만, 여기서는 이미 로드된 all_resume_data를 받음
    for data in all_resume_data:
        try:
            seek = str(data.get("SEEK_CUST_NO", ""))

            for t in data.get("CONTENTS", []):
                # ver2의 핵심 특징: 항목들이 t 레벨에 평탄하게 존재함
                if str(t.get("BASIC_RESUME_YN", "")) != "Y": continue

                base = {
                    "SEEK_CUST_NO": seek, "TEMPL_SEQNO": _none(t.get("TEMPL_SEQNO", None)),
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
            # 통합 JSON이므로 p 대신 seek_cust_no를 사용
            logger.debug(f"Error parsing resume (ver2) for SEEK_CUST_NO {seek}: {e}")
            continue

    return _process_resume_rows(rows)


def get_resume_parser(total_json_path: Path) -> tuple[Callable[[List[Dict[str, Any]]], pd.DataFrame], List[Dict[str, Any]]]:
    """
    이력서 통합 JSON 파일을 로드하여 유효한 파서 함수와 로드된 데이터를 반환합니다.
    (반환 타입 수정: (파서 함수, 로드된 전체 데이터) 튜플)
    """
    if not total_json_path.exists():
        logger.warning(f"Resume JSON file not found at {total_json_path}. Returning empty parser and data.")
        return lambda data: pd.DataFrame(columns=RESUME_COLS), []

    try:
        # 통합 JSON 파일 전체를 로드 (List[Dict] 형태)
        all_resume_data = _read_json_safe(total_json_path)
    except Exception as e:
        logger.error(f"Error loading total resume JSON file {total_json_path}: {e}")
        logger.warning("Fallback: Attempting to use Ver1 parser due to loading error.")
        # 로드 실패 시 빈 리스트와 Ver1 파서 함수를 반환
        return parse_resume_to_lists_ver1, []


    if not all_resume_data or not isinstance(all_resume_data, list):
        logger.warning("Total resume file is valid but data is empty or invalid list.")
        return lambda data: pd.DataFrame(columns=RESUME_COLS), []

    # 대표 템플릿 항목(contents의 첫 번째 항목)의 구조를 확인
    first_data = all_resume_data[0]
    contents = first_data.get("CONTENTS", [])

    parser_func = parse_resume_to_lists_ver1 # 기본값

    if contents and isinstance(contents, list) and len(contents) > 0:
        first_content = contents[0]
        
        if "RESUME_CONTENTS" in first_content:
            logger.info(f"Resume JSON structure detected as: ver1/ver3 (based on 'RESUME_CONTENTS' key).")
            parser_func = parse_resume_to_lists_ver1
        else:
            logger.info(f"Resume JSON structure detected as: ver2 (based on absence of 'RESUME_CONTENTS' key).")
            parser_func = parse_resume_to_lists_ver2
    else:
        logger.warning("Sample resume data is valid but 'CONTENTS' field is empty or invalid list. Using Ver1 parser as fallback.")
    
    # 파서 함수와 로드된 전체 데이터를 반환
    return parser_func, all_resume_data


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


def parse_cover_to_lists_ver1(all_cover_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """COVERLETTERS_JSON/ver1 파서: CONTENTS/s/COVERLETTER_CONTENTS/it 구조 (오타 포함)"""
    rows = []
    # Note: preprocessor.py에서는 paths를 받지만, 여기서는 이미 로드된 all_cover_data를 받음
    for data in all_cover_data:
        try:
            seek = str(data.get("SEEK_CUST_NO", ""))
            for s in data.get("CONTENTS", []):
                sfid_no = _none(s.get("SFID_NO", None))
                bass    = _none(s.get("BASS_SFID_YN", None))
                # ver1의 핵심 특징: 항목들이 'COVERLETTER_CONTENTS' 리스트 안에 있음 (오타 주의)
                for it in s.get("COVERLETTER_CONTENTS", []): 
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
            # 통합 JSON이므로 p 대신 seek_cust_no를 사용
            logger.debug(f"Error parsing cover (ver1) for SEEK_CUST_NO {seek}: {e}")
            continue
    return _process_cover_rows(rows)


def parse_cover_to_lists_ver2(all_cover_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """COVERLETTERS_JSON/ver2 파서: CONTENTS/s에 항목 키들이 평탄하게 존재"""
    rows = []
    # Note: preprocessor.py에서는 paths를 받지만, 여기서는 이미 로드된 all_cover_data를 받음
    for data in all_cover_data:
        try:
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
            # 통합 JSON이므로 p 대신 seek_cust_no를 사용
            logger.debug(f"Error parsing cover (ver2) for SEEK_CUST_NO {seek}: {e}")
            continue
    return _process_cover_rows(rows)


def get_cover_parser(total_json_path: Path) -> tuple[Callable[[List[Dict[str, Any]]], pd.DataFrame], List[Dict[str, Any]]]:
    """
    자기소개서 통합 JSON 파일을 로드하여 유효한 파서 함수와 로드된 데이터를 반환합니다.
    (반환 타입 수정: (파서 함수, 로드된 전체 데이터) 튜플)
    """
    if not total_json_path.exists():
        logger.warning(f"Cover letter JSON file not found at {total_json_path}. Returning empty parser and data.")
        return lambda data: pd.DataFrame(columns=COVER_COLS), []

    try:
        # 통합 JSON 파일 전체를 로드 (List[Dict] 형태)
        all_cover_data = _read_json_safe(total_json_path)
    except Exception as e:
        logger.error(f"Error loading total cover JSON file {total_json_path}: {e}")
        logger.warning("Fallback: Attempting to use Ver1 parser due to loading error.")
        return parse_cover_to_lists_ver1, []


    if not all_cover_data or not isinstance(all_cover_data, list):
        logger.warning("Total cover file is valid but data is empty or invalid list.")
        return lambda data: pd.DataFrame(columns=COVER_COLS), []

    # 대표 템플릿 항목(contents의 첫 번째 항목)의 구조를 확인
    first_data = all_cover_data[0]
    contents = first_data.get("CONTENTS", [])
    
    parser_func = parse_cover_to_lists_ver1 # 기본값

    if contents and isinstance(contents, list) and len(contents) > 0:
        first_content = contents[0]
        
        if "COVERLETTER_CONTENTS" in first_content:
            logger.info(f"Cover letter JSON structure detected as: ver1 (based on 'COVERLETTER_CONTENTS' key).")
            parser_func = parse_cover_to_lists_ver1
        else:
            logger.info(f"Cover letter JSON structure detected as: ver2 (based on absence of 'COVERLETTER_CONTENTS' key).")
            parser_func = parse_cover_to_lists_ver2
    else:
        logger.warning("Sample cover letter data is valid but 'CONTENTS' field is empty or invalid list. Using Ver1 parser as fallback.")
        
    # 파서 함수와 로드된 전체 데이터를 반환
    return parser_func, all_cover_data


# =========================
# 3) 직업훈련 파서 (공통)
# =========================
TRAINING_BASE_COLS = [
    "CLOS_YM","JHNT_CTN","JHCR_DE","CRSE_ID","TGCR_TME",
    "TRNG_CRSN","TRNG_BGDE","TRNG_ENDE","TRNG_JSCD","KECO_CD","SORT_SN","ETL_DT",
]

def parse_train_to_lists(all_train_data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    # Note: preprocessor.py에서는 paths를 받지만, 여기서는 이미 로드된 all_train_data를 받음
    for data in all_train_data:
        jhnt = None # 오류 로깅을 위해 try 바깥에서 정의
        try:
            clos = _none(data.get("CLOS_YM", None))
            jhnt = _none(data.get("JHNT_CTN", None))
            jhcr_de = _none(data.get("JHCR_DE", None))
            contents = data.get("TRAININGS_JSON", [])
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
        except Exception as e:
            # 통합 JSON이므로 p 대신 JHNT_CTN을 사용
            logger.debug(f"Error parsing training data for JHNT_CTN {jhnt}: {e}")
            continue

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

def parse_license_to_lists(all_license_data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    # Note: preprocessor.py에서는 paths를 받지만, 여기서는 이미 로드된 all_license_data를 받음
    for data in all_license_data:
        jhnt = None # 오류 로깅을 위해 try 바깥에서 정의
        try:
            clos = _none(data.get("CLOS_YM", None))
            jhnt = _none(data.get("JHNT_CTN", None))
            contents = data.get("LICENSES_JSON", [])
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
        except Exception as e:
            # 통합 JSON이므로 p 대신 JHNT_CTN을 사용
            logger.debug(f"Error parsing license for JHNT_CTN {jhnt}: {e}")
            continue

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

    dtype_map = {}
    for k in ["JHNT_MBN", "JHNT_CTN"]:
        dtype_map[k] = str

    base = pd.read_csv(RAW_CSV, encoding="utf-8", dtype=dtype_map)

    # 1. 이력서 데이터 처리
    logger.info(f"Detecting resume JSON structure based on {TOTAL_RESUME_JSON}.")
    # 파서 함수와 로드된 전체 데이터를 함께 받음
    resume_parser_func, all_resume_data = get_resume_parser(TOTAL_RESUME_JSON)
    logger.info("Starting resume data parsing.")
    # 로드된 전체 데이터를 파서 함수에 전달
    resume_df = resume_parser_func(all_resume_data) 

    # 2. 자기소개서 데이터 처리
    logger.info(f"Detecting cover letter JSON structure based on {TOTAL_COVER_JSON}.")
    # 파서 함수와 로드된 전체 데이터를 함께 받음
    cover_parser_func, all_cover_data = get_cover_parser(TOTAL_COVER_JSON)
    logger.info("Starting cover letter data parsing.")
    # 로드된 전체 데이터를 파서 함수에 전달
    cover_df = cover_parser_func(all_cover_data)

    # 3. 직업훈련 데이터 처리
    logger.info(f"Reading total training data from {TOTAL_TRAINING_JSON}.")
    try:
        all_train_data = _read_json_safe(TOTAL_TRAINING_JSON)
        # 로드된 전체 데이터를 파서 함수에 전달
        training_df = parse_train_to_lists(all_train_data)
    except Exception as e:
        logger.error(f"Failed to load or parse total training data: {e}")
        training_df = pd.DataFrame(columns=TRAINING_BASE_COLS)

    # 4. 자격증 데이터 처리
    logger.info(f"Reading total license data from {TOTAL_LICENSE_JSON}.")
    try:
        all_license_data = _read_json_safe(TOTAL_LICENSE_JSON)
        # 로드된 전체 데이터를 파서 함수에 전달
        license_df  = parse_license_to_lists(all_license_data)
    except Exception as e:
        logger.error(f"Failed to load or parse total license data: {e}")
        license_df = pd.DataFrame(columns=LICENSE_BASE_COLS)

    out = base.copy()
    logger.info(f"Base DataFrame shape: {out.shape}")

    # 데이터 병합 (이하 기존 로직과 동일)
    if not resume_df.empty and "JHNT_MBN" in out.columns:
        out = out.merge(resume_df, left_on="JHNT_MBN", right_on="SEEK_CUST_NO", how="left", suffixes=("", "_resume"))
        out = out.drop(columns=["SEEK_CUST_NO"], errors="ignore")
        logger.info(f"Merged resume data. Current shape: {out.shape}")

    if not cover_df.empty and "JHNT_MBN" in out.columns:
        out = out.merge(cover_df, left_on="JHNT_MBN", right_on="SEEK_CUST_NO", how="left", suffixes=("", "_cover"))
        out = out.drop(columns=["SEEK_CUST_NO"], errors="ignore")
        logger.info(f"Merged cover letter data. Current shape: {out.shape}")

    if not training_df.empty and "JHNT_CTN" in out.columns:
        out = out.merge(training_df, on="JHNT_CTN", how="left", suffixes=("", "_training"))
        logger.info(f"Merged training data. Current shape: {out.shape}")

    if not license_df.empty and "JHNT_CTN" in out.columns:
        out = out.merge(license_df, on="JHNT_CTN", how="left", suffixes=("", "_license"))
        logger.info(f"Merged license data. Current shape: {out.shape}")

    return out

# =========================
# 6) 후처리 (옵션)
# =========================
def postprocess(df: pd.DataFrame, logger: logging.LoggerAdapter, data_output_dir) -> pd.DataFrame:
    logger.info("Starting postprocessing: Binary mapping and date calculation.")
    
    # ---- (1) 바이너리 매핑 ----
    bin_map = {"예":1, "아니오":0, "아니요":0, "필요":1, "불필요":0}
    mapped_cols = [] 
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(bin_map)
            if df[col].dtype != object:
                mapped_cols.append(col)
                
    if mapped_cols:
        logger.info(f"Successfully applied Binary Mapping to {len(mapped_cols)} columns: {', '.join(mapped_cols)}") 
    else:
        logger.info("No object columns were successfully converted by binary mapping.")

    # ---- (2) 날짜 차이 계산  ----
    date_diff_cols = [] 
    if "JHCR_DE" in df.columns:
        anchor = pd.to_datetime(df["JHCR_DE"], errors="coerce")
        date_cols = [c for c in df.columns if any(x in c.upper() for x in ["DE","DT","DATE","BGDE","ENDE","STDT","ENDT"]) and "MDTN" not in c.upper()]
        
        for col in date_cols:
            if col == "JHCR_DE":
                continue
            
            vals = pd.to_datetime(df[col], errors="coerce")
            
            if not anchor.isna().all() and not vals.isna().all():
                diff = (vals - anchor).dt.days
                df[col] = diff.abs()
                date_diff_cols.append(col)
            else:
                 logger.warning(f"Skipped date diff for column '{col}' due to all-NaN anchor or all-NaN target date values.")

        if date_diff_cols:
            logger.info(f"Calculated Date Difference (days from JHCR_DE) for {len(date_diff_cols)} columns: {', '.join(date_diff_cols)}")
        else:
            logger.info("No date columns were processed for date difference calculation.")
    else:
        logger.warning("Anchor column 'JHCR_DE' not found. Skipping date difference calculation.")

    # ---- (3) 모든 값이 결측인 컬럼 제거 ----
    original_cols = df.shape[1]
    cols_to_drop = df.columns[df.isnull().all()].tolist()
    
    df = df.dropna(axis=1, how="all")
    dropped_cols_count = original_cols - df.shape[1]
    
    if dropped_cols_count > 0:
        logger.info(f"Dropped {dropped_cols_count} columns that were entirely missing values: {', '.join(cols_to_drop)}")
    
    # ---- (4) label encoding ----
    clos_ym_prefix_cols = [c for c in df.columns if c.startswith('CLOS_YM')]
    jhcr_de_prefix_cols = [c for c in df.columns if c.startswith('JHCR_DE')]
    excluded_cols = list(set(EXCLUDE_COLS + clos_ym_prefix_cols + jhcr_de_prefix_cols))

    cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c not in excluded_cols]

    encoding_map = {}
    for c in cat_cols:
        cat_dtype = pd.Categorical(df[c], categories=sorted(df[c].dropna().unique()))
        
        mapping = {label: code for code, label in enumerate(cat_dtype.categories)}
        encoding_map[c] = mapping
        
        df[c] = cat_dtype.codes

    df = df.assign(**{c: df[c].astype('int64') for c in df.select_dtypes(include=['bool']).columns})
    df[cat_cols] = df[cat_cols].astype('str')

    # 로깅 추가
    if cat_cols:
        logger.info(f"Applied Label Encoding to {len(cat_cols)} columns:")
        logger.info(f"    Encoded Columns: {cat_cols}") 
        logger.info(f"    Excluded Columns: {', '.join(excluded_cols)}")

        map_path = data_output_dir / "label_encoding_map.json"
        try:
            with map_path.open("w", encoding="utf-8") as f:
                json.dump(encoding_map, f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"Label Encoding Map saved to: {map_path.name}")
        except Exception as e:
            logger.error(f"Failed to save Label Encoding Map to {map_path.name}: {e}")
    else:
        logger.info(f"No categorical columns (excluding {', '.join(excluded_cols)}) found for Label Encoding.")
    
    logger.info("Postprocessing complete.")
    return df