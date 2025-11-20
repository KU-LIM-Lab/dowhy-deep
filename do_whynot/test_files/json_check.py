import json
from typing import List, Dict
import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import warnings
import uuid
from datetime import datetime
import pytz
import csv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from do_whynot.config import TOTAL_RESUME_JSON, TOTAL_COVER_JSON, TOTAL_TRAINING_JSON, TOTAL_LICENSE_JSON

RESULTS_DIR = None

class KSTFormatter(logging.Formatter):
    """UTC 타임스탬프를 KST (Asia/Seoul)로 변환하는 포맷터"""
    def converter(self, timestamp):
        # timestamp는 UTC 시간을 기준으로 합니다.
        KST = pytz.timezone('Asia/Seoul')
        dt = datetime.fromtimestamp(timestamp, pytz.utc)
        return dt.astimezone(KST).timetuple()

def setup_logger():
    """테스트 로깅 설정을 초기화하고 LoggerAdapter 객체를 반환합니다. """
    # 1) warnings 전부 무시 + 브릿지 차단
    warnings.filterwarnings("ignore")
    logging.captureWarnings(False)

    # 2) 외부 로거 경고 숨기기
    for name in [
        "py.warnings",
        "dowhy",
        "dowhy.causal_model",
        "dowhy.causal_identifier",
        "dowhy.causal_estimator",
    ]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)

    # 3) 고유 run_id (timestamp + uuid8)
    KST = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(KST)

    run_ts = now_kst.strftime("%Y%m%d-%H%M%S")
    run_uid = uuid.uuid4().hex[:8]
    run_id = f"{run_ts}_{run_uid}"


    # 4) 로그 디렉토리: 현재 파일 위치 기준으로 새 폴더 생성 후 로그 파일 저장
    global RESULTS_DIR
    log_base_dir = Path(__file__).resolve().parent / "logs"
    
    RESULTS_DIR = log_base_dir / run_id
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_path = RESULTS_DIR / f"run_log_{run_id}.log"

    # 5) 로거 + 핸들러
    base_logger = logging.getLogger("kubig.validation.tabpfn")
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    if not base_logger.handlers:
        fmt = KSTFormatter(
            "%(asctime)s | %(levelname)s | run=%(run_id)s | %(message)s"
        )

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        base_logger.addHandler(fh)
        base_logger.addHandler(ch)

    # 6) run_id를 필드로 주입하는 어댑터
    logger = logging.LoggerAdapter(base_logger, extra={"run_id": run_id})
    logger.info("Logging initialized. File: %s", str(log_path))
    return logger

def print_first_json(file_path: str, logger):
    with open(file_path, 'r', encoding='utf-8') as f:
        data: List[Dict] = json.load(f)
    first_five = data[:5]

    for i, item in enumerate(first_five):
        logger.info(f"dictionary #{i+1}")
        logger.info(json.dumps(item, indent=4, ensure_ascii=False))

def main():
    main_logger = setup_logger()
    main_logger.info("Starting Data Preprocessing Pipeline from imported functions.")

    print_first_json(TOTAL_RESUME_JSON, main_logger)
    print_first_json(TOTAL_COVER_JSON, main_logger)
    print_first_json(TOTAL_TRAINING_JSON, main_logger)
    print_first_json(TOTAL_LICENSE_JSON, main_logger)

if __name__ == "__main__":
    main()