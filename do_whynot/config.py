## run settings
IS_TEST_MODE = True    # test mode 실행 여부
TEST_SAMPLE_SIZE = 100   # test mode시 샘플링할 데이터 수
DAG_INDICES_TEST = [1, 2]   # test mode시 사용할 DAG 인덱스

BATCH_SIZE = 100   # 한 batch에서 처리할 데이터 수 (Tabpfn 추정기 한계로 10000 이하 권장)
EXCLUDE_COLS = ["SELF_INTRO_CONT", "JHNT_MBN", "JHNT_CTN"]  # label encoding에서 제외할 컬럼 리스트
DAG_INDICES = range(1, 43)  # 실행할 DAG 인덱스 범위

MULTICLASS_THRESHOLD = 20  # 다중 클래스 판단 기준 (고유값 수)
MULTICLASS_PASS = False    # threshold 이상의 레이블을 갖는 다중 클래스 처리 허용 여부

P_VALUE_THRESHOLD = 0.05   # 유의성 기준 (인과성 반박이 실패하지 않았다고 간주하는 P-value의 하한선)

## directory settings
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
folder_name = "do_whynot"   # 폴더명

DATA_OUTPUT_DIR = project_root / folder_name / "data" / "output"   # 데이터 처리 결과물 저장 경로

DAG_DIR = project_root / folder_name / "dags"   # DAG 파일 경로
MODEL_DIR = project_root / folder_name / "models"   # 모델 파일 경로

RAW_CSV = project_root / folder_name / "data" / "synthetic_data_raw.csv"   # 구직인증데이터 경로

RESUME_DIR   =  project_root / folder_name / "data" / "RESUME_JSON/ver1"    # 이력서 json 파일 경로
COVER_DIR    = project_root / folder_name / "data"  / "COVERLETTERS_JSON/ver1"   # 자기소개서 json 파일 경로
TRAINING_DIR = project_root / folder_name / "data"  / "TRAININGS_JSON/output"  # 직업훈련 json 파일 경로
LICENSE_DIR  = project_root / folder_name / "data"  / "LICENSES_JSON/output"  # 자격증 json 파일 경로