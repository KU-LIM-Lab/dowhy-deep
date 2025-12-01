## run settings
IS_TEST_MODE = False    # test mode 실행 여부
TEST_SAMPLE_SIZE = 100   # test mode시 샘플링할 데이터 수
DAG_INDICES_TEST = [1]   # test mode시 사용할 DAG 인덱스

BATCH_SIZE = 1000   # 한 batch에서 처리할 데이터 수 (Tabpfn 추정기 한계로 10000 이하 권장)
EXCLUDE_COLS = ["SELF_INTRO_CONT", "JHNT_MBN", "JHNT_CTN"]  # label encoding에서 제외할 컬럼 리스트
PREFIX_COLS = ['CLOS_YM', 'JHCR_DE'] # label encoding에서 제외할 접두사 컬럼 리스트
# DAG_INDICES = range(1, 43)  # 실행할 DAG 인덱스 범위
DAG_INDICES = [2, 4, 6, 12, 13, 14, 15, 21, 22, 23, 24, 31, 37, 41, 42]

MULTICLASS_THRESHOLD = 20  # 다중 클래스 판단 기준 (고유값 수)
MULTICLASS_PASS = True    # threshold 이상의 레이블을 갖는 다중 클래스 처리 허용 여부

P_VALUE_THRESHOLD = 0.05   # 유의성 기준 (인과성 반박이 실패하지 않았다고 간주하는 P-value의 하한선)

BATCH_NUM = 10
BATCH_NUM_FIX = False

START_BATCH_NUM = 33

## directory settings
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
folder_name = "do_whynot"   # 폴더명

DATA_OUTPUT_DIR = project_root / folder_name / "data" / "final_output"   # 데이터 처리 결과물 저장 경로

DAG_DIR = project_root / folder_name / "dags"   # DAG 파일 경로
MODEL_DIR = project_root / folder_name / "models"   # 모델 파일 경로

RAW_CSV = project_root / folder_name / "data" / "keis_raw.csv"   # 구직인증데이터 경로

RESUME_DIR   =  project_root / folder_name / "data" / "RESUME_JSON"    # 이력서 json 파일 경로
COVER_DIR    = project_root / folder_name / "data"  / "COVERLETTERS_JSON"   # 자기소개서 json 파일 경로
TRAINING_DIR = project_root / folder_name / "data"  / "TRAININGS_JSON"  # 직업훈련 json 파일 경로
LICENSE_DIR  = project_root / folder_name / "data"  / "LICENSES_JSON"  # 자격증 json 파일 경로

TOTAL_RESUME_JSON = RESUME_DIR / "resume.json"
TOTAL_COVER_JSON = COVER_DIR / "coverletters.json"
TOTAL_TRAINING_JSON = TRAINING_DIR / "trainings.json"
TOTAL_LICENSE_JSON = LICENSE_DIR / "licenses.json"