# run settings
IS_TEST_MODE = True
TEST_SAMPLE_SIZE = 100
BATCH_SIZE = 50

DAG_INDICES_TEST = [1, 10]
DAG_INDICES = range(1, 43)

EXCLUDE_COLS = ["SELF_INTRO_CONT", "JHNT_MBN", "JHNT_CTN"]

MULTICLASS_THRESHOLD = 20
MULTICLASS_PASS = False

# directory settings
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
folder_name = "kubig_experiments"

DATA_OUTPUT_DIR = project_root / folder_name / "data" / "output"

DAG_DIR = project_root / folder_name / "dags"
MODEL_DIR = project_root / folder_name / "models"

RAW_CSV = project_root / folder_name / "data" / "synthetic_data_raw.csv"

RESUME_DIR   =  project_root / folder_name / "data" / "RESUME_JSON/ver1"
COVER_DIR    = project_root / folder_name / "data"  / "COVERLETTERS_JSON/ver1"
TRAINING_DIR = project_root / folder_name / "data"  / "TRAININGS_JSON"
LICENSE_DIR  = project_root / folder_name / "data"  / "LICENSES_JSON"