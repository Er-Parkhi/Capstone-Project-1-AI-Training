
from pathlib import Path
 
# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_sample.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"
 
# Target and feature definitions

TARGET_COLUMN = "isFraud"
 
NUMERIC_FEATURES = [

    "step",

    "amount",

    "oldbalanceOrg",

    "newbalanceOrig",

    "oldbalanceDest",

    "newbalanceDest",

]
 
CATEGORICAL_FEATURES = ["type"]
 
# Excluded intentionally:

# nameOrig, nameDest: raw account identifiers

# isFlaggedFraud: existing fraud flag that may leak information
 
RANDOM_STATE = 42

TEST_SIZE = 0.20
 
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

