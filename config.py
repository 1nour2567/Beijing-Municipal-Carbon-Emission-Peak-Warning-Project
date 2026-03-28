
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

YEARS = list(range(2005, 2026))
MONTHS = list(range(1, 13))
SECTORS = ["total", "building", "transport", "industry"]

STL_PARAMS = {
    "period": 12,
    "seasonal": 7,
    "trend": 21,
    "robust": True
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

CALIBRATION_VARIABLES = ["temperature", "precipitation", "holiday_flag", "travel_index"]

SEASONAL_WEIGHTS_PRIOR = {
    1: 0.085,
    2: 0.078,
    3: 0.082,
    4: 0.080,
    5: 0.083,
    6: 0.085,
    7: 0.090,
    8: 0.088,
    9: 0.084,
    10: 0.082,
    11: 0.083,
    12: 0.080
}
