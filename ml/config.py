"""Central configuration: paths, random seed and constants shared across the project."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "bengaluru_house_data.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
PROCESSED_DATA_PATH: Path = PROCESSED_DIR / "bengaluru_clean.csv"

ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"
MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
LOCATIONS_PATH: Path = ARTIFACTS_DIR / "locations.json"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Reproducibility / training
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
SEARCH_ITERATIONS: int = 20
INTERVAL_QUANTILES: tuple[float, float] = (0.10, 0.90)

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
DROP_COLUMNS: list[str] = ["area_type", "availability", "society", "balcony"]
CATEGORICAL_FEATURES: list[str] = ["location"]
NUMERIC_FEATURES: list[str] = ["total_sqft", "bhk", "bath"]
FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET: str = "price"  # in lakhs (1 lakh = 100,000 INR)

OTHER_LOCATION: str = "other"
MIN_LISTINGS_PER_LOCATION: int = 10  # locations with <= this many listings -> "other"
MIN_SQFT_PER_BHK: float = 300.0
MAX_EXTRA_BATHS: int = 2  # bath must be <= bhk + MAX_EXTRA_BATHS
MIN_GROUP_SIZE_FOR_BHK_RULE: int = 5  # (n-1)-BHK group must have > this many rows

# Conversion factors from a unit to square feet.
UNIT_TO_SQFT: dict[str, float] = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Acres": 43560.0,
    "Cents": 435.6,
    "Guntha": 1089.0,
    "Grounds": 2400.0,
    "Perch": 272.25,
}

# API validation bounds (mirrored in the frontend).
SQFT_MIN, SQFT_MAX = 300, 30000
BHK_MIN, BHK_MAX = 1, 10
BATH_MIN, BATH_MAX = 1, 10
