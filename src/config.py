"""Configuration constants for consumer DQ vs unemployment model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# FRED series: unemployment (predictor), all-loan delinquency (target), credit-card (comparison)
SERIES_IDS = ["UNRATE", "DRALACBS", "DRCCLACBS"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(SERIES_IDS)

# File paths
RAW_PATH = RAW_DIR / "fred_raw.csv"
ALIGNED_PATH = PROCESSED_DIR / "aligned.csv"
PARAMS_PATH = PROCESSED_DIR / "model_params.json"

# Model specification
TARGET = "DRALACBS"
PREDICTOR = "UNRATE"
LAG_MONTHS = 12          # months of unemployment history fed into the model
HOLDOUT_MONTHS = 60      # trailing months held out of the fit for evaluation

# Unemployment scenario presets for the stress test
SCENARIOS = {
    "baseline": {"kind": "hold"},
    "moderate": {"kind": "step", "delta": 2.0},
    "severe": {"kind": "step", "delta": 5.0},
}
