"""Configuration constants for consumer DQ vs unemployment model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# FRED series: unemployment (predictor), all-loan delinquency (target), credit-card (comparison)
# UNRATE is monthly; DRALACBS/DRCCLACBS are quarterly (FFIEC Call Reports) — aligned at Q freq.
SERIES_IDS = ["UNRATE", "DRALACBS", "DRCCLACBS"]
FRED_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
RAW_PATHS = {sid: RAW_DIR / f"{sid}.csv" for sid in SERIES_IDS}

# File paths
ALIGNED_PATH = PROCESSED_DIR / "aligned.csv"
PARAMS_PATH = PROCESSED_DIR / "model_params.json"

# Model specification (quarterly)
TARGET = "DRALACBS"
PREDICTOR = "UNRATE"
LAG_QUARTERS = 4         # quarters of unemployment history fed into the model (~1 year)
HOLDOUT_QUARTERS = 20    # trailing quarters (5 years) held out of the fit for evaluation

# Unemployment scenario presets for the stress test (percentage-point step)
SCENARIOS = {
    "baseline": {"kind": "hold"},
    "moderate": {"kind": "step", "delta": 2.0},
    "severe": {"kind": "step", "delta": 5.0},
}

# COVID anomaly window — delinquency was policy-distorted (forbearance/stimulus),
# so these quarters get an intercept-shift dummy in the regressions.
COVID_START = "2020Q1"
COVID_END = "2021Q4"
