"""Configuration constants for consumer DQ vs unemployment model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# FRED series: unemployment (predictor), credit-card delinquency (target — unsecured
# consumer credit is the closest FRED proxy for a fintech unsecured book), all-loan
# delinquency (comparison). UNRATE is monthly; DRALACBS/DRCCLACBS are quarterly (FFIEC).
SERIES_IDS = ["UNRATE", "DRALACBS", "DRCCLACBS"]
FRED_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
RAW_PATHS = {sid: RAW_DIR / f"{sid}.csv" for sid in SERIES_IDS}

# File paths
ALIGNED_PATH = PROCESSED_DIR / "aligned.csv"
PARAMS_PATH = PROCESSED_DIR / "model_params.json"

# Model specification (quarterly)
TARGET = "DRCCLACBS"     # credit-card delinquency — unsecured consumer credit
PREDICTOR = "UNRATE"
LAG_QUARTERS = 4         # quarters of unemployment history fed into the model (~1 year)
HOLDOUT_QUARTERS = 20    # trailing quarters (5 years) held out of the fit for evaluation
START_DATE = "2005Q1"    # earliest quarter retained (drops the noisy pre-2005 regime)

# Unemployment scenario presets for the stress test (percentage-point step)
SCENARIOS = {
    "baseline": {"kind": "hold"},
    "moderate": {"kind": "step", "delta": 2.0},
    "severe": {"kind": "step", "delta": 5.0},
}

# COVID anomaly window — delinquency was policy-distorted (forbearance/stimulus).
# Two ways to handle it:
#   * EXCLUDE_COVID=True  → drop these quarters entirely (default; static R² 0.72→0.92)
#   * EXCLUDE_COVID=False → keep them and add an intercept-shift dummy instead
COVID_START = "2020Q1"
COVID_END = "2021Q4"
EXCLUDE_COVID = True
