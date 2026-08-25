"""Configuration constants for consumer DQ vs unemployment model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# FRED series (downloaded by `dq-download`): unemployment (predictor) and one
# FFIEC delinquency comparison (consumer ex-credit-card). UNRATE is monthly;
# DROCLACBS is quarterly.
FRED_SERIES_IDS = ["UNRATE", "DROCLACBS"]
# All series loaded into the aligned frame: FRED + the NY Fed/Equifax "Other"
# 90+ day serious-delinquency series (not on FRED — pulled by `dq-nyfed`).
SERIES_IDS = FRED_SERIES_IDS + ["NYFED_OTHER_90DPD"]
FRED_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
RAW_PATHS = {sid: RAW_DIR / f"{sid}.csv" for sid in SERIES_IDS}

# File paths
ALIGNED_PATH = PROCESSED_DIR / "aligned.csv"
PARAMS_PATH = PROCESSED_DIR / "model_params.json"

# Model specification (quarterly)
TARGET = "NYFED_OTHER_90DPD"   # NY Fed/Equifax "Other" 90+ day serious delinquency
PREDICTOR = "UNRATE"
LAG_QUARTERS = 4         # quarters of unemployment history fed into the model (~1 year)
HOLDOUT_QUARTERS = 20    # trailing quarters (5 years) held out of the fit for evaluation
START_DATE = "2005Q1"    # earliest quarter retained (drops the noisy pre-2005 regime)

# COVID anomaly window — delinquency was policy-distorted (forbearance/stimulus).
# Two ways to handle it:
#   * EXCLUDE_COVID=True  → drop these quarters entirely (default; static R² 0.72→0.92)
#   * EXCLUDE_COVID=False → keep them and add an intercept-shift dummy instead
COVID_START = "2020Q1"
COVID_END = "2021Q4"
EXCLUDE_COVID = True
