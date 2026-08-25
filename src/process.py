"""Align FRED series and build lagged features for the model."""

import pandas as pd

from src.config import (
    ALIGNED_PATH,
    HOLDOUT_MONTHS,
    LAG_MONTHS,
    PREDICTOR,
    PROCESSED_DIR,
    RAW_PATH,
    TARGET,
)


def load_aligned() -> pd.DataFrame:
    """Load raw FRED CSV and align on a clean monthly index with no NaNs."""
    df = pd.read_csv(RAW_PATH, parse_dates=["DATE"])
    df = df.set_index("DATE").sort_index()
    df.index = df.index.to_period("M")
    return df.dropna()


def build_features(df: pd.DataFrame, lag_months: int = LAG_MONTHS) -> pd.DataFrame:
    """Add lagged unemployment columns (u_lag0..u_lagN) and lagged target (dq_lag1)."""
    out = df.copy()
    for i in range(lag_months + 1):
        out[f"u_lag{i}"] = out[PREDICTOR].shift(i)
    out["dq_lag1"] = out[TARGET].shift(1)
    return out


def split_train_test(df: pd.DataFrame, holdout: int = HOLDOUT_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a lagged frame into train and trailing holdout sets."""
    df = df.dropna()
    return df.iloc[:-holdout], df.iloc[-holdout:]


def process_all() -> pd.DataFrame:
    """Run the full processing step and persist the aligned/lagged frame."""
    aligned = load_aligned()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(ALIGNED_PATH)
    print(f"Aligned {len(aligned)} rows -> {ALIGNED_PATH}")
    return aligned


if __name__ == "__main__":
    process_all()
