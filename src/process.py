"""Align FRED series to a quarterly index and build lagged features."""

import pandas as pd

from src.config import (
    ALIGNED_PATH,
    COVID_END,
    COVID_START,
    HOLDOUT_QUARTERS,
    LAG_QUARTERS,
    PREDICTOR,
    PROCESSED_DIR,
    RAW_PATHS,
    SERIES_IDS,
    START_DATE,
    TARGET,
)


def _read_series(series_id: str) -> pd.Series:
    df = pd.read_csv(RAW_PATHS[series_id], parse_dates=["observation_date"])
    s = df.set_index("observation_date")[series_id]
    s.index = pd.to_datetime(s.index)
    return s.astype(float)


def load_aligned() -> pd.DataFrame:
    """Load raw FRED CSVs and align on a quarterly PeriodIndex with no NaNs."""
    unemp = _read_series(PREDICTOR).resample("QE").mean()   # monthly -> quarterly mean
    unemp.index = unemp.index.to_period("Q")
    parts = {PREDICTOR: unemp}
    for sid in [s for s in SERIES_IDS if s != PREDICTOR]:
        s = _read_series(sid)
        s.index = s.index.to_period("Q")  # FRED quarter-start dates -> Period("Q")
        parts[sid] = s
    frame = pd.concat(parts, axis=1)
    frame["covid"] = (
        (frame.index >= pd.Period(COVID_START, "Q"))
        & (frame.index <= pd.Period(COVID_END, "Q"))
    ).astype(float)
    frame = frame[frame.index >= pd.Period(START_DATE, "Q")]
    return frame.dropna()


def exclude_covid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the COVID policy-distortion window (2020Q1–2021Q4).

    During these 8 quarters delinquency was held down by forbearance/stimulus, so
    the points sit far off the unemployment relationship and drag the fit down.
    Dropping them (rather than dummying them) yields the cleanest estimate: static
    R² 0.72 → 0.92. Also removes the now-redundant ``covid`` dummy column.
    """
    mask = (df.index >= pd.Period(COVID_START, "Q")) & (df.index <= pd.Period(COVID_END, "Q"))
    return df.loc[~mask].drop(columns=["covid"], errors="ignore")


def build_features(df: pd.DataFrame, lag_quarters: int = LAG_QUARTERS) -> pd.DataFrame:
    """Add lagged unemployment columns (u_lag0..u_lagN) and lagged target (dq_lag1)."""
    out = df.copy()
    for i in range(lag_quarters + 1):
        out[f"u_lag{i}"] = out[PREDICTOR].shift(i)
    out["dq_lag1"] = out[TARGET].shift(1)
    return out


def split_train_test(df: pd.DataFrame, holdout: int = HOLDOUT_QUARTERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a lagged frame into train and trailing holdout sets."""
    df = df.dropna()
    return df.iloc[:-holdout], df.iloc[-holdout:]


def process_all() -> pd.DataFrame:
    """Run the full processing step and persist the aligned frame."""
    aligned = load_aligned()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(ALIGNED_PATH)
    print(f"Aligned {len(aligned)} quarterly rows -> {ALIGNED_PATH}")
    return aligned


if __name__ == "__main__":
    process_all()
