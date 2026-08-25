import numpy as np
import pandas as pd

from src import process
from src.config import TARGET


def _frame():
    idx = pd.period_range("2020Q1", periods=24, freq="Q")
    return pd.DataFrame(
        {
            "UNRATE": np.arange(24) + 4.0,
            "DRALACBS": np.arange(24) + 1.0,
            "DRCCLACBS": np.arange(24) + 2.0,
        },
        index=idx,
    )


def test_build_features_shifts_correctly():
    df = process.build_features(_frame(), lag_quarters=2)
    assert list(df.columns) == [
        "UNRATE",
        "DRALACBS",
        "DRCCLACBS",
        "u_lag0",
        "u_lag1",
        "u_lag2",
        "dq_lag1",
    ]
    assert df["u_lag0"].iloc[2] == 6.0   # current UNRATE
    assert df["u_lag2"].iloc[2] == 4.0   # UNRATE shifted 2 quarters
    assert df["dq_lag1"].iloc[3] == df[TARGET].iloc[2]  # previous TARGET (shift 1)


def test_split_train_test_holdout():
    df = process.build_features(_frame(), lag_quarters=2).dropna()
    train, test = process.split_train_test(df, holdout=6)
    assert len(train) + len(test) == len(df)
    assert len(test) == 6
