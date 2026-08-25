import numpy as np
import pandas as pd

from src import model, process


def _df():
    idx = pd.period_range("2000Q1", periods=120, freq="Q")
    u = np.random.default_rng(0).normal(5, 1, 120)
    dq = 1.0 + 0.4 * np.roll(u, 3)  # delinquency responds to unemployment 3 quarters prior
    frame = pd.DataFrame(
        {"UNRATE": u, "DRALACBS": dq, "DRCCLACBS": dq + 1}, index=idx
    )
    return process.build_features(frame, lag_quarters=6).dropna()


def test_fit_static_returns_r_squared():
    r = model.fit_static(_df(), lag_quarters=6)
    assert 0 < r.r_squared <= 1.0
    assert len(r.params) == 1 + 6 + 1  # intercept + 7 lag terms (u_lag0..u_lag6)


def test_fit_dynamic_adds_ar_term():
    r = model.fit_dynamic(_df(), lag_quarters=6)
    assert "dq_lag1" in r.params
    assert len(r.params) == 1 + 7 + 1  # intercept + 7 lags + dq_lag1


def test_ccf_peak_negative_lag():
    # x leads y by 3 periods -> CCF should peak at lag -3
    x = pd.Series(np.sin(np.arange(100) / 5))
    y = x.shift(3).fillna(0)
    ccf = model.compute_ccf(x, y, max_lag=6)
    assert max(ccf, key=ccf.get) == -3
