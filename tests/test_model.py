import numpy as np
import pandas as pd

from src import model, process


def _df():
    idx = pd.period_range("2000Q1", periods=120, freq="Q")
    u = np.random.default_rng(0).normal(5, 1, 120)
    dq = 1.0 + 0.4 * np.roll(u, 3)  # delinquency responds to unemployment 3 quarters prior
    frame = pd.DataFrame(
        {"UNRATE": u, "DRALACBS": dq, "DRCCLACBS": dq + 1, "NYFED_OTHER_90DPD": dq},
        index=idx,
    )
    return process.build_features(frame, lag_quarters=6).dropna()


def test_fit_contemporaneous():
    r = model.fit_contemporaneous(_df())
    assert 0 < r.r_squared <= 1.0
    assert len(r.params) == 2  # intercept + u_lag0
    assert "u_lag0" in r.t_values
    assert "u_lag0" in r.p_values


def test_fit_tracking_has_ar_term():
    r = model.fit_tracking(_df())
    assert "dq_lag1" in r.params
    assert "u_lag0" in r.params
    assert 0 < r.r_squared <= 1.0


def test_ccf_peak_negative_lag():
    # x leads y by 3 periods -> CCF should peak at lag -3
    x = pd.Series(np.sin(np.arange(100) / 5))
    y = x.shift(3).fillna(0)
    ccf = model.compute_ccf(x, y, max_lag=6)
    assert max(ccf, key=ccf.get) == -3


def test_fit_piecewise_recovers_knots_and_convexity():
    # Convex piecewise: steeper slopes in higher-unemployment segments.
    rng = np.random.default_rng(0)
    u = rng.uniform(3, 10, 400)
    c1, c2 = 5.0, 8.0
    y = (0.1 * u + 0.2 * np.maximum(u - c1, 0) + 0.3 * np.maximum(u - c2, 0)
         + rng.normal(0, 0.05, 400))
    idx = pd.period_range("1980Q1", periods=400, freq="Q")
    df = pd.DataFrame({"UNRATE": u, "DRALACBS": y, "DRCCLACBS": y, "NYFED_OTHER_90DPD": y}, index=idx)

    r = model.fit_piecewise(df, n_knots=2)
    assert r.r_squared > 0.9
    assert abs(r.knots[0] - c1) < 0.6
    assert abs(r.knots[1] - c2) < 0.6
    assert r.slopes[0] < r.slopes[1] < r.slopes[2]  # convex: increasing slopes


def test_fit_piecewise_slope_count():
    idx = pd.period_range("1980Q1", periods=100, freq="Q")
    rng = np.random.default_rng(0)
    u = rng.uniform(3, 10, 100)
    y = 0.1 * u + rng.normal(0, 0.1, 100)
    df = pd.DataFrame({"UNRATE": u, "DRALACBS": y, "DRCCLACBS": y, "NYFED_OTHER_90DPD": y}, index=idx)

    r = model.fit_piecewise(df, n_knots=4)
    assert len(r.knots) == 4
    assert len(r.slopes) == 5


def test_select_knots_bic_prefers_true_count():
    # True DGP has 2 knots; BIC should not overfit to 4-5.
    rng = np.random.default_rng(2)
    u = rng.uniform(3, 10, 400)
    c1, c2 = 5.0, 8.0
    y = (0.1 * u + 0.2 * np.maximum(u - c1, 0) + 0.3 * np.maximum(u - c2, 0)
         + rng.normal(0, 0.05, 400))
    idx = pd.period_range("1980Q1", periods=400, freq="Q")
    df = pd.DataFrame({"UNRATE": u, "DRALACBS": y, "DRCCLACBS": y, "NYFED_OTHER_90DPD": y}, index=idx)

    best, results = model.select_knots_bic(df, max_knots=5)
    assert best <= 2                       # penalised — does not chase 3+ knots
    assert any(r["n_knots"] == 5 for r in results)  # full sweep ran


def test_covid_dummy_included_when_present():
    idx = pd.period_range("2018Q1", periods=24, freq="Q")
    u = np.arange(24) + 4.0
    dq = np.arange(24) + 1.0
    frame = pd.DataFrame({"UNRATE": u, "DRALACBS": dq, "DRCCLACBS": dq, "NYFED_OTHER_90DPD": dq}, index=idx)
    frame["covid"] = ((idx >= pd.Period("2020Q1")) & (idx <= pd.Period("2021Q4"))).astype(float)
    df = process.build_features(frame, lag_quarters=2).dropna()

    pw = model.fit_piecewise(df, n_knots=2)
    assert "covid" in df.columns
    assert pw.covid_coef != 0.0


def test_exclude_covid_drops_window_and_column():
    idx = pd.period_range("2019Q3", periods=12, freq="Q")  # 2019Q3..2022Q2
    frame = pd.DataFrame(
        {"UNRATE": np.arange(12.0), "DRALACBS": np.arange(12.0),
         "DRCCLACBS": np.arange(12.0)}, index=idx
    )
    frame["covid"] = (
        (idx >= pd.Period("2020Q1")) & (idx <= pd.Period("2021Q4"))
    ).astype(float)

    out = process.exclude_covid(frame)
    assert "covid" not in out.columns          # dummy no longer needed
    assert pd.Period("2020Q2", "Q") not in out.index
    assert pd.Period("2021Q3", "Q") not in out.index
    assert pd.Period("2019Q4", "Q") in out.index
    assert pd.Period("2022Q1", "Q") in out.index
    assert len(out) == 4                       # 12 quarters minus 8 COVID quarters


def test_fit_piecewise_monotone_nonnegative_slopes():
    # Noisy relationship that can tempt a negative slope somewhere; the
    # monotone fit must still keep every segment slope >= 0.
    rng = np.random.default_rng(3)
    u = rng.uniform(3, 11, 300)
    y = 0.4 * u + rng.normal(0, 0.5, 300)
    idx = pd.period_range("1980Q1", periods=300, freq="Q")
    df = pd.DataFrame({"UNRATE": u, "DRALACBS": y, "DRCCLACBS": y, "NYFED_OTHER_90DPD": y}, index=idx)

    mono = model.fit_piecewise_monotone(df, n_knots=4)
    assert all(s >= 0 for s in mono.slopes)
    assert 0 <= mono.r_squared <= 1.0
