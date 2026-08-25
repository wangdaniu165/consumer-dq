"""Fit delinquency-on-unemployment models and lead/lag diagnostics."""

import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    LAG_QUARTERS,
    PARAMS_PATH,
    PREDICTOR,
    PROCESSED_DIR,
    TARGET,
)
from src.process import build_features, load_aligned


@dataclass
class FitResult:
    """A fitted OLS model, reduced to the quantities the dashboard needs."""

    params: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted: pd.Series


@dataclass
class ECMResult:
    """A fitted two-step Engle-Granger error-correction model."""

    long_run_params: dict[str, float]   # cointegrating regression DQ = β₀ + β₁·U
    speed_of_adjustment: float          # λ on ECT_{t-1}; must be < 0 for error correction
    short_run_params: dict[str, float]  # γᵢ on ΔU_{t-i} + const
    r_squared: float
    residuals: pd.Series
    fitted: pd.Series


@dataclass
class PiecewiseResult:
    """A fitted piecewise-linear (linear spline) model in unemployment."""

    knots: list[float]        # interior breakpoints, sorted
    params: dict[str, float]  # const, UNRATE, hinge1, hinge2
    slopes: list[float]       # slope of each segment (len(knots) + 1)
    r_squared: float
    sse: float
    fitted: pd.Series
    residuals: pd.Series


def _fit_ols(df: pd.DataFrame, regressors: list[str], y_col: str = TARGET) -> FitResult:
    y = df[y_col]
    X = sm.add_constant(df[regressors])
    model = sm.OLS(y, X).fit()
    return FitResult(
        params={col: float(model.params[col]) for col in X.columns},
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        residuals=model.resid,
        fitted=model.fittedvalues,
    )


def fit_static(df: pd.DataFrame, lag_quarters: int = LAG_QUARTERS) -> FitResult:
    """Distributed-lag OLS: DQ_t = α + Σ βᵢ · U_{t-i}  (no AR term)."""
    regressors = [f"u_lag{i}" for i in range(lag_quarters + 1)]
    return _fit_ols(df, regressors)


def fit_dynamic(df: pd.DataFrame, lag_quarters: int = LAG_QUARTERS) -> FitResult:
    """Dynamic distributed-lag OLS: adds AR(1) term DQ_{t-1} for persistence."""
    regressors = [f"u_lag{i}" for i in range(lag_quarters + 1)] + ["dq_lag1"]
    return _fit_ols(df, regressors)


def fit_ecm(df: pd.DataFrame, lag_quarters: int = LAG_QUARTERS) -> ECMResult:
    """Two-step Engle-Granger error-correction model.

    Step 1 — long-run cointegrating regression (levels): DQ_t = β₀ + β₁·U_t + e_t.
    Step 2 — short-run dynamics (differences):
        ΔDQ_t = α + λ·e_{t-1} + Σ_{i=0}^{lag_quarters} γᵢ·ΔU_{t-i} + ε_t
    λ is the speed of adjustment and must be negative for error correction.
    """
    # Step 1: long-run relationship; residual is the error-correction term (ECT).
    long_run = _fit_ols(df, [PREDICTOR])
    ect = df[TARGET] - long_run.fitted

    # Step 2: short-run regression in first differences.
    d = df.copy()
    d["dDQ"] = df[TARGET].diff()
    d["ECT_lag1"] = ect.shift(1)
    for i in range(lag_quarters + 1):
        d[f"du_lag{i}"] = df[PREDICTOR].diff().shift(i)

    regressors = ["ECT_lag1"] + [f"du_lag{i}" for i in range(lag_quarters + 1)]
    short = _fit_ols(d.dropna(), regressors, y_col="dDQ")

    return ECMResult(
        long_run_params=long_run.params,
        speed_of_adjustment=short.params.get("ECT_lag1", 0.0),
        short_run_params=short.params,
        r_squared=short.r_squared,
        residuals=short.residuals,
        fitted=short.fitted,
    )


def lead_lag_diagnostic(df: pd.DataFrame, lag_quarters: int = LAG_QUARTERS) -> pd.DataFrame:
    """Two-sided regression with both leads and lags of unemployment.

    Significant LEAD coefficients (future unemployment) would suggest delinquency
    actually leads unemployment; significant LAG coefficients support the reverse.
    """
    df2 = df.copy()
    leads = []
    for i in range(1, lag_quarters + 1):
        col = f"u_lead{i}"
        df2[col] = df[PREDICTOR].shift(-i)
        leads.append(col)
    lags = []
    for i in range(lag_quarters + 1):
        col = f"u_lag{i}"
        df2[col] = df[PREDICTOR].shift(i)
        lags.append(col)
    result = _fit_ols(df2.dropna(), leads + lags)
    return pd.DataFrame({"coefficient": result.params}).rename_axis("regressor")


def compute_ccf(x: pd.Series, y: pd.Series, max_lag: int = 24) -> dict[int, float]:
    """Cross-correlation of `x` against `y` over lags `-max_lag .. max_lag`.

    CCF[lag] = corr(x_{t+lag}, y_t). A **negative** lag means **x leads y**
    (x moves first, y follows) — e.g. unemployment leading delinquency. The
    empirical peak of this function is the answer to "leading or lagging?".
    """
    result: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        result[lag] = x.shift(-lag).corr(y)
    return result


def fit_piecewise(df: pd.DataFrame, n_knots: int = 4, n_candidates: int = 18) -> PiecewiseResult:
    """Piecewise-linear (linear spline) regression of DQ on unemployment.

    Fits ``DQ = β₀ + β₁·U + Σⱼ β₁₊ⱼ·max(U − cⱼ, 0)`` and grid-searches the
    knot locations ``cⱼ`` (ordered combinations of candidate unemployment levels)
    to minimise SSE. Each segment's slope is the cumulative sum of the β's.
    """
    x = df[PREDICTOR].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    lo, hi = np.quantile(x, [0.05, 0.95])
    candidates = np.linspace(lo, hi, n_candidates)

    best_sse: float | None = None
    best_knots = None
    best_beta = None

    for knots in combinations(candidates, n_knots):
        cols = [np.ones_like(x), x] + [np.maximum(x - c, 0) for c in knots]
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ beta) ** 2).sum())
        if best_sse is None or sse < best_sse:
            best_sse, best_knots, best_beta = sse, list(knots), beta

    X = np.column_stack([np.ones_like(x), x]
                        + [np.maximum(x - c, 0) for c in best_knots])
    fitted = X @ best_beta
    residuals = y - fitted
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - best_sse / sst

    params = {"const": float(best_beta[0]), PREDICTOR: float(best_beta[1])}
    for i in range(n_knots):
        params[f"hinge{i + 1}"] = float(best_beta[2 + i])
    slopes = np.cumsum(best_beta[1:]).tolist()  # β₁, β₁+β₂, β₁+β₂+β₃, …

    return PiecewiseResult(
        knots=[float(c) for c in best_knots],
        params=params,
        slopes=[float(s) for s in slopes],
        r_squared=r2,
        sse=best_sse,
        fitted=pd.Series(fitted, index=df.index),
        residuals=pd.Series(residuals, index=df.index),
    )


def predict_piecewise(u, result: PiecewiseResult):
    """Evaluate the piecewise curve at unemployment value(s) `u`."""
    out = result.params["const"] + result.params[PREDICTOR] * np.asarray(u)
    for i, c in enumerate(result.knots, start=1):
        out = out + result.params[f"hinge{i}"] * np.maximum(np.asarray(u) - c, 0)
    return out


def select_knots_bic(df: pd.DataFrame, max_knots: int = 5, n_candidates: int = 18) -> tuple[int, list[dict]]:
    """Choose the number of knots (0..max_knots) minimising BIC.

    BIC = n·ln(SSE/n) + k·ln(n), with k = n_knots + 2 parameters (intercept + U +
    hinges). Penalises overfitting so the data — not the user — picks the count.
    """
    n = len(df)
    results = []
    for k in range(0, max_knots + 1):
        pw = fit_piecewise(df, n_knots=k, n_candidates=n_candidates)
        bic = n * np.log(pw.sse / n) + (k + 2) * np.log(n)
        results.append({"n_knots": k, "bic": bic, "r2": pw.r_squared})
    best = min(results, key=lambda r: r["bic"])
    return best["n_knots"], results


def fit_all() -> dict:
    """Fit static + dynamic + ECM + piecewise models and persist params to JSON."""
    df = build_features(load_aligned()).dropna()
    static = fit_static(df)
    dynamic = fit_dynamic(df)
    ecm = fit_ecm(df)
    piecewise = fit_piecewise(df)
    params = {
        "static": static.params,
        "dynamic": dynamic.params,
        "r2_static": static.r_squared,
        "r2_dynamic": dynamic.r_squared,
        "ecm": {
            "long_run": ecm.long_run_params,
            "speed_of_adjustment": ecm.speed_of_adjustment,
            "short_run": ecm.short_run_params,
            "r2_ecm": ecm.r_squared,
        },
        "piecewise": {
            "knots": piecewise.knots,
            "slopes": piecewise.slopes,
            "r2_piecewise": piecewise.r_squared,
        },
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Fitted models -> {PARAMS_PATH}")
    return params


if __name__ == "__main__":
    fit_all()
