"""Fit delinquency-on-unemployment models and lead/lag diagnostics."""

import json
from dataclasses import dataclass

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


def fit_all() -> dict:
    """Fit static + dynamic + ECM models and persist params to JSON."""
    df = build_features(load_aligned()).dropna()
    static = fit_static(df)
    dynamic = fit_dynamic(df)
    ecm = fit_ecm(df)
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
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Fitted models -> {PARAMS_PATH}")
    return params


if __name__ == "__main__":
    fit_all()
