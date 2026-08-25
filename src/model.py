"""Fit delinquency-on-unemployment models and lead/lag diagnostics."""

import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import lsq_linear

from src.config import (
    EXCLUDE_COVID,
    LAG_QUARTERS,
    PARAMS_PATH,
    PREDICTOR,
    PROCESSED_DIR,
    TARGET,
)
from src.process import build_features, exclude_covid, load_aligned


@dataclass
class FitResult:
    """A fitted OLS model, reduced to the quantities the dashboard needs."""

    params: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted: pd.Series
    t_values: dict[str, float]
    p_values: dict[str, float]


@dataclass
class PiecewiseResult:
    """A fitted piecewise-linear (linear spline) model in unemployment."""

    knots: list[float]     # interior breakpoints, sorted
    intercept: float       # β₀
    covid_coef: float      # intercept shift during the COVID window (0 if absent)
    slopes: list[float]    # slope of each segment (len(knots) + 1)
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
        t_values={col: float(model.tvalues[col]) for col in X.columns},
        p_values={col: float(model.pvalues[col]) for col in X.columns},
    )


def fit_contemporaneous(df: pd.DataFrame) -> FitResult:
    """Contemporaneous-only levels regression: DQ_t = α + β·U_t.

    The parsimonious specification: unemployment lags are ~95% collinear, so
    their individual coefficients are unidentified, but the contemporaneous
    slope is cleanly estimated (t ≈ 8.6).
    """
    return _fit_ols(df, ["u_lag0"])


def fit_tracking(df: pd.DataFrame) -> FitResult:
    """ARX tracking fit: DQ_t = α + ρ·DQ_{t-1} + β·U_t.

    NOT a causal model — the AR(1) term is near-unit-root persistence (ρ ≈ 0.93),
    which mechanically "explains" ~94% of variance and absorbs unemployment's
    effect (β becomes insignificant). Used only as a backcast/tracking overlay.
    """
    return _fit_ols(df, ["u_lag0", "dq_lag1"])


def fit_diff(df: pd.DataFrame) -> FitResult:
    """First-difference model: ΔDQ_t = α + β·ΔU_t + γ·ΔDGS10_t.

    The stationary (non-spurious) specification. ΔDGS10 is the change in the
    10-year Treasury — a forward-looking recession signal (negative sign: rates
    fall when the economy weakens, which is when delinquency rises).
    """
    d = pd.DataFrame({
        "dDQ": df[TARGET].diff(),
        "dU": df[PREDICTOR].diff(),
        "dDGS10": df["DGS10"].diff(),
    }).dropna()
    return _fit_ols(d, ["dU", "dDGS10"], y_col="dDQ")


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


def fit_piecewise(df: pd.DataFrame, n_knots: int = 1, n_candidates: int = 18) -> PiecewiseResult:
    """Piecewise-linear (linear spline) regression of DQ on unemployment.

    Fits ``DQ = β₀ + β₁·U + Σⱼ β₁₊ⱼ·max(U − cⱼ, 0)`` (plus a COVID intercept
    dummy when a ``covid`` column is present) and grid-searches the knot locations
    ``cⱼ`` to minimise SSE. Each segment's slope is the cumulative sum of the β's.
    """
    x = df[PREDICTOR].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    covid = df["covid"].to_numpy(dtype=float) if "covid" in df.columns else None
    lo, hi = np.quantile(x, [0.05, 0.95])
    candidates = np.linspace(lo, hi, n_candidates)

    def design(knots):
        cols = [np.ones_like(x)]
        if covid is not None:
            cols.append(covid)
        cols.append(x)
        cols += [np.maximum(x - c, 0) for c in knots]
        return np.column_stack(cols)

    best_sse: float | None = None
    best_knots = None
    best_beta = None

    for knots in combinations(candidates, n_knots):
        X = design(knots)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ beta) ** 2).sum())
        if best_sse is None or sse < best_sse:
            best_sse, best_knots, best_beta = sse, list(knots), beta

    X = design(best_knots)
    fitted = X @ best_beta
    residuals = y - fitted
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - best_sse / sst

    offset = 1 if covid is not None else 0
    return PiecewiseResult(
        knots=[float(c) for c in best_knots],
        intercept=float(best_beta[0]),
        covid_coef=float(best_beta[1]) if covid is not None else 0.0,
        slopes=[float(s) for s in np.cumsum(best_beta[1 + offset:])],
        r_squared=r2,
        sse=best_sse,
        fitted=pd.Series(fitted, index=df.index),
        residuals=pd.Series(residuals, index=df.index),
    )


def _ispline_basis(u: np.ndarray, knots: list[float]) -> np.ndarray:
    """I-spline basis (degree 1): non-negative, non-decreasing basis functions.

    Columns are g0..gk for knots c0..c_{k-1}; a non-negative coefficient vector
    yields a monotone non-decreasing piecewise-linear function.
    """
    k = len(knots)
    if k == 0:
        return np.asarray(u, dtype=float).reshape(-1, 1)
    cols = [np.minimum(u, knots[0])]
    for j in range(1, k):
        cols.append(np.maximum(0.0, np.minimum(u, knots[j]) - knots[j - 1]))
    cols.append(np.maximum(u - knots[-1], 0.0))
    return np.column_stack(cols)


def predict_piecewise(u, result: PiecewiseResult):
    """Evaluate the piecewise curve at unemployment value(s) `u`."""
    u = np.asarray(u, dtype=float)
    if len(result.knots) == 0:
        return result.intercept + result.slopes[0] * u
    return result.intercept + _ispline_basis(u, result.knots) @ np.asarray(result.slopes)


def fit_piecewise_monotone(df: pd.DataFrame, n_knots: int = 1, n_candidates: int = 18) -> PiecewiseResult:
    """Monotone non-decreasing piecewise-linear fit (all segment slopes ≥ 0).

    Reuses the unconstrained fit's knot locations, then refits the slopes with a
    non-negativity constraint on every segment via an I-spline basis and bounded
    least-squares (free intercept).
    """
    base = fit_piecewise(df, n_knots=n_knots, n_candidates=n_candidates)
    x = df[PREDICTOR].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    has_covid = "covid" in df.columns

    spline = _ispline_basis(x, base.knots)
    if has_covid:
        X = np.column_stack([np.ones_like(x), df["covid"].to_numpy(dtype=float), spline])
        lb = np.concatenate(([-np.inf, -np.inf], np.zeros(len(base.knots) + 1)))
    else:
        X = np.column_stack([np.ones_like(x), spline])
        lb = np.concatenate(([-np.inf], np.zeros(len(base.knots) + 1)))
    ub = np.full(X.shape[1], np.inf)
    beta = lsq_linear(X, y, bounds=(lb, ub), method="trf").x

    offset = 1 if has_covid else 0
    intercept = float(beta[0])
    covid_coef = float(beta[1]) if has_covid else 0.0
    slopes = [float(s) for s in beta[1 + offset:]]
    fitted = X @ beta
    residuals = y - fitted
    sse = float((residuals ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst

    return PiecewiseResult(
        knots=[float(c) for c in base.knots],
        intercept=intercept,
        covid_coef=covid_coef,
        slopes=slopes,
        r_squared=r2,
        sse=sse,
        fitted=pd.Series(fitted, index=df.index),
        residuals=pd.Series(residuals, index=df.index),
    )


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
    """Fit contemporaneous + piecewise models and persist params to JSON."""
    df = build_features(load_aligned()).dropna()
    full = df  # full sample — kept for the R² comparison
    if EXCLUDE_COVID:
        df = exclude_covid(df)
    contemp = fit_contemporaneous(df)
    piecewise = fit_piecewise(df)
    params = {
        "exclude_covid": EXCLUDE_COVID,
        "r2_full_sample": fit_contemporaneous(full).r_squared,
        "contemporaneous": contemp.params,
        "r2_contemporaneous": contemp.r_squared,
        "piecewise": {
            "knots": piecewise.knots,
            "slopes": piecewise.slopes,
            "r2_piecewise": piecewise.r_squared,
            "covid_coef": piecewise.covid_coef,
        },
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Fitted models -> {PARAMS_PATH}")
    return params


if __name__ == "__main__":
    fit_all()
