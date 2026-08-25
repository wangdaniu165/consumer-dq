"""Project delinquency forward under unemployment scenarios."""

import pandas as pd

from src.config import (
    LAG_QUARTERS,
    PREDICTOR,
    SCENARIOS,
    TARGET,
)
from src.model import FitResult, fit_dynamic
from src.process import build_features, load_aligned


def project(
    model: FitResult,
    df: pd.DataFrame,
    unemployment_path: pd.Series,
    horizon: int,
) -> pd.Series:
    """Recurse the dynamic model forward along a future unemployment path.

    Seeds the recurrence with the last delinquency and the last `LAG_QUARTERS`
    unemployment observations in `df` (the `build_features` output), then applies
    ``DQ_t = α + ρ·DQ_{t-1} + Σ βᵢ·U_{t-i}`` one quarter at a time.
    """
    if len(unemployment_path) < horizon:
        raise ValueError("unemployment_path shorter than horizon")

    params = model.params
    # seed: last LAG_QUARTERS unemployment observations (oldest → newest) + last DQ
    history = df[PREDICTOR].iloc[-LAG_QUARTERS:].tolist()
    dq_prev = df[TARGET].iloc[-1]

    buf = history[:]  # rolling window of the most recent unemployment values
    out: list[float] = []
    for t in range(horizon):
        buf.append(float(unemployment_path.iloc[t]))
        dq = params.get("const", 0.0)
        dq += params.get("dq_lag1", 0.0) * dq_prev
        for i in range(LAG_QUARTERS + 1):
            dq += params.get(f"u_lag{i}", 0.0) * buf[-1 - i]
        out.append(dq)
        dq_prev = dq
        buf = buf[1:]
    return pd.Series(out, dtype=float)


def build_scenario_path(last_value: float, scenario: dict, horizon: int) -> pd.Series:
    """Build a future unemployment path from a scenario preset."""
    kind = scenario.get("kind")
    if kind == "hold":
        return pd.Series([last_value] * horizon)
    if kind == "step":
        target = last_value + scenario.get("delta", 0.0)
        return pd.Series([target] * horizon)
    raise ValueError(f"unknown scenario kind {kind!r}")


def project_scenarios(horizon: int = 24) -> dict[str, pd.Series]:
    """Fit the dynamic model and project each scenario preset. CLI entry point."""
    df = build_features(load_aligned()).dropna()
    dynamic = fit_dynamic(df)
    last_value = df[PREDICTOR].iloc[-1]

    results: dict[str, pd.Series] = {}
    for name, scenario in SCENARIOS.items():
        path = build_scenario_path(last_value, scenario, horizon)
        results[name] = project(dynamic, df, path, horizon)
        print(f"{name}: last DQ {results[name].iloc[-1]:.2f}%")
    return results


if __name__ == "__main__":
    project_scenarios()
