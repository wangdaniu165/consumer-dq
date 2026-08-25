# Consumer Delinquency vs Unemployment

A quarterly model relating US consumer/loan delinquency to the unemployment rate,
with an empirical lead/lag analysis and a scenario stress-test dashboard.

## Data

| Series | ID | Description | Frequency |
|---|---|---|---|
| Unemployment rate | `UNRATE` | Civilian Unemployment Rate, % (BLS) | Monthly → quarterly |
| Delinquency (target) | `DRALACBS` | Delinquency Rate on All Loans, All Commercial Banks, % | Quarterly |
| Delinquency (comparison) | `DRCCLACBS` | Delinquency Rate on Credit Card Loans, % | Quarterly |

Source: FRED (Federal Reserve Bank of St. Louis), downloaded per-series as CSV (no API key).
`UNRATE` is aggregated to quarterly mean; all series share a quarterly `Period("Q")` index.

## Model

- **Static (distributed-lag):** `DQ_t = α + Σ βᵢ·U_{t-i}` for `i = 0..4` quarters.
- **Dynamic (ARX):** adds `ρ·DQ_{t-1}` for persistence.
- **Lead/lag:** cross-correlation (CCF) plus a two-sided regression report the
  empirical direction — the model does not assume unemployment leads.
- **Piecewise-linear (spline):** `DQ = β₀ + β₁·U + Σⱼ β₁₊ⱼ·max(U−cⱼ, 0)` with
  grid-searched knots (default 3, adjustable in the dashboard) and an optional
  **monotone constraint** (all segment slopes ≥ 0) via an I-spline basis + bounded
  least-squares, so delinquency never falls as unemployment rises.
- **COVID dummy:** all regressions include an intercept-shift dummy for
  2020Q1–2021Q4, when delinquency was policy-distorted (forbearance/stimulus) —
  the coefficient estimates ~−2.2pp, and recovers most of the fit lost to that
  structural break.

## Install & run

```bash
# Recommended: a per-project venv (house style), so the top-level `src`
# package doesn't collide with sibling projects' `src` packages.
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
pip install -e ".[dev]"

dq-download && dq-process && dq-fit   # pull data -> align -> fit models
python -m streamlit run src/app.py    # dashboard
pytest                                # tests
```

> **Note on the `src` package:** this project uses a top-level `src/` package
> (same as `eu-hpi` and the other siblings). If you run it in the *shared*
> environment where `eu-hpi` is editable-installed, `import src` can resolve to
> `eu-hpi/src` instead of this project's. Use a venv, or launch from this
> directory with `python -m streamlit run src/app.py` so the project root takes
> precedence on `sys.path`.

## Caveat

Delinquency is highly persistent (near unit root). The dynamic model's AR(1)
coefficient estimates at ~1.0, so its high R² overstates predictive power and the
level regression is best read as a long-run relationship. The static distributed-lag
fit (R² ≈ 0.72) is the more conservative stress-test read.

An Engle-Granger error-correction model (see the Model tab) finds **no
cointegration**: the speed of adjustment λ estimates positive (~+0.03) rather than
negative, so delinquency and unemployment share no stable long-run equilibrium —
they co-move over the cycle but drift apart in levels. The statistically sound
specification is the **first-difference model** ΔDQ on ΔU lags (R² ≈ 0.23),
with a long-run multiplier of ~0.74pp delinquency per 1pp unemployment.
