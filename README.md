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

## Install & run

```bash
pip install -e ".[dev]"

dq-download          # pull FRED data -> data/raw/
dq-process           # align quarterly + lag features -> data/processed/
dq-fit               # fit static + dynamic models -> data/processed/model_params.json
dq-project           # project preset scenarios

streamlit run src/app.py   # dashboard
pytest                     # tests
```

## Caveat

Delinquency is highly persistent (near unit root). The dynamic model's AR(1)
coefficient estimates at ~1.0, so its high R² overstates predictive power and the
level regression is best read as a long-run relationship. The static distributed-lag
fit (R² ≈ 0.72) is the more conservative stress-test read. A first-difference or
error-correction specification is a natural extension.
