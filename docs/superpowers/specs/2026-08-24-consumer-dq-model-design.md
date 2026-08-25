# Consumer Delinquency vs Unemployment — Design Spec

**Date:** 2026-08-24
**Author:** Daniel Wang
**Status:** Approved (pending review)

## 1. Purpose

Build a model that relates **consumer/loan delinquency** to the **unemployment rate**, and expose it
through a Streamlit dashboard. The model serves two goals:

1. **Explore** — quantify the historical relationship (correlation, lead/lag structure, fit quality).
2. **Stress-test** — project delinquency forward under user-chosen unemployment scenarios (CCAR-style).

## 2. Data

**Source:** FRED (Federal Reserve Bank of St. Louis), CSV download endpoint — no API key required.

| Series | ID | Description | Frequency |
|---|---|---|---|
| Unemployment rate | `UNRATE` | Civilian Unemployment Rate, % | Monthly, SA |
| Delinquency (target) | `DRALACBS` | Delinquency Rate on All Loans, All Commercial Banks, % | Monthly, SA |
| Delinquency (secondary) | `DRCCLACBS` | Delinquency Rate on Credit Card Loans, All Commercial Banks, % | Monthly, SA |

- Align all series on a shared monthly index from the earliest common date (`DRALACBS` starts 1985-01).
- `DRALACBS` is the primary target; `DRCCLACBS` is shown in the dashboard as a comparison view only.

## 3. Model

### 3.1 Primary model — dynamic distributed-lag (ARX)

```
DQ_t = α + ρ · DQ_{t-1} + Σ_{i=0}^{12} βᵢ · U_{t-i} + ε_t
```

- 12 monthly lags of unemployment (one year), default per approval.
- AR(1) term (lagged DQ) captures delinquency persistence, default per approval.
- Estimated with OLS (`statsmodels`). Outputs: coefficients, t-stats, R² / adjusted R², Durbin-Watson.

### 3.2 Static variant (comparison)

```
DQ_t = α + Σ_{i=0}^{12} βᵢ · U_{t-i} + ε_t
```

Same specification without the AR term. Fit both and report side-by-side, so the dashboard can show
the interpretability vs. fit trade-off.

### 3.3 Lead/lag analysis (empirical direction)

The model must **not assume** unemployment leads delinquency. Determine direction from the data:

- **Cross-correlation function (CCF)** between unemployment and delinquency over lags `-24 … +24`
  (negative lag = unemployment leads). Report the peak lag and correlation.
- **Two-sided diagnostic regression** — include both leads and lags of unemployment
  (`U_{t-12} … U_t … U_{t+12}`) and report which side carries significant coefficients
  (Granger-causality direction).
- The stress-test model uses **12 lags by default**; the CCF peak is reported in the dashboard so the
  user can see the empirically-optimal lag and override the default via `config.py` (`LAG_MONTHS`).

## 4. Architecture

House style from `eu-hpi`: `src/` package, `config.py` → download → process → model → project → app.

```
consumer-dq/
├── pyproject.toml          # deps + entry points
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # FRED URLs, series IDs, paths, model hyperparams
│   ├── download.py         # pull UNRATE/DRALACBS/DRCCLACBS from FRED → data/raw/
│   ├── process.py          # align index, build lag features, train/test split (default holdout: last 60 months)
│   ├── model.py            # fit static + dynamic OLS, lead/lag + CCF diagnostics
│   ├── project.py          # scenario → projected delinquency path
│   └── app.py              # Streamlit dashboard (4 tabs)
├── data/
│   ├── raw/                # downloaded CSVs
│   └── processed/          # aligned/lagged frame + fitted model params
└── tests/
    ├── test_download.py
    ├── test_process.py
    ├── test_model.py
    └── test_project.py
```

### Entry points (`pyproject.toml`)

- `dq-download` → `src.download:download_all`
- `dq-process` → `src.process:process_all`
- `dq-fit` → `src.model:fit_all`
- `dq-project` → `src.project:project_scenarios`
- dashboard run via `streamlit run src/app.py`

## 5. Scenario projection

`project.py` maps a user-supplied unemployment path to a delinquency forecast using the fitted
dynamic model (recursing through the AR term):

- **Presets:** baseline (unemployment held at last value), moderate (+2pp step), severe (+5pp step,
  mirroring 2008/2020 peak).
- **Custom:** dashboard sliders set unemployment per future quarter.
- Returns a DataFrame of projected delinquency with the scenario path attached.

## 6. Dashboard (`app.py`)

Four tabs:

1. **Overview** — both series over time (with `DRCCLACBS` comparison).
2. **Relationship** — scatter (U vs DQ), rolling correlation, CCF lead/lag chart.
3. **Model** — coefficient lag-profile bar chart, fit-vs-actual, residuals, static vs. dynamic stats.
4. **Stress test** — unemployment scenario controls → projected delinquency path overlay.

## 7. Error handling

- Download: `requests` with timeout; `raise_for_status()`; cache and skip if file exists (mirror
  `eu-hpi`).
- Process: log and drop/forward-fill missing observations; assert shared index alignment.
- Model: guard against singular matrices / insufficient data; surface clean errors, not stack traces.
- Project: validate scenario length and unemployment bounds.

## 8. Testing (pytest)

- `test_download.py` — CSV parse into expected columns; cache-skip behavior (mocked).
- `test_process.py` — index alignment, lag-feature construction (correct shift), train/test split.
- `test_model.py` — fit reproducibility (fixed seed/data), coefficient shape, CCF computation.
- `test_project.py` — projection recursion matches manual computation; scenario bounds.

## 9. Out of scope (YAGNI)

- Multi-factor macro model (GDP, HPI, inflation) — single predictor per "use only".
- Multi-geography (non-US) data.
- Non-linear / ML models (random forest, etc.).
- Production scheduling / auto-refresh jobs.
