# Consumer DQ vs Unemployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that models US consumer/loan delinquency (`DRALACBS`) as a distributed-lag function of unemployment (`UNRATE`), with empirical lead/lag analysis and scenario stress-testing.

**Architecture:** `src/` package following `eu-hpi` house style: `config.py` → `download.py` (FRED CSV) → `process.py` (align + lag features) → `model.py` (OLS + CCF) → `project.py` (scenario projection) → `app.py` (Streamlit). All modules are plain functions with no hidden state; data flows through `data/raw/` and `data/processed/`.

**Tech Stack:** Python 3.12, pandas, numpy, statsmodels, requests, streamlit, plotly, pytest.

## Global Constraints

- Python `>=3.11`.
- Dependencies (from spec §2/§4): `pandas>=2.0`, `numpy>=1.24`, `statsmodels>=0.14`, `requests>=2.28`, `streamlit>=1.35`, `plotly>=5.0`.
- FRED series IDs: `UNRATE`, `DRALACBS` (target), `DRCCLACBS` (comparison). Source URL pattern: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ids>`.
- Default lag length: `LAG_MONTHS = 12` (overridable in `config.py`).
- Default train/test split: holdout = last 60 months (`HOLDOUT_MONTHS = 60`).
- House style: `src/` package, `[project.scripts]` entry points, `data/raw` + `data/processed`, pytest with a `tests/` package.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | deps + entry points `dq-download`, `dq-process`, `dq-fit`, `dq-project` |
| `src/config.py` | FRED URL/IDs, paths, `LAG_MONTHS`, `HOLDOUT_MONTHS`, scenario presets |
| `src/download.py` | fetch FRED CSVs → `data/raw/` |
| `src/process.py` | align monthly index, build lag features, train/test split |
| `src/model.py` | fit static + dynamic OLS, CCF, lead/lag diagnostic |
| `src/project.py` | scenario → projected delinquency path |
| `src/app.py` | Streamlit dashboard (4 tabs) |
| `tests/*` | pytest coverage per module |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/__init__.py`, `src/config.py`, `tests/__init__.py`

**Interfaces:**
- Produces: `config` constants used by every later module — `RAW_DIR`, `PROCESSED_DIR`, `FRED_URL`, `SERIES_IDS`, `LAG_MONTHS`, `HOLDOUT_MONTHS`, `SCENARIOS`.

- [ ] **Step 1: Write `config.py`**

```python
"""Configuration constants for consumer DQ vs unemployment model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SERIES_IDS = ["UNRATE", "DRALACBS", "DRCCLACBS"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(SERIES_IDS)

RAW_PATH = RAW_DIR / "fred_raw.csv"
ALIGNED_PATH = PROCESSED_DIR / "aligned.csv"

TARGET = "DRALACBS"
PREDICTOR = "UNRATE"
LAG_MONTHS = 12
HOLDOUT_MONTHS = 60

SCENARIOS = {
    "baseline": {"kind": "hold"},
    "moderate": {"kind": "step", "delta": 2.0},
    "severe": {"kind": "step", "delta": 5.0},
}
```

- [ ] **Step 2: Write `pyproject.toml`** — mirror `eu-hpi`: setuptools backend, name `consumer-dq`, `requires-python >=3.11`, dependencies from Global Constraints, and `[project.scripts]`:

```toml
[project.scripts]
dq-download = "src.download:download_all"
dq-process = "src.process:process_all"
dq-fit = "src.model:fit_all"
dq-project = "src.project:project_scenarios"
```

- [ ] **Step 3: Write `.gitignore`** — ignore `data/raw/`, `__pycache__/`, `.venv/`, `.pytest_cache/`, `*.egg-info/`.
- [ ] **Step 4: Verify** `python -c "import src.config as c; print(c.LAG_MONTHS)"` prints `12`.
- [ ] **Step 5: Commit** — `feat: scaffold consumer-dq project`.

---

### Task 2: Data download

**Files:**
- Create: `src/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `config.FRED_URL`, `config.RAW_PATH`, `config.RAW_DIR`.
- Produces: `download_all(force: bool = False) -> None`; `download_fred(force: bool = False) -> None`. Writes a CSV with columns `DATE, UNRATE, DRALACBS, DRCCLACBS` to `RAW_PATH`.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from src import download, config

def test_download_fred_parses_columns(monkeypatch, tmp_path):
    csv = "DATE,UNRATE,DRALACBS,DRCCLACBS\n2020-01-01,3.5,1.5,2.5\n"
    monkeypatch.setattr(download, "RAW_PATH", tmp_path / "fred.csv")
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    class Resp:
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            return [csv.encode()]
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: Resp())
    download.download_fred(force=True)
    df = pd.read_csv(download.RAW_PATH)
    assert list(df.columns) == ["DATE", "UNRATE", "DRALACBS", "DRCCLACBS"]
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_download.py -v` → FAIL (`download` module missing).
- [ ] **Step 3: Implement `download.py`** — mirror `eu-hpi/src/download.py`: `requests.get(FRED_URL, stream=True, timeout=120)`, `raise_for_status()`, stream to file; skip if `RAW_PATH` exists and `not force`.
- [ ] **Step 4: Run to verify it passes**.
- [ ] **Step 5: Commit** — `feat: download FRED unemployment and delinquency data`.

---

### Task 3: Processing + lag features

**Files:**
- Create: `src/process.py`
- Test: `tests/test_process.py`

**Interfaces:**
- Consumes: `config.RAW_PATH`, `config.ALIGNED_PATH`, `config.LAG_MONTHS`, `config.HOLDOUT_MONTHS`, `config.TARGET`, `config.PREDICTOR`.
- Produces:
  - `load_aligned() -> pd.DataFrame` — monthly-indexed frame, columns `UNRATE`, `DRALACBS`, `DRCCLACBS`, no NaNs.
  - `build_features(df, lag_months=12) -> pd.DataFrame` — adds columns `u_lag0..u_lag{lag_months}` and `dq_lag1`.
  - `split_train_test(df, holdout=60) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from src import process

def _frame():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    return pd.DataFrame({"UNRATE": np.arange(24) + 4.0,
                         "DRALACBS": np.arange(24) + 1.0,
                         "DRCCLACBS": np.arange(24) + 2.0}, index=idx)

def test_build_features_shifts_correctly():
    df = process.build_features(_frame(), lag_months=2)
    assert list(df.columns) == ["UNRATE", "DRALACBS", "DRCCLACBS",
                                "u_lag0", "u_lag1", "u_lag2", "dq_lag1"]
    # u_lag0 equals current UNRATE; u_lag2 equals UNRATE shifted 2
    assert df["u_lag0"].iloc[2] == 6.0
    assert df["u_lag2"].iloc[2] == 4.0

def test_split_train_test_holdout():
    df = process.build_features(_frame(), lag_months=2).dropna()
    train, test = process.split_train_test(df, holdout=6)
    assert len(train) + len(test) == len(df)
    assert len(test) == 6
```

- [ ] **Step 2: Run to verify it fails**.
- [ ] **Step 3: Implement `process.py`** — `load_aligned` reads CSV, parses `DATE` to monthly `PeriodIndex`/`DatetimeIndex`, drops NaNs. `build_features` uses `df[PREDICTOR].shift(i)` for lags and `df[TARGET].shift(1)`. `split_train_test` slices the last `holdout` rows.
- [ ] **Step 4: Run to verify it passes**.
- [ ] **Step 5: Commit** — `feat: align data and build lag features`.

---

### Task 4: Model — OLS + lead/lag diagnostics

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `process.build_features`, `config.LAG_MONTHS`, `config.TARGET`.
- Produces:
  - `fit_static(df, lag_months) -> FitResult` — OLS of `DRALACBS` on `u_lag0..u_lag{lag_months}`.
  - `fit_dynamic(df, lag_months) -> FitResult` — same plus `dq_lag1` regressor.
  - `FitResult` — dataclass with `params: dict[str, float]`, `r_squared: float`, `adj_r_squared: float`, `residuals: pd.Series`, `fitted: pd.Series`.
  - `compute_ccf(x: pd.Series, y: pd.Series, max_lag=24) -> dict[int, float]` — lag → Pearson correlation of `y_t` vs `x_{t-lag}` (negative lag = x leads y).
  - `lead_lag_diagnostic(df, lag_months) -> pd.DataFrame` — two-sided OLS of DQ on both leads and lags (`u_lead{1..12}` and `u_lag{0..12}`); returns a coefficient frame so the dashboard can show which side is significant.
  - `fit_all() -> None` — fit both, save params to `PROCESSED_DIR`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from src import model, process

def _df():
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    u = np.random.default_rng(0).normal(5, 1, 120)
    dq = 1.0 + 0.4 * np.roll(u, 3)  # delinquency responds to unemployment 3 months prior
    frame = pd.DataFrame({"UNRATE": u, "DRALACBS": dq, "DRCCLACBS": dq + 1}, index=idx)
    return process.build_features(frame, lag_months=6).dropna()

def test_fit_static_returns_r_squared():
    r = model.fit_static(_df(), lag_months=6)
    assert 0 < r.r_squared <= 1.0
    assert len(r.params) == 1 + 6 + 1  # intercept + 7 lag terms (u_lag0..u_lag6)

def test_ccf_peak_negative_lag():
    # x leads y by 3 -> peak at lag -3
    x = pd.Series(np.sin(np.arange(100) / 5))
    y = x.shift(3).fillna(0)
    ccf = model.compute_ccf(x, y, max_lag=6)
    assert max(ccf, key=ccf.get) == -3
```

- [ ] **Step 2: Run to verify it fails**.
- [ ] **Step 3: Implement `model.py`** — use `statsmodels.api.OLS(endog, exog).fit()`. `FitResult` dataclass wraps the fitted model. `compute_ccf` loops lag in `-max_lag..max_lag`, computes `y.corr(x.shift(lag))`, returns dict.
- [ ] **Step 4: Run to verify it passes**.
- [ ] **Step 5: Commit** — `feat: fit distributed-lag OLS and CCF diagnostics`.

---

### Task 5: Scenario projection

**Files:**
- Create: `src/project.py`
- Test: `tests/test_project.py`

**Interfaces:**
- Consumes: `model.fit_dynamic`, `config.SCENARIOS`, `config.LAG_MONTHS`.
- Produces: `project(model_result, unemployment_path: pd.Series, horizon: int) -> pd.Series` — recurses the dynamic model: `DQ_t = α + ρ·DQ_{t-1} + Σ βᵢ·U_{t-i}` using last-known values to seed lags.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from src import project

class FakeModel:
    def __init__(self):
        self.params = {"const": 1.0, "dq_lag1": 0.5}
        for i in range(13):
            self.params[f"u_lag{i}"] = 0.1

def test_project_recurses_ar_term():
    m = FakeModel()
    path = pd.Series([5.0] * 10)  # constant unemployment
    out = project.project(m, path, horizon=2)
    # t=0: 1 + 0.5*last_dq + 0.1*13*5 ; last_dq seeded = 0
    assert out.iloc[0] == pytest.approx(1.0 + 0.1 * 13 * 5.0)
    # t=1 reuses prior DQ through the AR term
    assert out.iloc[1] == pytest.approx(1.0 + 0.5 * out.iloc[0] + 0.1 * 13 * 5.0)
```

- [ ] **Step 2: Run to verify it fails**.
- [ ] **Step 3: Implement `project.py`** — seed lagged DQ with last fitted value; loop `horizon` steps applying the ARX recurrence; return a Series indexed 0..horizon-1.
- [ ] **Step 4: Run to verify it passes**.
- [ ] **Step 5: Commit** — `feat: scenario projection for stress testing`.

---

### Task 6: Streamlit dashboard

**Files:**
- Create: `src/app.py`

**Interfaces:**
- Consumes: `process.load_aligned`, `process.build_features`, `model.fit_static`, `model.fit_dynamic`, `model.compute_ccf`, `project.project`, `config.SCENARIOS`.
- Produces: a Streamlit app with tabs **Overview / Relationship / Model / Stress test**.

- [ ] **Step 1: Implement `app.py`** — four `st.tabs`, each rendering a Plotly chart from the module functions: Overview (both series), Relationship (scatter + rolling corr + CCF bar), Model (lag-profile bar + fit-vs-actual + residuals + static/dynamic stats), Stress test (sliders → `project.project` overlay).
- [ ] **Step 2: Smoke test** — `streamlit run src/app.py` renders without import errors (verify manually or via `python -c "import src.app"`).
- [ ] **Step 3: Commit** — `feat: Streamlit dashboard`.

---

### Task 7: README + integration

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml` (optional dev extra)

- [ ] **Step 1: Write `README.md`** — purpose, data sources, install (`pip install -e ".[dev]"`), run commands (`dq-download`, `dq-process`, `dq-fit`, `streamlit run src/app.py`), test (`pytest`).
- [ ] **Step 2: Full test run** — `pytest -q` all green.
- [ ] **Step 3: End-to-end run** — `dq-download && dq-process && dq-fit` succeed against live FRED.
- [ ] **Step 4: Commit** — `docs: README and final integration`.
