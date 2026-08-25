# Consumer Delinquency vs Unemployment

A quarterly model relating US unsecured consumer delinquency (NY Fed/Equifax
"Other" 90+ day serious delinquency) to the unemployment rate, with an empirical
lead/lag analysis and a scenario stress-test dashboard.

**Delivery:** a self-contained static **`dashboard.html`** (open in any browser, no
server). Full methodology is in **[`docs/model.md`](docs/model.md)**.

## Data

| Series | ID | Description | Frequency |
|---|---|---|---|
| Unemployment rate | `UNRATE` | Civilian Unemployment Rate, % (BLS) | Monthly → quarterly |
| Delinquency (target) | `NYFED_OTHER_90DPD` | "Other" loans 90+ days delinquent, % (NY Fed / Equifax) | Quarterly |
| Delinquency (comparison) | `DROCLACBS` | Delinquency Rate on Consumer Loans ex-credit-card, % (FFIEC) | Quarterly |

Source: FRED for `UNRATE`/`DROCLACBS` (plain CSV, no API key); the NY Fed
Household Debt & Credit report for the target (Equifax, **not** on FRED — pulled by
`dq-download`). `UNRATE` is aggregated to quarterly mean; all series share a
quarterly `Period("Q")` index. The sample starts at `START_DATE = "2005Q1"`.

**Why the NY Fed "Other" as the target:** the model stress-tests an *unsecured
consumer* (fintech) book. The NY Fed's "Other" category (retail + personal
installment + other consumer credit) is the closest published proxy — unsecured
and non-revolving, unlike revolving credit cards and secured auto/mortgage. Its
level is far higher than the FFIEC series (≈8% vs ≈3%) because it measures *90+
day serious* delinquency on riskier, unsecured balances.

## Model

- **Contemporaneous (levels):** `DQ_t = α + β·U_t` — the parsimonious specification
  (unemployment lags are ~95% collinear, so only the contemporaneous slope is
  cleanly identified; β ≈ +0.53, t ≈ 8.6).
- **Lead/lag:** cross-correlation (CCF) plus a two-sided regression report the
  empirical direction — the model does not assume unemployment leads.
- **Piecewise-linear (spline):** `DQ = β₀ + β₁·U + Σⱼ β₁₊ⱼ·max(U−cⱼ, 0)` with a
  single knot and a **monotone constraint** (all segment slopes ≥ 0) via an
  I-spline basis + bounded least-squares, so delinquency never falls as unemployment
  rises.
- **COVID window:** 2020Q1–2021Q4 was policy-distorted (forbearance/stimulus).
  By default (`EXCLUDE_COVID = True` in `src/config.py`) these 8 quarters are
  dropped from estimation — for the NY Fed "Other" target this barely moves the fit,
  because unsecured personal/retail debt was less policy-distorted than credit cards.

## Install & run

```bash
# Recommended: a per-project venv (house style), so the top-level `src`
# package doesn't collide with sibling projects' `src` packages.
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
pip install -e ".[dev]"

dq-download && dq-process && dq-fit   # pull data (incl. NY Fed) -> align -> fit models
python -m src.export_html             # generate dashboard.html (static, no server)
python -m src.export_model_html       # generate model.html (model doc, math rendered)
python -m streamlit run src/app.py    # interactive dashboard (optional)
pytest                                # tests
```

> **Note on the `src` package:** this project uses a top-level `src/` package
> (same as `eu-hpi` and the other siblings). If you run it in the *shared*
> environment where `eu-hpi` is editable-installed, `import src` can resolve to
> `eu-hpi/src` instead of this project's. Use a venv, or launch from this
> directory with `python -m streamlit run src/app.py` so the project root takes
> precedence on `sys.path`.

## Caveat

Delinquency is highly persistent (near unit root), so the level regression is best
read as a long-run relationship, not a forecast. The contemporaneous fit on the
estimation sample (2005Q1+, COVID excluded) gives R² ≈ 0.51 with a slope of ~0.53pp
delinquency per 1pp unemployment (t ≈ 8.6).
