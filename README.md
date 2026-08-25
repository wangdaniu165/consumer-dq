# Consumer Delinquency vs Unemployment

A quarterly model relating US unsecured consumer (credit-card) delinquency to the
unemployment rate, with an empirical lead/lag analysis and a scenario stress-test
dashboard.

**Delivery:** a self-contained static **`dashboard.html`** (open in any browser, no
server). Full methodology is in **[`docs/model.md`](docs/model.md)**.

## Data

| Series | ID | Description | Frequency |
|---|---|---|---|
| Unemployment rate | `UNRATE` | Civilian Unemployment Rate, % (BLS) | Monthly → quarterly |
| Delinquency (target) | `DRCCLACBS` | Delinquency Rate on Credit Card Loans, All Commercial Banks, % | Quarterly |
| Delinquency (comparison) | `DRALACBS` | Delinquency Rate on All Loans, All Commercial Banks, % | Quarterly |
| Delinquency (comparison) | `DROCLACBS` | Delinquency Rate on Consumer Loans *ex-credit-card* (auto + personal), % | Quarterly |

Source: FRED (Federal Reserve Bank of St. Louis), downloaded per-series as CSV (no API key).
`UNRATE` is aggregated to quarterly mean; all series share a quarterly `Period("Q")` index.
The sample starts at `START_DATE = "2005Q1"` (`src/config.py`) — the pre-2005
regime is noisier and dropped.

The **ideal** target for a fintech unsecured book is the NY Fed / Equifax
"Other" serious-delinquency series (retail + personal installment), which is *not*
on FRED — `python -m src.download_nyfed` pulls it (90+ days, by loan type).

**Why credit card as the target:** the model is built to stress-test an *unsecured
consumer* (fintech) book. Credit-card delinquency is the closest FRED proxy for
that risk class — unsecured, consumer, unemployment-sensitive — whereas "all loans"
is dominated by secured mortgage/commercial credit with a different default
mechanism. The trade-off is a noisier unemployment fit (R² ≈ 0.80 vs 0.94 for all
loans), because unsecured consumer default carries more idiosyncratic drivers and
charge-off timing distorts the measured delinquency rate.

## Model

- **Static (distributed-lag):** `DQ_t = α + Σ βᵢ·U_{t-i}` for `i = 0..4` quarters.
- **Dynamic (ARX):** adds `ρ·DQ_{t-1}` for persistence.
- **Lead/lag:** cross-correlation (CCF) plus a two-sided regression report the
  empirical direction — the model does not assume unemployment leads.
- **Piecewise-linear (spline):** `DQ = β₀ + β₁·U + Σⱼ β₁₊ⱼ·max(U−cⱼ, 0)` with
  grid-searched knots (default 2 — BIC-optimal; 3 overfits a jumpy tail) and an optional
  **monotone constraint** (all segment slopes ≥ 0) via an I-spline basis + bounded
  least-squares, so delinquency never falls as unemployment rises.
- **COVID window:** 2020Q1–2021Q4 was policy-distorted (forbearance/stimulus).
  By default (`EXCLUDE_COVID = True` in `src/config.py`) these 8 quarters are
  dropped from estimation, lifting static R² from ~0.45 to ~0.80 for credit cards.
  Set it to `False` to keep the quarters and add an intercept-shift dummy instead
  (which helps little here — credit-card delinquency spiked rather than shifting
  down uniformly).

## Install & run

```bash
# Recommended: a per-project venv (house style), so the top-level `src`
# package doesn't collide with sibling projects' `src` packages.
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
pip install -e ".[dev]"

dq-download && dq-process && dq-fit   # pull data -> align -> fit models
dq-nyfed                              # pull NY Fed "Other" 90+ day delinquency (Equifax)
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

Delinquency is highly persistent (near unit root). The dynamic model's AR(1)
coefficient estimates at ~1.0, so its high R² overstates predictive power and the
level regression is best read as a long-run relationship. The static distributed-lag
fit on the estimation sample (2005Q1+, COVID excluded) gives R² ≈ 0.80 and is the
more conservative stress-test read.

An Engle-Granger error-correction model (see the Model tab) finds **no
cointegration**: the speed of adjustment λ estimates ≈ −0.01 (essentially zero),
so delinquency and unemployment share no stable long-run equilibrium — they
co-move over the cycle but drift apart in levels. The statistically sound
specification is the **first-difference model** ΔDQ on ΔU lags (R² ≈ 0.45 on the
estimation sample), with a long-run multiplier of ~0.33pp delinquency per 1pp
unemployment. Note also that the credit-card delinquency series has a secular
downtrend, so the level cross-correlation function drifts upward with lag rather
than showing a clean peak — another reason to prefer the first-difference read.
