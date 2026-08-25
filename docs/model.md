# Consumer Delinquency vs Unemployment — Model Documentation

**Date:** 2026-08-24
**Target:** credit-card delinquency rate (unsecured consumer credit)
**Predictor:** unemployment rate
**Frequency:** quarterly

---

## 1. Purpose

Estimate the empirical relationship between the US unemployment rate and consumer
delinquency, for the purpose of **stress-testing an unsecured consumer loan
(fintech) book**. The deliverable is a mapping from an unemployment scenario to an
expected delinquency response, plus an exploration of lead/lag timing and the
shape of the relationship.

This is a *macro* transmission model: it answers "if unemployment rises by X pp,
what happens to delinquency?", not a portfolio-level loss forecast. It has no
loan-level, origination, or vintage information.

---

## 2. Data

| Series | FRED ID | Description | Frequency |
|---|---|---|---|
| Unemployment rate | `UNRATE` | Civilian unemployment rate, % (BLS) | monthly → quarterly mean |
| Delinquency (target) | `DRCCLACBS` | Delinquency rate on credit-card loans, all commercial banks, % (FFIEC) | quarterly |
| Delinquency (comparison) | `DRALACBS` | Delinquency rate on all loans, all commercial banks, % (FFIEC) | quarterly |

- Source: FRED (Federal Reserve Bank of St. Louis), plain-CSV download, no API key.
- `UNRATE` is resampled to a quarterly mean to match the delinquency series'
  FFIEC Call Report frequency; all series share a quarterly `Period("Q")` index.

### 2.1 Target choice — why credit cards

The model's consumer is an **unsecured** consumer lender (fintech personal loans /
point-of-sale credit). The default mechanism is a household income shock (job loss
→ inability to service an unsecured obligation). The closest FRED proxy for that
risk class is **credit-card delinquency**, not "all loans":

- "All loans" (`DRALACBS`) is dominated by **secured** mortgage and commercial
  lending, whose delinquency is driven by collateral values, house prices, and the
  business cycle — a fundamentally different default process.
- Credit cards are **unsecured**, **consumer**, and **unemployment-sensitive**,
  matching the fintech book's risk profile.

The cost of this choice is statistical: unemployment explains credit-card
delinquency *less cleanly* (R² ≈ 0.72) than all-loans (R² ≈ 0.94), because
unsecured consumer default carries more idiosyncratic drivers (lending standards,
payment behaviour, charge-off policy) layered on top of the macro cycle. See §7.

---

## 3. Sample selection

Two sample decisions, both in `src/config.py`:

- **`START_DATE = "2000Q1"`** — the pre-2000 regime is noisier and dropped.
- **`EXCLUDE_COVID = True`** — the 8 quarters **2020Q1–2021Q4** are dropped from
  estimation. During this window delinquency was policy-distorted (stimulus,
  forbearance, payment holidays) and sat far off the unemployment relationship.
  Exclusion lifts the credit-card static R² from **0.42 → 0.72** — far more than an
  intercept-shift dummy can (credit-card delinquency *spiked* rather than shifting
  down uniformly, so a dummy is a poor model of the break).

With lag construction (`build_features` drops the first `LAG_QUARTERS` rows), the
estimation sample is **2001Q1 → 2026Q1, 93 quarters**.

---

## 4. Methodology

### 4.1 Static distributed-lag (level)

$$ DQ_t = \alpha + \sum_{i=0}^{4} \beta_i \, U_{t-i} + \varepsilon_t $$

Four quarters of unemployment history feed delinquency. The individual $\beta_i$
are **collinear** (unemployment is highly autocorrelated), so only their **sum**
— the long-run multiplier — is interpretable; individual lags oscillate in sign.

### 4.2 Dynamic (ARX)

$$ DQ_t = \alpha + \rho \, DQ_{t-1} + \sum_{i=0}^{4} \beta_i \, U_{t-i} + \varepsilon_t $$

Adds an AR(1) term for delinquency persistence.

### 4.3 Lead/lag

The empirical direction is *not assumed*. Two diagnostics:

- **Cross-correlation function (CCF)** over lags ±8 quarters. Sign convention:
  `CCF[k] = corr(U_{t+k}, DQ_t)`; a negative lag means unemployment leads.
- **Two-sided regression** with both leads and lags of unemployment.

### 4.4 Error-correction model (Engle–Granger, two-step)

1. Long-run (cointegrating) regression in levels: $DQ_t = \beta_0 + \beta_1 U_t + e_t$.
2. Short-run dynamics in first differences:
   $\Delta DQ_t = \alpha + \lambda \, e_{t-1} + \sum_{i=0}^{4} \gamma_i \, \Delta U_{t-i} + \varepsilon_t$.

$\lambda$ is the speed of adjustment; it must be **negative** for genuine error
correction (mean reversion to a long-run equilibrium).

### 4.5 Piecewise-linear (linear spline), monotone

$$ DQ = \beta_0 + \beta_1 U + \sum_j \beta_{1+j} \max(U - c_j, 0) $$

Knot locations $c_j$ are grid-searched to minimise SSE; the number of knots is
chosen by BIC. A **monotone** variant forces every segment slope ≥ 0 (via an
I-spline basis and bounded least squares), so delinquency never falls as
unemployment rises.

---

## 5. Results

Estimated on the 93-quarter sample (2000Q1+, COVID excluded). Credit-card target.

| Model | Statistic | Value |
|---|---|---|
| Static distributed-lag | R² | **0.718** |
| Static distributed-lag | long-run multiplier (Σ βᵢ) | **+0.157** pp DQ / 1pp U |
| Dynamic (ARX) | ρ | 0.935 |
| Dynamic (ARX) | R² | 0.982 |
| ECM | speed of adjustment λ | −0.011 |
| ECM | long-run β | +0.307 pp DQ / 1pp U |
| ECM (first-difference) | R² | 0.339 |
| Piecewise (monotone, 3 knots) | R² | 0.327 |

Piecewise knots at unemployment **6.0%, 6.4%, 8.8%**; segment slopes
**0.41, 0.0, 0.0, 2.04** — i.e. delinquency rises ~0.4pp per 1pp unemployment at
low levels, is flat through the 6–9% range, and steepens sharply above ~8.8%.

Static lag profile: β₀ +1.32, β₁ −0.60, β₂ +0.12, β₃ +0.31, β₄ −0.99 (collinear —
see §4.1).

---

## 6. Key findings

1. **The unemployment → credit-card-delinquency link is weaker and noisier than
   the aggregate book.** R² ≈ 0.72 vs ≈ 0.94 for all-loans. Unsecured consumer
   default has material idiosyncratic drivers beyond the macro cycle.

2. **No cointegration.** λ ≈ −0.01 is essentially zero, so delinquency and
   unemployment share no stable long-run equilibrium — they co-move over the cycle
   but drift apart in levels. The statistically sound specification is the
   **first-difference** model (ΔDQ on ΔU lags, R² ≈ 0.34).

3. **No clean lead/lag.** The level CCF rises monotonically toward long lags
   rather than peaking — a symptom of the credit-card series' secular downtrend,
   not evidence of a causal lead. This reinforces the preference for the
   first-difference read.

4. **The response is nonlinear at the tail.** The monotone piecewise fit is flat
   through the mid-range and steepens sharply above ~9% unemployment — consistent
   with "delinquency only blows up in a severe recession". This is the shape a
   stress test should lean on, not the linear slope.

---

## 7. Limitations & caveats

- **Proxy, not the actual book.** Credit cards are revolving; fintech personal
  loans are installment. The *unemployment sensitivity* is the relevant shared
  trait, but the level and severity of a specific fintech book will differ. The
  cleanest source (NY Fed / Equifax serious-delinquency by product, incl. a
  personal-loan bucket) is not on FRED.
- **Measurement distortion.** Credit-card *delinquency* is mechanically damped by
  charge-off timing — banks charge off cards aggressively in recessions, pulling
  accounts out of the delinquency pool. The credit-card **charge-off** rate
  (`CORCACBS`) is a cleaner unemployment signal (R² ≈ 0.78 vs 0.72) and worth a
  companion check.
- **Near unit root.** Delinquency is highly persistent (dynamic AR(1) ≈ 0.94), so
  the dynamic model's R² overstates predictive power; treat the level regression as
  a long-run relationship, not a forecast.
- **Small sample.** 93 quarterly observations; the tail (unemployment > 9%) is
  represented by only the 2008–09 and early-2020 recessions.
- **Single predictor.** Unemployment only, by design. Other drivers (rates,
  credit supply, fiscal policy) are omitted and can shift the relationship.

---

## 8. Reproducibility

```bash
python -m venv venv && venv\Scripts\activate
pip install -e ".[dev]"

dq-download && dq-process && dq-fit     # pull data → align → fit → persist params
pytest                                   # 17 tests
python -m src.export_html                # regenerate dashboard.html
```

- `dashboard.html` is a self-contained static file (open directly; no server).
- `streamlit run src/app.py` remains available as an interactive alternative
  (sliders for knots/horizon), superseded by the static HTML.
- `src/config.py` holds all sample and model choices: `TARGET`, `START_DATE`,
  `EXCLUDE_COVID`, `COVID_START`/`COVID_END`, `LAG_QUARTERS`.
