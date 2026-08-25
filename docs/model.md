# Consumer Delinquency vs Unemployment — Model Documentation

**Date:** 2026-08-24
**Target:** NY Fed/Equifax "Other" 90+ day serious delinquency (unsecured consumer)
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

| Series | ID | Description | Frequency |
|---|---|---|---|
| Unemployment rate | `UNRATE` | Civilian unemployment rate, % (BLS) | monthly → quarterly mean |
| Delinquency (target) | `NYFED_OTHER_90DPD` | "Other" loans 90+ days delinquent, % (NY Fed / Equifax) | quarterly |
| Delinquency (comparison) | `DROCLACBS` | Delinquency rate on consumer loans ex-credit-card, % (FFIEC) | quarterly |

- Source: FRED (`UNRATE`, `DROCLACBS`) plus the NY Fed Household Debt & Credit
  report (`NYFED_OTHER_90DPD`, via `dq-nyfed` — Equifax-based, not on FRED).
- `UNRATE` is resampled to a quarterly mean; all series share a quarterly
  `Period("Q")` index.

### 2.1 Target choice — why the NY Fed "Other" series

The model stress-tests an **unsecured consumer** (fintech personal loans / POS
credit) book. The closest published proxy is the NY Fed / Equifax **"Other"**
category — retail + personal installment + other consumer credit — which is both
**unsecured** and **non-revolving**, matching the fintech book better than:

- **Credit cards** (`DRCCLACBS`) — unsecured but *revolving*; delinquency accrues
  and cures differently from an installment personal loan.
- **All loans** (`DRALACBS`) — dominated by *secured* mortgage/commercial credit
  with a collateral-driven default process.

`DROCLACBS` (consumer loans ex-credit-card) is kept as a comparison: non-revolving
(installment-like) but still containing secured auto loans.

The NY Fed "Other" series runs far higher than the FFIEC series (~8% vs ~3%)
because it measures **90+ day serious** delinquency on riskier, unsecured balances.

---

## 3. Sample selection

Two sample decisions, both in `src/config.py`:

- **`START_DATE = "2005Q1"`** — the pre-2005 regime is noisier and dropped.
- **`EXCLUDE_COVID = True`** — the 8 quarters **2020Q1–2021Q4** are dropped from
  estimation. For the NY Fed "Other" series this barely moves the fit (static R²
  **0.508 → 0.515**), unlike the earlier credit-card target where exclusion mattered
  greatly — unsecured personal/retail delinquency was less policy-distorted than
  credit cards (no forbearance), so the 2020–21 points sit closer to the line.

With lag construction (`build_features` drops the first `LAG_QUARTERS` rows), the
estimation sample is **2006Q1 → 2026Q1, 73 quarters**.

---

## 4. Methodology

### 4.1 Contemporaneous (levels)

$$ DQ_t = \alpha + \beta\,U_t + \varepsilon_t $$

The parsimonious specification: unemployment lags are ~95% collinear, so their
individual coefficients are unidentified, but the contemporaneous slope is cleanly
estimated.

### 4.2 Lead/lag

The empirical direction is *not assumed*. Two diagnostics:

- **Cross-correlation function (CCF)** over lags ±8 quarters:

  $$ \mathrm{CCF}(k) = \mathrm{corr}\!\left(U_{t+k},\, DQ_t\right),\qquad k = -8,\dots,+8 $$

  Sign convention: a **negative** $k$ means unemployment leads (moves first);
  a positive $k$ means delinquency leads.
- **Two-sided regression** with both leads and lags of unemployment:

  $$ DQ_t = \alpha + \sum_{i=1}^{4}\phi_i\, U_{t+i} + \sum_{i=0}^{4}\beta_i\, U_{t-i} + \varepsilon_t $$

  Large $\phi_i$ (future unemployment) would imply delinquency actually leads
  unemployment; large $\beta_i$ support the reverse.

### 4.3 Piecewise-linear (linear spline), monotone

$$ DQ = \beta_0 + \beta_1 U + \sum_j \beta_{1+j} \max(U - c_j, 0) $$

Knot locations $c_j$ are grid-searched to minimise SSE; the number of knots is
chosen by **BIC**:

$$ \mathrm{BIC} = n\ln\!\left(\tfrac{\mathrm{SSE}}{n}\right) + k\ln n,\qquad
k = n_{\text{knots}} + 2 $$

A **monotone** variant forces every segment slope ≥ 0, via an **I-spline basis**
(degree-1, non-negative, non-decreasing) and bounded least squares, so delinquency
never falls as unemployment rises. For knots $c_1 < \dots < c_K$:

$$ g_0(u) = \min(u, c_1),\qquad
   g_j(u) = \max\!\left(0,\ \min(u, c_{j+1}) - c_j\right)\ (1\le j < K),\qquad
   g_K(u) = \max(u - c_K, 0) $$

then $DQ = \theta_0 + \sum_{j=0}^{K}\theta_j\, g_j(u)$ with $\theta_j \ge 0$;
each segment slope is the cumulative sum of the $\theta_j$.

### 4.4 First-difference (stationary)

$$ \Delta DQ_t = \alpha + \beta\,\Delta U_t + \gamma\,\Delta DGS10_t + \varepsilon_t $$

The statistically sound specification: differencing removes the unit root, so the
inference is not spurious. $\Delta DGS10$ is the change in the 10-year Treasury — a
forward-looking recession signal (negative sign: rates fall when the economy
weakens, which is when delinquency rises).

---

## 5. Results

Estimated on the 73-quarter sample (2005Q1+, COVID excluded). Target is the NY Fed
"Other" 90+ day serious delinquency (`NYFED_OTHER_90DPD`, %). Standard errors are
ordinary OLS.

| Model | R² | adj. R² | n |
|---|---|---|---|
| Contemporaneous (levels) | 0.508 | 0.501 | 73 |
| Piecewise (monotone, 1 knot) | 0.559 | — | 73 |
| First-difference (ΔU + ΔDGS10) | 0.210 | 0.187 | 72 |

### 5.1 Contemporaneous (levels)

$$\widehat{DQ}_t = 5.131 + 0.526\,U_t$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +5.131 | 0.369 | +13.91 | 0.000 |
| U_t | +0.526 | 0.061 | +8.56 | 0.000 |

The slope is cleanly estimated (t = 8.6, p < 0.001); R² = 0.508.

### 5.2 Piecewise-linear (monotone, 1 knot)

Knot at U = **4.64%**; intercept **7.295**; segment slopes **[0.0, 0.623]**:

$$\widehat{DQ}(u) = 7.295 + 0.0\,g_0(u) + 0.623\,g_1(u)$$

i.e. delinquency is flat below ~4.6% unemployment and rises ~0.62pp/pp above it.
BIC prefers 1 knot over 0 (7.7 vs 15.3). R² = 0.559.

### 5.3 Lead/lag

**Level CCF** peaks at **−1** (corr 0.72) — unemployment leads delinquency by one
quarter, a sensible direction (unlike the spurious +8 the credit-card series gave).

**First-difference CCF** (ΔDQ vs ΔU) peaks at **+1 (0.50)**:

| lag | corr |
|---|---|
| −1 | +0.30 |
| 0 | +0.35 |
| **+1** | **+0.50** |
| +2 | +0.38 |

**Granger causality** (does lagged ΔX help predict Y; p-values, F-test):

| direction | lag 1 | lag 2 | lag 3 | lag 4 |
|---|---|---|---|---|
| **ΔDQ → ΔU** (delinquency leads unemployment) | **0.001** | **0.006** | **0.012** | **0.005** |
| ΔU → ΔDQ (unemployment leads delinquency) | 0.020 | 0.158 | 0.022 | 0.309 |

Lagged delinquency changes significantly predict unemployment at every lag; lagged
unemployment predicts delinquency only weakly (lags 1 and 3). The evidence is
**bidirectional**, with delinquency→unemployment the stronger direction.

Two-sided regression (leads and lags of U, levels):

| term | coef |
|---|---|
| const | +5.193 |
| u_lead1 … u_lead4 | +0.687, −0.183, +1.129, −1.037 |
| u_lag0 … u_lag4 | −0.417, +0.050, −0.072, +0.482, −0.127 |

### 5.4 First-difference (stationary)

$$\Delta\widehat{DQ}_t = 0.055 + 0.377\,\Delta U_t - 0.321\,\Delta DGS10_t$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +0.055 | 0.040 | +1.38 | 0.172 |
| ΔU_t | +0.377 | 0.131 | +2.87 | 0.005 |
| ΔDGS10 | −0.321 | 0.115 | −2.78 | 0.007 |

R² = 0.210. Both predictors are significant, and ΔDGS10's **negative** sign is a
risk-off signal — the 10-year Treasury falls when the economy weakens (which is when
delinquency rises), not a borrowing-cost effect. The actual consumer lending rates
(credit-card APR, personal-loan rate) were tested and add nothing.

---

## 6. Key findings

1. **Higher level, steeper slope.** The NY Fed "Other" series averages ~8% (vs ~3%
   for the FFIEC series) and responds **+0.53pp per 1pp unemployment** — more than
   2× the credit-card slope (+0.19). Serious unsecured delinquency is both larger
   and more unemployment-sensitive.

2. **But the fit is noisier.** Unemployment explains only 0.52 of serious
   delinquency (vs 0.80 for credit cards) — 90+ day serious delinquency is more
   persistent and carries more idiosyncratic drivers.

3. **The response is nearly linear above a low kink.** The single-knot piecewise
   fit is flat below ~4.6% unemployment and rises ~0.62pp/pp above it — there is no
   steep "severe recession" tail, unlike the credit-card series.

4. **Level CCF peaks at −1** — unemployment leads serious delinquency by one
   quarter, the economically sensible direction. In differences the lead/lag is
   bidirectional, with delinquency→unemployment the stronger Granger direction.

---

## 7. Limitations & caveats

- **Proxy, not the actual book.** The NY Fed "Other" category is a grab-bag (retail
  + personal installment + other consumer credit), so it is still not a *pure*
  personal-loan series — and it was reclassified historically, which can create
  level shifts. The *unemployment sensitivity* is the relevant shared trait, but a
  specific fintech book's level and severity will differ.
- **90+ day threshold.** The target is *serious* (90+ day) delinquency, which is
  more persistent and "sticky" than the 30+ day FFIEC measures. That is a feature
  for severity, but it also means the series responds more slowly to the
  unemployment cycle.
- **Near unit root.** Delinquency is highly persistent, so the level regression is
  best read as a long-run relationship, not a forecast — the high slope
  t-statistic is partly a spurious-regression artifact (no cointegration).
- **Small sample.** 73 quarterly observations; the tail (unemployment > 9%) is
  represented by only the 2008–09 and early-2020 recessions.
- **Single predictor.** Unemployment only, by design. Other drivers (rates,
  credit supply, fiscal policy) are omitted and can shift the relationship.

---

## 8. Reproducibility

```bash
python -m venv venv && venv\Scripts\activate
pip install -e ".[dev]"

dq-download && dq-process && dq-fit     # pull data → align → fit → persist params
pytest                                   # 12 tests
python -m src.export_html                # regenerate dashboard.html
```

- `dashboard.html` is a self-contained static file (open directly; no server).
- `streamlit run src/app.py` remains available as an interactive alternative.
- `src/config.py` holds all sample and model choices: `TARGET`, `START_DATE`,
  `EXCLUDE_COVID`, `COVID_START`/`COVID_END`.
