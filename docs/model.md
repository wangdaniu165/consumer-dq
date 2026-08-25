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

- **Cross-correlation function (CCF)** over lags ±8 quarters:

  $$ \mathrm{CCF}(k) = \mathrm{corr}\!\left(U_{t+k},\, DQ_t\right),\qquad k = -8,\dots,+8 $$

  Sign convention: a **negative** $k$ means unemployment leads (moves first);
  a positive $k$ means delinquency leads.
- **Two-sided regression** with both leads and lags of unemployment:

  $$ DQ_t = \alpha + \sum_{i=1}^{4}\phi_i\, U_{t+i} + \sum_{i=0}^{4}\beta_i\, U_{t-i} + \varepsilon_t $$

  Large $\phi_i$ (future unemployment) would imply delinquency actually leads
  unemployment; large $\beta_i$ support the reverse.

### 4.4 Error-correction model (Engle–Granger, two-step)

1. Long-run (cointegrating) regression in levels: $DQ_t = \beta_0 + \beta_1 U_t + e_t$.
2. Short-run dynamics in first differences:
   $\Delta DQ_t = \alpha + \lambda \, e_{t-1} + \sum_{i=0}^{4} \gamma_i \, \Delta U_{t-i} + \varepsilon_t$.

$\lambda$ is the speed of adjustment; it must be **negative** for genuine error
correction (mean reversion to a long-run equilibrium).

### 4.5 Piecewise-linear (linear spline), monotone

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

---

## 5. Results

Estimated on the 73-quarter sample (2005Q1+, COVID excluded). Target is the NY Fed
"Other" 90+ day serious delinquency (`NYFED_OTHER_90DPD`, %). Standard errors are
ordinary OLS.

| Model | R² | adj. R² | n |
|---|---|---|---|
| Static distributed-lag | 0.515 | 0.479 | 73 |
| Dynamic (ARX) | 0.952 | 0.947 | 73 |
| ECM — long-run (levels) | 0.508 | 0.501 | 73 |
| ECM — short-run (first diffs) | 0.266 | 0.194 | 68 |
| Piecewise (monotone, 2 knots) | 0.559 | — | 73 |

### 5.1 Static distributed-lag (levels)

$$\widehat{DQ}_t = 5.078 + 0.260\,U_t + 0.252\,U_{t-1} - 0.251\,U_{t-2} + 0.459\,U_{t-3} - 0.186\,U_{t-4}$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +5.078 | 0.430 | +11.82 | 0.000 |
| u_lag0 (U_t) | +0.260 | 0.661 | +0.39 | 0.695 |
| u_lag1 | +0.252 | 1.247 | +0.20 | 0.841 |
| u_lag2 | −0.251 | 1.309 | −0.19 | 0.849 |
| u_lag3 | +0.459 | 1.212 | +0.38 | 0.706 |
| u_lag4 | −0.186 | 0.620 | −0.30 | 0.765 |

**Long-run multiplier** $\sum_i\beta_i = \mathbf{+0.534}$ pp DQ per 1pp U. Every
individual lag is insignificant (severe collinearity — §4.1), but the **sum** is a
stable, economically meaningful multiplier.

### 5.2 Dynamic (ARX)

$$\widehat{DQ}_t = 0.411 + 0.964\,DQ_{t-1} + 0.267\,U_t - 0.233\,U_{t-1} - 0.247\,U_{t-2} + 0.654\,U_{t-3} - 0.450\,U_{t-4}$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +0.411 | 0.235 | +1.75 | 0.084 |
| dq_lag1 (ρ) | +0.964 | 0.039 | +24.44 | 0.000 |
| u_lag0 | +0.267 | 0.210 | +1.27 | 0.207 |
| u_lag1 | −0.233 | 0.397 | −0.59 | 0.559 |
| u_lag2 | −0.247 | 0.416 | −0.59 | 0.556 |
| u_lag3 | +0.654 | 0.385 | +1.70 | 0.095 |
| u_lag4 | −0.450 | 0.197 | −2.28 | 0.026 |

ρ ≈ 0.96 dominates — serious delinquency is near a unit root, so this R² overstates
predictive power (§7).

### 5.3 Error-correction model (Engle–Granger)

**Long-run (levels):** $\ \widehat{DQ}_t = 5.131 + 0.526\,U_t$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +5.131 | 0.369 | +13.91 | 0.000 |
| UNRATE | +0.526 | 0.061 | +8.56 | 0.000 |

**Short-run (first differences):**

$$\Delta\widehat{DQ}_t = 0.068 - 0.057\,e_{t-1} + 0.408\,\Delta U_t - 0.041\,\Delta U_{t-1} - 0.320\,\Delta U_{t-2} + 0.231\,\Delta U_{t-3} + 0.364\,\Delta U_{t-4}$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +0.068 | 0.041 | +1.67 | 0.100 |
| ECT_lag1 (λ) | −0.057 | 0.046 | −1.25 | 0.215 |
| du_lag0 | +0.408 | 0.202 | +2.02 | 0.047 |
| du_lag1 | −0.041 | 0.243 | −0.17 | 0.865 |
| du_lag2 | −0.320 | 0.241 | −1.33 | 0.189 |
| du_lag3 | +0.231 | 0.240 | +0.96 | 0.339 |
| du_lag4 | +0.364 | 0.201 | +1.82 | 0.074 |

λ = −0.057 is negative but insignificant (p = 0.22) → **no strong cointegration**.
Only the contemporaneous ΔU (du_lag0) is individually significant.

### 5.4 Piecewise-linear (monotone, 2 knots)

Knots at U = **4.29%, 4.64%**; intercept **7.295**; segment slopes
**[0.0, 0.0, 0.623]**:

$$\widehat{DQ}(u) = 7.295 + 0.0\,g_0(u) + 0.0\,g_1(u) + 0.623\,g_2(u)$$

i.e. delinquency is flat below ~4.6% and rises ~0.62pp/pp above it — the two knots
collapse to a single effective kink. R² = 0.559 (better than the contemporaneous
linear fit, but below the lagged models).

### 5.5 Lead/lag

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

---

## 6. Key findings

1. **Higher level, steeper slope.** The NY Fed "Other" series averages ~8% (vs ~3%
   for the FFIEC series) and responds **+0.53pp per 1pp unemployment** — more than
   2× the credit-card slope (+0.19). Serious unsecured delinquency is both larger
   and more unemployment-sensitive.

2. **But the fit is noisier.** Unemployment explains only 0.52 of serious
   delinquency (vs 0.80 for credit cards) — 90+ day serious delinquency is more
   persistent and carries more idiosyncratic drivers.

3. **No cointegration.** λ = −0.057 is negative but insignificant (p = 0.22), so
   delinquency and unemployment share no stable long-run equilibrium. The sound
   specification is the **first-difference** model (R² ≈ 0.27).

4. **Level CCF peaks at −1** — unemployment leads serious delinquency by one
   quarter, the economically sensible direction. In differences the lead/lag is
   bidirectional, with delinquency→unemployment the stronger Granger direction.

5. **The response is nearly linear above a low kink.** The monotone piecewise fit
   is flat below ~4.6% unemployment and rises ~0.62pp/pp above it — there is no
   steep "severe recession" tail, unlike the credit-card series.

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
- **Near unit root.** Delinquency is highly persistent (dynamic AR(1) ≈ 0.96), so
  the dynamic model's R² overstates predictive power; treat the level regression as
  a long-run relationship, not a forecast.
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
pytest                                   # 17 tests
python -m src.export_html                # regenerate dashboard.html
```

- `dashboard.html` is a self-contained static file (open directly; no server).
- `streamlit run src/app.py` remains available as an interactive alternative
  (sliders for knots/horizon), superseded by the static HTML.
- `src/config.py` holds all sample and model choices: `TARGET`, `START_DATE`,
  `EXCLUDE_COVID`, `COVID_START`/`COVID_END`, `LAG_QUARTERS`.
