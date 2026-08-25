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
delinquency *less cleanly* (R² ≈ 0.80) than all-loans (R² ≈ 0.94), because
unsecured consumer default carries more idiosyncratic drivers (lending standards,
payment behaviour, charge-off policy) layered on top of the macro cycle. See §7.

---

## 3. Sample selection

Two sample decisions, both in `src/config.py`:

- **`START_DATE = "2005Q1"`** — the pre-2005 regime is noisier and dropped.
- **`EXCLUDE_COVID = True`** — the 8 quarters **2020Q1–2021Q4** are dropped from
  estimation. During this window delinquency was policy-distorted (stimulus,
  forbearance, payment holidays) and sat far off the unemployment relationship.
  Exclusion lifts the credit-card static R² from **0.45 → 0.80** — far more than an
  intercept-shift dummy can (credit-card delinquency *spiked* rather than shifting
  down uniformly, so a dummy is a poor model of the break).

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

Estimated on the 73-quarter sample (2005Q1+, COVID excluded). Credit-card target
(`DRCCLACBS`, %). Standard errors are ordinary OLS.

| Model | R² | adj. R² | n |
|---|---|---|---|
| Static distributed-lag | 0.796 | 0.781 | 73 |
| Dynamic (ARX) | 0.987 | 0.986 | 73 |
| ECM — long-run (levels) | 0.298 | 0.288 | 73 |
| ECM — short-run (first diffs) | 0.453 | 0.400 | 68 |
| Piecewise (monotone, 3 knots) | 0.447 | — | 73 |

### 5.1 Static distributed-lag (levels)

$$\widehat{DQ}_t = 2.319 + 1.346\,U_t - 0.494\,U_{t-1} - 0.103\,U_{t-2} + 0.291\,U_{t-3} - 0.855\,U_{t-4}$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +2.319 | 0.225 | +10.29 | 0.000 |
| u_lag0 (U_t) | +1.346 | 0.347 | +3.88 | 0.000 |
| u_lag1 | −0.494 | 0.654 | −0.76 | 0.453 |
| u_lag2 | −0.103 | 0.687 | −0.15 | 0.881 |
| u_lag3 | +0.291 | 0.636 | +0.46 | 0.649 |
| u_lag4 | −0.855 | 0.325 | −2.63 | 0.011 |

**Long-run multiplier** $\sum_i\beta_i = \mathbf{+0.185}$ pp DQ per 1pp U. Only the
contemporaneous term (u_lag0) and the 4th lag are individually significant; the
intermediate lags are collinear, so their signs oscillate (§4.1).

### 5.2 Dynamic (ARX)

$$\widehat{DQ}_t = 0.359 + 0.947\,DQ_{t-1} + 0.559\,U_t - 0.514\,U_{t-1} - 0.041\,U_{t-2} - 0.260\,U_{t-3} + 0.225\,U_{t-4}$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +0.359 | 0.084 | +4.26 | 0.000 |
| dq_lag1 (ρ) | +0.947 | 0.030 | +31.44 | 0.000 |
| u_lag0 | +0.559 | 0.091 | +6.14 | 0.000 |
| u_lag1 | −0.514 | 0.165 | −3.12 | 0.003 |
| u_lag2 | −0.041 | 0.173 | −0.24 | 0.813 |
| u_lag3 | −0.260 | 0.161 | −1.61 | 0.112 |
| u_lag4 | +0.225 | 0.089 | +2.53 | 0.014 |

ρ ≈ 0.95 dominates — delinquency is near a unit root, so this R² overstates
predictive power (§7).

### 5.3 Error-correction model (Engle–Granger)

**Long-run (levels):** $\ \widehat{DQ}_t = 1.420 + 0.326\,U_t$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | +1.420 | 0.356 | +3.99 | 0.000 |
| UNRATE | +0.326 | 0.059 | +5.49 | 0.000 |

**Short-run (first differences):**

$$\Delta\widehat{DQ}_t = -0.019 - 0.010\,e_{t-1} + 0.420\,\Delta U_t + 0.169\,\Delta U_{t-1} + 0.021\,\Delta U_{t-2} - 0.011\,\Delta U_{t-3} - 0.438\,\Delta U_{t-4}$$

| term | coef | std. err | t | p |
|---|---|---|---|---|
| const | −0.019 | 0.025 | −0.77 | 0.445 |
| ECT_lag1 (λ) | −0.010 | 0.053 | −0.19 | 0.848 |
| du_lag0 | +0.420 | 0.140 | +3.01 | 0.004 |
| du_lag1 | +0.169 | 0.150 | +1.13 | 0.264 |
| du_lag2 | +0.021 | 0.148 | +0.14 | 0.889 |
| du_lag3 | −0.011 | 0.147 | −0.08 | 0.938 |
| du_lag4 | −0.438 | 0.131 | −3.34 | 0.001 |

λ = −0.010 is indistinguishable from zero (p = 0.85) → **no cointegration**. Only the
contemporaneous ΔU (du_lag0) and the 4th lag (du_lag4) are significant.

### 5.4 Piecewise-linear (monotone, 3 knots)

Knots at U = **4.64%, 4.99%, 8.85%**; intercept **0.033**; segment slopes
**[0.681, 0.0, 0.034, 2.819]**:

$$\widehat{DQ}(u) = 0.033 + 0.681\,g_0(u) + 0.0\,g_1(u) + 0.034\,g_2(u) + 2.819\,g_3(u)$$

i.e. delinquency rises ~0.7pp per 1pp U below 4.6%, is flat through the 5–9% range,
and steepens sharply (+2.82pp/pp) above 8.9%. The flat middle is the monotone
constraint flooring the otherwise-negative mid-range slope. This is a shape
diagnostic, not a high-R² fit (R² = 0.447).

### 5.5 Lead/lag

**Level CCF** peaks at **+8** (corr 0.86) — the maximum lag allowed, i.e. it never
peaks but keeps rising: a spurious, trend-driven signature, not a real lead (§4.3).

**First-difference CCF** (ΔDQ vs ΔU) — stationary, peaks at **+1 (0.58)**:

| lag | corr |
|---|---|
| −1 | +0.31 |
| 0 | +0.47 |
| **+1** | **+0.58** |
| +2 | +0.41 |

**Granger causality** (does lagged ΔX help predict Y; p-values, F-test):

| direction | lag 1 | lag 2 | lag 3 | lag 4 |
|---|---|---|---|---|
| **ΔDQ → ΔU** (delinquency leads unemployment) | **0.001** | **0.001** | **0.003** | **0.005** |
| ΔU → ΔDQ (unemployment leads delinquency) | 0.560 | 0.413 | 0.170 | 0.004 |

Lagged delinquency changes significantly predict unemployment at every lag, while
lagged unemployment does **not** predict delinquency at short lags — so credit-card
delinquency is a **~1-quarter leading indicator** of the unemployment rate.

Two-sided regression (leads and lags of U, levels):

| term | coef |
|---|---|
| const | +1.465 |
| u_lead1 … u_lead4 | +0.183, −0.013, −0.600, +0.855 |
| u_lag0 … u_lag4 | +0.289, +0.142, +0.045, +0.339, −0.929 |

---

## 6. Key findings

1. **The unemployment → credit-card-delinquency link is weaker and noisier than
   the aggregate book.** R² ≈ 0.80 vs ≈ 0.94 for all-loans. Unsecured consumer
   default has material idiosyncratic drivers beyond the macro cycle.

2. **No cointegration.** λ ≈ −0.01 is essentially zero, so delinquency and
   unemployment share no stable long-run equilibrium — they co-move over the cycle
   but drift apart in levels. The statistically sound specification is the
   **first-difference** model (ΔDQ on ΔU lags, R² ≈ 0.45).

3. **Delinquency leads unemployment, not the reverse.** The first-difference CCF
   peaks at +1 quarter (0.58), and lagged ΔDQ significantly Granger-predicts ΔU
   (p ≤ 0.005 at every lag) while lagged ΔU does not predict ΔDQ at short lags.
   Credit-card delinquency is a ~1-quarter *leading indicator* of the unemployment
   rate — coherent, because the unemployment rate is a lagging survey statistic
   while delinquency registers household stress immediately. (The level CCF's
   apparent lead is a spurious trend artifact.)

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
  (`CORCACBS`) is a cleaner unemployment signal and worth a companion check.
- **Near unit root.** Delinquency is highly persistent (dynamic AR(1) ≈ 0.95), so
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
