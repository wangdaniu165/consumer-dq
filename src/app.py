"""Streamlit dashboard for the consumer DQ vs unemployment model."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import LAG_QUARTERS, PREDICTOR, SCENARIOS, TARGET
from src.model import (
    compute_ccf,
    fit_dynamic,
    fit_ecm,
    fit_piecewise,
    fit_piecewise_monotone,
    fit_static,
    lead_lag_diagnostic,
    predict_piecewise,
    select_knots_bic,
)
from src.process import build_features, load_aligned
from src.project import build_scenario_path, project

# --- palette (light) -------------------------------------------------------
C_UNEMP = "#2a78d6"   # categorical slot 1 — unemployment
C_DQ = "#eb6834"      # categorical slot 2 — delinquency (target)
C_CC = "#1baf7a"      # categorical slot 3 — credit card (comparison)
C_LEAD = "#2a78d6"    # diverging pole — unemployment leads
C_LAG = "#e34948"     # diverging pole — delinquency leads
C_ZERO = "#898781"    # diverging midpoint
INK = "#0b0b0b"
GRID = "#e1e0d9"
SCEN_COLORS = {"baseline": "#2a78d6", "moderate": "#eb6834", "severe": "#1baf7a"}


def _base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="system-ui, 'Segoe UI', sans-serif", color=INK, size=13),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        margin=dict(l=48, r=24, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", zerolinecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", zerolinecolor="#c3c2b7")
    return fig


@st.cache_data
def get_data():
    aligned = load_aligned()
    feats = build_features(aligned).dropna()
    return aligned, feats


def _chart_overview(aligned: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Unemployment rate (%)", "Delinquency rate (%)"))
    fig.add_trace(go.Scatter(x=aligned.index.to_timestamp(), y=aligned[PREDICTOR],
                             name="Unemployment", line=dict(color=C_UNEMP, width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=aligned.index.to_timestamp(), y=aligned[TARGET],
                             name="All loans", line=dict(color=C_DQ, width=2)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=aligned.index.to_timestamp(), y=aligned["DRCCLACBS"],
                             name="Credit card", line=dict(color=C_CC, width=2)),
                  row=2, col=1)
    return _base_layout(fig)


def _chart_relationship(aligned: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    x = aligned[PREDICTOR]
    y = aligned[TARGET]

    ccf = compute_ccf(x, y, max_lag=8)
    lags = sorted(ccf)
    vals = [ccf[l] for l in lags]
    colors = [C_LEAD if l < 0 else (C_LAG if l > 0 else C_ZERO) for l in lags]
    bar = go.Figure(go.Bar(x=lags, y=vals, marker_color=colors))
    bar.add_vline(x=0, line_color=C_ZERO, line_width=1)
    peak = max(ccf, key=ccf.get)
    bar.update_layout(xaxis_title="lag (quarters) — negative = unemployment leads",
                      yaxis_title="correlation")
    bar.add_annotation(x=peak, y=ccf[peak], text=f"peak {peak:+d}q",
                       showarrow=True, arrowhead=2, font=dict(color=INK))

    diag = lead_lag_diagnostic(aligned, lag_quarters=LAG_QUARTERS)
    leadlag = go.Figure(go.Bar(x=diag.index, y=diag["coefficient"],
                               marker_color=C_UNEMP))
    leadlag.update_layout(xaxis_title="regressor (u_lead = future unemployment)",
                          yaxis_title="coefficient")

    scatter = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="quarterly",
                                   marker=dict(color=C_UNEMP, size=8, opacity=0.7)))
    scatter.update_layout(xaxis_title="Unemployment rate (%)",
                          yaxis_title="Delinquency rate (%)")

    roll = x.rolling(20).corr(y)
    rolling = go.Figure(go.Scatter(x=roll.index.to_timestamp(), y=roll,
                                   name="20q rolling corr",
                                   line=dict(color=C_UNEMP, width=2)))
    rolling.update_layout(yaxis_title="correlation", yaxis=dict(range=[-1, 1]))

    return _base_layout(bar), _base_layout(leadlag), _base_layout(scatter), _base_layout(rolling)


def _chart_model(feats: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure]:
    dynamic = fit_dynamic(feats)
    static = fit_static(feats)

    lags = [f"u_lag{i}" for i in range(LAG_QUARTERS + 1)]
    beta = [dynamic.params.get(l, 0.0) for l in lags]
    profile = go.Figure(go.Bar(x=list(range(LAG_QUARTERS + 1)), y=beta,
                               marker_color=C_UNEMP))
    profile.update_layout(xaxis_title="quarters of unemployment lag",
                          yaxis_title="coefficient βᵢ")

    fit = go.Figure()
    fit.add_trace(go.Scatter(x=feats.index.to_timestamp(), y=feats[TARGET],
                             name="actual", line=dict(color=C_DQ, width=2)))
    fit.add_trace(go.Scatter(x=feats.index.to_timestamp(), y=dynamic.fitted,
                             name="fitted", line=dict(color=C_UNEMP, width=2)))

    resid = go.Figure(go.Scatter(x=feats.index.to_timestamp(), y=dynamic.residuals,
                                 name="residuals", line=dict(color=C_ZERO, width=1)))
    resid.add_hline(y=0, line_color=C_ZERO, line_width=1)

    return _base_layout(profile), _base_layout(fit), _base_layout(resid)


def _chart_piecewise(aligned: pd.DataFrame, n_knots: int, monotone: bool = False):
    fitter = fit_piecewise_monotone if monotone else fit_piecewise
    pw = fitter(aligned, n_knots=n_knots)
    lin = fit_static(build_features(aligned, 0).dropna(), lag_quarters=0)
    u_grid = np.linspace(aligned[PREDICTOR].min(), aligned[PREDICTOR].max(), 200)
    curve = predict_piecewise(u_grid, pw)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aligned[PREDICTOR], y=aligned[TARGET], mode="markers",
                             name="quarterly", marker=dict(color=C_UNEMP, size=6, opacity=0.5)))
    fig.add_trace(go.Scatter(x=u_grid, y=curve, name="piecewise fit",
                             line=dict(color=C_DQ, width=3)))
    for k in pw.knots:
        fig.add_vline(x=k, line=dict(color=C_ZERO, dash="dot", width=1))
    fig.update_layout(xaxis_title="Unemployment rate (%)",
                      yaxis_title="Delinquency rate (%)")
    return pw, lin, _base_layout(fig)


def _chart_ecm(feats: pd.DataFrame):
    ecm = fit_ecm(feats)
    dus = [f"du_lag{i}" for i in range(LAG_QUARTERS + 1)]
    gamma = [ecm.short_run_params.get(k, 0.0) for k in dus]
    fig = go.Figure(go.Bar(x=list(range(LAG_QUARTERS + 1)), y=gamma,
                           marker_color=C_UNEMP))
    fig.update_layout(xaxis_title="quarters of ΔU lag",
                      yaxis_title="short-run coefficient γᵢ")
    return ecm, _base_layout(fig)


def _chart_stress(feats: pd.DataFrame, last_unemp: float, horizon: int) -> go.Figure:
    dynamic = fit_dynamic(feats)
    hist = feats[TARGET][-40:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index.to_timestamp(), y=hist,
                             name="historical", line=dict(color=INK, width=2)))

    future = pd.period_range(feats.index[-1] + 1, periods=horizon, freq="Q")
    for name, scenario in SCENARIOS.items():
        path = build_scenario_path(last_unemp, scenario, horizon)
        proj = project(dynamic, feats, path, horizon)
        fig.add_trace(go.Scatter(x=future.to_timestamp(), y=proj, name=name,
                                 line=dict(color=SCEN_COLORS[name], width=2, dash="dot")))
    return _base_layout(fig)


def main():
    st.set_page_config(page_title="Consumer DQ vs Unemployment", layout="wide")
    st.title("Consumer Delinquency vs Unemployment")

    try:
        aligned, feats = get_data()
    except FileNotFoundError:
        st.warning("No data yet — run `dq-download` then `dq-process` first.")
        st.stop()

    last_unemp = float(aligned[PREDICTOR].iloc[-1])

    tab_overview, tab_rel, tab_model, tab_stress = st.tabs(
        ["Overview", "Relationship", "Model", "Stress test"])

    with tab_overview:
        st.plotly_chart(_chart_overview(aligned), use_container_width=True)

    with tab_rel:
        ccf, leadlag, scatter, rolling = _chart_relationship(aligned)
        st.subheader("Lead / lag (cross-correlation)")
        st.plotly_chart(ccf, use_container_width=True)
        st.subheader("Lead / lag (two-sided regression)")
        st.plotly_chart(leadlag, use_container_width=True)
        st.subheader("Scatter")
        st.plotly_chart(scatter, use_container_width=True)

        st.subheader("Piecewise-linear fit")
        best_knots, bic_results = select_knots_bic(aligned, max_knots=5)
        nk = st.slider("Number of knots", 1, 5, max(best_knots, 1), key="pw_knots")
        monotone = st.checkbox("Enforce monotonicity (slopes ≥ 0)", value=True)
        pw, lin, pw_fig = _chart_piecewise(aligned, nk, monotone=monotone)
        c1, c2, c3 = st.columns(3)
        c1.metric("Piecewise R²", f"{pw.r_squared:.3f}")
        c2.metric("Linear R²", f"{lin.r_squared:.3f}")
        c3.metric("BIC-selected knots", str(best_knots))
        st.plotly_chart(pw_fig, use_container_width=True)

        bic_fig = go.Figure(go.Scatter(
            x=[r["n_knots"] for r in bic_results], y=[r["bic"] for r in bic_results],
            mode="lines+markers", name="BIC", line=dict(color=C_UNEMP, width=2)))
        bic_fig.add_trace(go.Scatter(
            x=[best_knots], y=[min(r["bic"] for r in bic_results)],
            mode="markers", name="minimum", marker=dict(color=C_LAG, size=14)))
        bic_fig.update_layout(xaxis_title="knots", yaxis_title="BIC")
        st.plotly_chart(_base_layout(bic_fig), use_container_width=True)

        st.caption(
            f"Knots at unemployment {', '.join(f'{k:.1f}%' for k in pw.knots)}. "
            "Monotonicity forces every segment slope ≥ 0 so delinquency never "
            "falls as unemployment rises — at the cost of a slightly lower R²."
        )

        st.subheader("Rolling correlation")
        st.plotly_chart(rolling, use_container_width=True)

    with tab_model:
        profile, fit, resid = _chart_model(feats)
        dynamic = fit_dynamic(feats)
        static = fit_static(feats)
        c1, c2 = st.columns(2)
        c1.metric("Dynamic R²", f"{dynamic.r_squared:.3f}")
        c2.metric("Static R²", f"{static.r_squared:.3f}")
        st.subheader("Lag profile (dynamic)")
        st.plotly_chart(profile, use_container_width=True)
        st.subheader("Fit vs actual")
        st.plotly_chart(fit, use_container_width=True)
        st.subheader("Residuals")
        st.plotly_chart(resid, use_container_width=True)

        st.subheader("Error-correction model (Engle-Granger)")
        ecm, ecm_chart = _chart_ecm(feats)
        e1, e2, e3 = st.columns(3)
        e1.metric("Speed of adjustment λ", f"{ecm.speed_of_adjustment:+.3f}")
        e2.metric("Long-run β (per 1pp U)", f"{ecm.long_run_params.get(PREDICTOR, 0.0):.3f}")
        e3.metric("ECM R²", f"{ecm.r_squared:.3f}")
        st.plotly_chart(ecm_chart, use_container_width=True)
        st.caption(
            "λ must be negative for genuine error correction. λ ≈ 0 / positive "
            "means the series are not cointegrated — the short-run ΔU model "
            "(this chart) is the statistically sound read."
        )

    with tab_stress:
        horizon = st.slider("Projection horizon (quarters)", 4, 24, 8, step=4)
        st.plotly_chart(_chart_stress(feats, last_unemp, horizon),
                        use_container_width=True)
        st.caption(f"Current unemployment: {last_unemp:.1f}%. "
                   "Scenarios step unemployment up immediately and hold; the model "
                   f"feeds the shock through {LAG_QUARTERS} quarters of lags.")


if __name__ == "__main__":
    main()
