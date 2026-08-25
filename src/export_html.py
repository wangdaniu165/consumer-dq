"""Generate a self-contained static HTML dashboard (no server, no Streamlit).

Run:  python -m src.export_html
Output: ``dashboard.html`` at the project root — open it in any browser.

The page embeds every chart as Plotly JSON and renders it client-side with
Plotly.js (loaded from a CDN). It is a static file: no Python server, no
WebSocket, no ``streamlit`` dependency at view time.
"""

import json
from pathlib import Path

import plotly.graph_objects as go

from src.app import (
    C_LAG,
    C_UNEMP,
    _base_layout,
    _chart_ecm,
    _chart_model,
    _chart_overview,
    _chart_piecewise,
    _chart_relationship,
    get_data,
)
from src.config import EXCLUDE_COVID, PREDICTOR, START_DATE
from src.model import fit_dynamic, fit_ecm, fit_static, select_knots_bic

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "dashboard.html"

# Defaults for the parts that were interactive in the Streamlit app.
N_KNOTS = 3
MONOTONE = True


def build_dashboard() -> tuple[dict, list[tuple[str, str, str, go.Figure]]]:
    """Compute metrics and build every chart.

    Returns ``(metrics, figures)`` where ``figures`` is a list of
    ``(section, heading, div_id, figure)`` tuples in display order.
    """
    aligned, feats, est = get_data()

    static = fit_static(est)
    dynamic = fit_dynamic(est)
    ecm = fit_ecm(est)

    best_knots, bic_results = select_knots_bic(est, max_knots=3)
    pw, _, pw_fig = _chart_piecewise(est, N_KNOTS, monotone=MONOTONE)

    figures: list[tuple[str, str, str, go.Figure]] = []

    # --- Overview -----------------------------------------------------------
    figures.append(("Overview", "Unemployment vs delinquency", "overview",
                    _chart_overview(aligned)))

    # --- Relationship -------------------------------------------------------
    ccf, leadlag, scatter = _chart_relationship(aligned)
    figures.append(("Relationship", "Lead/lag cross-correlation", "ccf", ccf))
    figures.append(("Relationship", "Lead/lag two-sided regression", "leadlag", leadlag))
    figures.append(("Relationship", "Scatter (delinquency vs unemployment)", "scatter", scatter))
    figures.append(("Relationship", "Piecewise-linear fit (monotone)", "piecewise", pw_fig))

    bic_fig = go.Figure(go.Scatter(
        x=[r["n_knots"] for r in bic_results], y=[r["bic"] for r in bic_results],
        mode="lines+markers", name="BIC", line=dict(color=C_UNEMP, width=2)))
    bic_fig.add_trace(go.Scatter(
        x=[best_knots], y=[min(r["bic"] for r in bic_results)],
        mode="markers", name="minimum", marker=dict(color=C_LAG, size=14)))
    bic_fig.update_layout(xaxis_title="knots", yaxis_title="BIC")
    figures.append(("Relationship", "Knot-count selection (BIC)", "bic", _base_layout(bic_fig)))

    # --- Model --------------------------------------------------------------
    profile, fit_fig, resid = _chart_model(est)
    figures.append(("Model", "Lag profile (dynamic)", "lagprofile", profile))
    figures.append(("Model", "Fit vs actual", "fit", fit_fig))
    figures.append(("Model", "Residuals", "resid", resid))
    ecm_res, ecm_fig = _chart_ecm(est)
    figures.append(("Model", "Error-correction model (short-run ΔU)", "ecm", ecm_fig))

    metrics = {
        "static_r2": static.r_squared,
        "dynamic_r2": dynamic.r_squared,
        "ecm_r2": ecm.r_squared,
        "lambda": ecm.speed_of_adjustment,
        "long_run_beta": ecm.long_run_params.get(PREDICTOR, 0.0),
        "piecewise_r2": pw.r_squared,
        "knots": pw.knots,
        "n_obs": len(est),
        "exclude_covid": EXCLUDE_COVID,
    }
    if EXCLUDE_COVID:
        metrics["r2_dummy"] = fit_static(feats).r_squared

    return metrics, figures


def _metrics_html(m: dict) -> str:
    cells = [
        ("Static R²", f"{m['static_r2']:.3f}"),
        ("Dynamic R²", f"{m['dynamic_r2']:.3f}"),
        ("ECM R²", f"{m['ecm_r2']:.3f}"),
        ("Speed of adjustment λ", f"{m['lambda']:+.3f}"),
        ("Long-run β (pp DQ / 1pp U)", f"{m['long_run_beta']:+.3f}"),
        ("Piecewise R²", f"{m['piecewise_r2']:.3f}"),
    ]
    html = '<div class="metrics">'
    for label, value in cells:
        html += f'<div class="metric"><span class="metric-value">{value}</span>'
        html += f'<span class="metric-label">{label}</span></div>'
    html += "</div>"
    return html


def render_html(metrics: dict, figures: list[tuple[str, str, str, go.Figure]]) -> str:
    figs_json = {div_id: json.loads(fig.to_json()) for _, _, div_id, fig in figures}

    covid_note = ""
    if metrics.get("exclude_covid"):
        covid_note = (
            f"COVID window 2020Q1&ndash;2021Q4 excluded from estimation "
            f"(static R² with a dummy was {metrics['r2_dummy']:.3f}; "
            f"excluding raises it to {metrics['static_r2']:.3f})."
        )
    knots = ", ".join(f"{k:.1f}%" for k in metrics["knots"])

    # group figures by section while preserving order
    sections: dict[str, list[tuple[str, str]]] = {}
    for section, heading, div_id, _ in figures:
        sections.setdefault(section, []).append((heading, div_id))

    body = ['<div class="wrap">',
            "<h1>Consumer Delinquency vs Unemployment</h1>",
            '<p class="subtitle">Credit-card delinquency (unsecured consumer proxy) vs the '
            f"unemployment rate &mdash; quarterly, {START_DATE}+, {metrics['n_obs']} estimation "
            "quarters.</p>",
            _metrics_html(metrics),
            f'<p class="note">{covid_note}</p>']

    for section in sections:
        body.append(f'<h2 class="section">{section}</h2>')
        for heading, div_id in sections[section]:
            body.append(f'<div class="chart"><h3>{heading}</h3>'
                        f'<div id="{div_id}"></div></div>')

    body.append(
        f'<p class="note">Piecewise fit uses {N_KNOTS} knots ({knots}) with the '
        f"monotonicity constraint (all segment slopes &ge; 0). λ &asymp; 0 means no "
        "cointegration &mdash; the first-difference (ECM) model is the sound read. "
        'Full methodology: <a href="model.html">model documentation</a>.</p>')
    body.append("</div>")
    body.append(f'<script id="figs" type="application/json">{json.dumps(figs_json)}</script>')

    return TEMPLATE.replace("<!--BODY-->", "\n".join(body))


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Consumer Delinquency vs Unemployment</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 20px 64px;
    background: #fcfcfb; color: #0b0b0b;
    font: 15px/1.5 system-ui, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  .subtitle { color: #56554f; margin: 0 0 20px; }
  h2.section {
    font-size: 20px; margin: 36px 0 12px; padding-top: 20px;
    border-top: 1px solid #e1e0d9;
  }
  .chart { margin: 0 0 24px; }
  .chart h3 { font-size: 15px; margin: 0 0 6px; color: #33322e; }
  .chart > div { width: 100%; }
  .metrics { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 16px; }
  .metric {
    flex: 1 1 150px; background: #fff; border: 1px solid #e1e0d9;
    border-radius: 6px; padding: 10px 14px;
  }
  .metric-value { display: block; font-size: 22px; font-weight: 600; }
  .metric-label { display: block; font-size: 12px; color: #56554f; }
  .note { color: #56554f; font-size: 13px; max-width: 78ch; }
</style>
</head>
<body>
<!--BODY-->
<script>
  const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  const FIGS = JSON.parse(document.getElementById("figs").textContent);
  Object.keys(FIGS).forEach(id => {
    Plotly.newPlot(id, FIGS[id].data, FIGS[id].layout, CONFIG);
  });
</script>
</body>
</html>
"""


def main() -> None:
    metrics, figures = build_dashboard()
    html = render_html(metrics, figures)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(figures)} charts, {OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
