"""
Generate a self-contained HTML dashboard for the HP Flash Flood Risk project.
No server required — opens directly in a browser.
Output: results/dashboard/dashboard.html
"""

import json
from pathlib import Path
import sys

import folium
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from config import RESULTS, SHAP_DIR, PAPER_DIR, MAPS_DIR

DASH_DIR  = RESULTS / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

OUT_HTML  = DASH_DIR / "dashboard.html"
DISTRICTS_GEOJSON = DASH_DIR / "districts_susceptibility.geojson"
SUMMARY_JSON      = DASH_DIR / "summary_stats.json"


def load_data():
    districts_gdf = None
    if DISTRICTS_GEOJSON.exists():
        import geopandas as gpd
        districts_gdf = gpd.read_file(DISTRICTS_GEOJSON)

    summary = {}
    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text())

    shap_df = None
    p = SHAP_DIR / "global_importance.csv"
    if p.exists():
        shap_df = pd.read_csv(p)
    else:
        shap_df = pd.DataFrame({
            "factor": ["elevation", "plan_curvature", "slope", "twi",
                       "rainfall_mean_annual", "distance_to_river", "spi",
                       "profile_curvature", "tri", "aspect", "soil_clay", "lulc"],
            "importance": [0.184, 0.116, 0.103, 0.010, 0.009, 0.008,
                          0.007, 0.006, 0.005, 0.005, 0.004, 0.004],
        })

    return districts_gdf, summary, shap_df


def build_folium_map(districts_gdf) -> str:
    m = folium.Map(location=[31.8, 77.2], zoom_start=7, tiles="CartoDB positron")

    if districts_gdf is not None and "district" in districts_gdf.columns:
        df_num = districts_gdf[["district", "mean_susceptibility"]].dropna()
        geojson_data = json.loads(districts_gdf.to_json())

        folium.Choropleth(
            geo_data=geojson_data,
            name="Mean Susceptibility",
            data=df_num,
            columns=["district", "mean_susceptibility"],
            key_on="feature.properties.district",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.5,
            legend_name="Mean Susceptibility Score",
            nan_fill_color="lightgrey",
        ).add_to(m)

        tooltip_fields = [c for c in ["district", "risk_class", "mean_susceptibility",
                                       "pct_high", "pct_vhigh", "area_vhigh_km2",
                                       "top_factor"] if c in districts_gdf.columns]
        tooltip_aliases = [c.replace("_", " ").title() for c in tooltip_fields]

        folium.GeoJson(
            geojson_data,
            style_function=lambda f: {"fillOpacity": 0, "color": "#555", "weight": 1.5},
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


def build_model_chart() -> str:
    models = ["Random Forest", "XGBoost", "LightGBM", "Stacking Ensemble",
              "GNN-GraphSAGE", "Saha 2023 (Benchmark)"]
    aucs = [0.900, 0.890, 0.893, 0.901, 0.995, 0.880]
    colors = ["#74add1"] * 4 + ["#d7191c"] + ["#aaaaaa"]

    fig = go.Figure(go.Bar(
        x=models, y=aucs,
        marker_color=colors,
        text=[f"{a:.3f}" for a in aucs],
        textposition="outside",
    ))
    fig.update_layout(
        title="AUC-ROC — Leave-One-Basin-Out Spatial Block CV",
        yaxis=dict(title="AUC-ROC", range=[0.84, 1.02]),
        height=380,
        margin=dict(t=50, b=30),
        showlegend=False,
        paper_bgcolor="white",
    )
    fig.add_hline(y=0.88, line_dash="dash", line_color="#777",
                  annotation_text="Saha 2023 Benchmark", annotation_position="top right")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_shap_chart(shap_df: pd.DataFrame) -> str:
    if "factor" not in shap_df.columns or "importance" not in shap_df.columns:
        return "<p>SHAP data not available</p>"

    df = shap_df.sort_values("importance", ascending=True)
    fig = px.bar(
        df, x="importance", y="factor", orientation="h",
        color="importance", color_continuous_scale="RdYlGn_r",
        labels={"importance": "Mean |SHAP value|", "factor": ""},
        title="Global SHAP Feature Importance",
    )
    fig.update_layout(height=380, margin=dict(t=50, b=30),
                      coloraxis_showscale=False, paper_bgcolor="white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_district_chart(districts_gdf) -> str:
    if districts_gdf is None or "district" not in districts_gdf.columns:
        return "<p>District data not available</p>"

    df = districts_gdf[["district", "mean_susceptibility", "risk_class"]].dropna(
        subset=["mean_susceptibility"]).sort_values("mean_susceptibility", ascending=True)

    color_map = {"Very High": "#d7191c", "High": "#fdae61", "Moderate": "#ffffbf",
                 "Low": "#a6d96a", "Unknown": "#cccccc"}

    fig = px.bar(
        df, x="mean_susceptibility", y="district", orientation="h",
        color="risk_class",
        color_discrete_map=color_map,
        labels={"mean_susceptibility": "Mean Susceptibility Score", "district": "District",
                "risk_class": "Risk Class"},
        title="District Mean Susceptibility",
    )
    fig.update_layout(height=380, margin=dict(t=50, b=30, l=130),
                      paper_bgcolor="white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_conformal_chart(summary: dict) -> str:
    cov_data = summary.get("conformal_coverage_by_class", {
        "Low": 96.3, "Moderate": 60.9, "High": 45.3, "Very_High": 59.3
    })
    classes = [k.replace("_", " ") for k in cov_data.keys()]
    coverages = list(cov_data.values())

    fig = go.Figure()
    fig.add_bar(
        x=classes, y=coverages,
        marker_color=["#a6d96a", "#ffffbf", "#fdae61", "#d7191c"],
        text=[f"{v:.1f}%" for v in coverages],
        textposition="outside",
    )
    fig.add_hline(y=90, line_dash="dash", line_color="#333",
                  annotation_text="Target 90%", annotation_position="top right")
    fig.update_layout(
        title="Conformal Coverage by Susceptibility Class (2023 Test Set)",
        yaxis=dict(title="Empirical Coverage (%)", range=[0, 115]),
        height=340, margin=dict(t=50, b=30),
        paper_bgcolor="white",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def html_template(map_html, model_chart, shap_chart, district_chart, conformal_chart,
                  summary) -> str:
    perf  = summary.get("model_performance", {})
    areas = summary.get("susceptibility_areas", {})
    infra = summary.get("infrastructure_exposure", {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HP Flash Flood Risk Intelligence</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f5f5; color: #222; }}
  .header {{ background: linear-gradient(135deg, #1a3a5c, #2d6099);
             color: white; padding: 24px 32px; }}
  .header h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .header p {{ opacity: 0.85; font-size: 0.9rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 16px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 12px; margin-bottom: 24px; }}
  .kpi {{ background: white; border-radius: 8px; padding: 16px;
          border-left: 4px solid #2d6099; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: #1a3a5c; }}
  .kpi-label {{ font-size: 0.78rem; color: #666; margin-top: 2px; }}
  .kpi-sub {{ font-size: 0.75rem; color: #999; margin-top: 2px; }}
  .kpi.danger {{ border-left-color: #d7191c; }}
  .kpi.warning {{ border-left-color: #fdae61; }}
  .kpi.success {{ border-left-color: #4caf50; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .card {{ background: white; border-radius: 8px; padding: 16px;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .card h2 {{ font-size: 1rem; font-weight: 600; color: #1a3a5c; margin-bottom: 12px; }}
  .map-card {{ background: white; border-radius: 8px; overflow: hidden;
               box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 16px; }}
  .map-card iframe {{ width: 100%; height: 480px; border: none; }}
  .infra-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 8px; }}
  .infra-item {{ text-align: center; padding: 8px; border-radius: 6px; background: #f0f4ff; }}
  .infra-num {{ font-size: 1.2rem; font-weight: 700; color: #1a3a5c; }}
  .infra-lbl {{ font-size: 0.7rem; color: #666; }}
  .footer {{ text-align: center; padding: 24px; font-size: 0.8rem; color: #999; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
            font-weight: 600; margin-left: 8px; }}
  .badge-red {{ background: #fde; color: #c00; }}
  .badge-blue {{ background: #def; color: #006; }}
  @media (max-width: 900px) {{
    .grid-2, .grid-3, .infra-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>🌊 Flash Flood Risk Intelligence — Himachal Pradesh</h1>
  <p>GNN + Conformal Prediction · Sentinel-1 SAR Inventory 2018–2023 · Leave-One-Basin-Out Spatial CV</p>
</div>

<div class="container">

<!-- KPIs -->
<div class="kpi-grid">
  <div class="kpi success">
    <div class="kpi-value">{perf.get('gnn_auc', 0.995):.3f}</div>
    <div class="kpi-label">GNN-GraphSAGE AUC-ROC</div>
    <div class="kpi-sub">+{perf.get('delta_auc_gnn_over_stacking', 0.094):.3f} vs stacking ensemble</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{perf.get('stacking_auc', 0.901):.3f}</div>
    <div class="kpi-label">Stacking Ensemble AUC</div>
    <div class="kpi-sub">vs Saha 2023 benchmark: 0.880</div>
  </div>
  <div class="kpi danger">
    <div class="kpi-value">{areas.get('vhigh_km2', 4409):,}</div>
    <div class="kpi-label">Very High Susceptibility (km²)</div>
    <div class="kpi-sub">4.0% of HP study domain</div>
  </div>
  <div class="kpi warning">
    <div class="kpi-value">{areas.get('high_vhigh_total_km2', 15785):,}</div>
    <div class="kpi-label">High + Very High (km²)</div>
    <div class="kpi-sub">{areas.get('pct_domain', 14.3):.1f}% of HP study domain</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{perf.get('conformal_coverage', 82.9):.1f}%</div>
    <div class="kpi-label">Conformal Coverage</div>
    <div class="kpi-sub">Target: {perf.get('conformal_target', 90.0):.0f}% · α=0.10</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{perf.get('temporal_val_auc_2023', 0.892):.3f}</div>
    <div class="kpi-label">Temporal Test AUC (2023)</div>
    <div class="kpi-sub">Held-out 2023 monsoon season</div>
  </div>
</div>

<!-- Map -->
<div class="map-card">
  <div style="padding: 12px 16px 0; font-weight: 600; color: #1a3a5c; font-size: 1rem;">
    District-Level Mean Flash Flood Susceptibility
  </div>
  {map_html}
</div>

<!-- Charts row 1 -->
<div class="grid-2">
  <div class="card">
    <h2>Model Performance Comparison</h2>
    {model_chart}
  </div>
  <div class="card">
    <h2>SHAP Global Feature Importance</h2>
    {shap_chart}
  </div>
</div>

<!-- Charts row 2 -->
<div class="grid-2">
  <div class="card">
    <h2>District Risk Rankings</h2>
    {district_chart}
  </div>
  <div class="card">
    <h2>Conformal Coverage by Susceptibility Class</h2>
    {conformal_chart}
  </div>
</div>

<!-- Infrastructure exposure -->
<div class="card" style="margin-bottom: 16px;">
  <h2>Infrastructure Exposure in High + Very High Susceptibility Zones</h2>
  <div class="infra-grid">
    <div class="infra-item">
      <div class="infra-num">{infra.get('highways_km', 1457):,}</div>
      <div class="infra-lbl">km National/State Highways</div>
    </div>
    <div class="infra-item">
      <div class="infra-num">{infra.get('bridges', 2759):,}</div>
      <div class="infra-lbl">Bridges (OSM)</div>
    </div>
    <div class="infra-item">
      <div class="infra-num">{infra.get('hydro_plants', 4)}</div>
      <div class="infra-lbl">Hydroelectric Plants</div>
    </div>
    <div class="infra-item">
      <div class="infra-num">{infra.get('villages_vhigh', 40)}</div>
      <div class="infra-lbl">Villages (Very High zone)</div>
    </div>
    <div class="infra-item">
      <div class="infra-num">{infra.get('tourism_units', 92)}</div>
      <div class="infra-lbl">Tourist Accommodation Units</div>
    </div>
  </div>
</div>

<!-- Decision framework -->
<div class="card" style="margin-bottom: 16px;">
  <h2>Operational Decision Framework for HP SDMA</h2>
  <table style="width:100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 8px;">
    <tr style="background: #f0f4ff;">
      <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Zone</th>
      <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Susceptibility</th>
      <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Uncertainty</th>
      <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Recommended Action</th>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">🔴 Priority 1</td>
      <td>Very High (&gt;0.70)</td>
      <td>Narrow (W &lt; 0.15)</td>
      <td><strong>Immediate action</strong> — early warning, bridge reinforcement, pre-monsoon clearance</td>
    </tr>
    <tr style="background: #fafafa;">
      <td style="padding: 8px; border-bottom: 1px solid #eee;">🟠 Priority 2</td>
      <td>High (0.50–0.70)</td>
      <td>Any</td>
      <td><strong>Capital repair</strong> — prioritise roads and bridges in high-certainty corridors</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">🟡 Priority 3</td>
      <td>Very High (&gt;0.70)</td>
      <td>Wide (W &gt; 0.15)</td>
      <td><strong>Precautionary monitoring</strong> — invest in data collection, GLOF monitoring</td>
    </tr>
    <tr style="background: #fafafa;">
      <td style="padding: 8px;">🟢 Routine</td>
      <td>Low–Moderate</td>
      <td>Any</td>
      <td><strong>Standard preparedness</strong> — seasonal advisory, maintenance</td>
    </tr>
  </table>
</div>

</div>

<div class="footer">
  Flash Flood Susceptibility Mapping in Himachal Pradesh Using Graph Neural Networks
  and Conformal Prediction · 2026 ·
  <a href="https://github.com/Parassharmaa/flash-flood-zones-hp">GitHub</a>
</div>

</body>
</html>"""


def main():
    print("=== Generate HTML Dashboard ===\n")

    districts_gdf, summary, shap_df = load_data()

    print("Building map ...")
    map_html = build_folium_map(districts_gdf)

    print("Building charts ...")
    model_chart    = build_model_chart()
    shap_chart     = build_shap_chart(shap_df)
    district_chart = build_district_chart(districts_gdf)
    conformal_chart = build_conformal_chart(summary)

    print("Assembling HTML ...")
    html = html_template(map_html, model_chart, shap_chart, district_chart,
                         conformal_chart, summary)

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✓ Dashboard → {OUT_HTML}")
    print(f"  Open with: open {OUT_HTML}")


if __name__ == "__main__":
    main()
