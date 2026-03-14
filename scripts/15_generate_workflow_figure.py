"""
Generate Fig 2: Methodological workflow diagram for the paper.

Output: results/paper_figures/fig02_workflow.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from config import PAPER_DIR  # noqa: E402


PHASE_COLOURS = {
    "data":    "#AED6F1",
    "proc":    "#A9DFBF",
    "model":   "#F9E79F",
    "novel":   "#F1948A",
    "output":  "#D7BDE2",
}


def box(ax, x, y, w, h, label, sublabel="", colour="#DDDDDD", fontsize=8.5):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=colour, edgecolor="#444444", linewidth=0.8, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=4, wrap=True)
    if sublabel:
        ax.text(x, y - 0.12, sublabel,
                ha="center", va="center", fontsize=6.5,
                color="#555555", zorder=4, style="italic")


def arrow(ax, x0, y0, x1, y1, colour="#666666"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=colour,
                                lw=1.2, connectionstyle="arc3,rad=0.0"))


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ── Column headers ────────────────────────────────────────────────────────
    cols = [
        (1.5,  "Phase 1\nData Collection",       PHASE_COLOURS["data"]),
        (4.2,  "Phase 2\nPreprocessing",          PHASE_COLOURS["proc"]),
        (7.0,  "Phase 3–4\nML Pipeline",          PHASE_COLOURS["model"]),
        (9.8,  "Novel Contributions",             PHASE_COLOURS["novel"]),
        (12.5, "Phase 5–7\nOutput & Paper",       PHASE_COLOURS["output"]),
    ]
    for cx, lbl, col in cols:
        rect = FancyBboxPatch((cx - 1.2, 8.3), 2.4, 0.55,
                              boxstyle="round,pad=0.05",
                              facecolor=col, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(cx, 8.575, lbl, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    # ── Phase 1: Data sources ─────────────────────────────────────────────────
    data_boxes = [
        (1.5, 7.5, "Sentinel-1 SAR\n(GEE, 2018–2024)", "6 seasons, VV pol."),
        (1.5, 6.5, "GLO-30 DEM\n(Copernicus, 30m)",     "HP extent"),
        (1.5, 5.5, "GPM-IMERG\nRainfall",               "Monthly V07"),
        (1.5, 4.5, "ESA WorldCover\nLULC",              "10m → 30m"),
        (1.5, 3.5, "SoilGrids\nSoil clay",              "250m → 30m"),
        (1.5, 2.5, "OSM\nInfrastructure",               "Roads, bridges"),
    ]
    for x, y, lbl, sub in data_boxes:
        box(ax, x, y, 2.3, 0.65, lbl, sub, PHASE_COLOURS["data"])

    # ── Phase 2: Preprocessing ────────────────────────────────────────────────
    proc_boxes = [
        (4.2, 7.5, "SAR Flood\nInventory",          "Seasonal composite\nchange detect."),
        (4.2, 6.2, "Terrain Factors\n(8 DEM vars)",  "Slope,TWI,SPI,TRI…"),
        (4.2, 4.8, "Multicollinearity\nCheck",       "Pearson |r|>0.8, VIF>10"),
        (4.2, 3.5, "Sub-watershed\nDelineation",     "D8, ~460 sub-watersheds"),
        (4.2, 2.5, "Watershed Graph\nConstruction",  "Directed edges upstr→dstr"),
    ]
    for x, y, lbl, sub in proc_boxes:
        box(ax, x, y, 2.3, 0.75, lbl, sub, PHASE_COLOURS["proc"])

    # ── Phase 3-4: Models ─────────────────────────────────────────────────────
    model_boxes = [
        (7.0, 7.3, "Baseline Models",         "RF · XGBoost · LightGBM\nStacking Ensemble"),
        (7.0, 5.7, "GraphSAGE GNN\n★ Novel",  "Watershed graph\n3 layers, 64 dim"),
        (7.0, 4.1, "Spatial Block CV",        "Leave-one-block-out\n5 k-means blocks"),
    ]
    for x, y, lbl, sub in model_boxes:
        box(ax, x, y, 2.3, 0.95, lbl, sub,
            PHASE_COLOURS["novel"] if "Novel" in lbl else PHASE_COLOURS["model"])

    # ── Novel contributions ───────────────────────────────────────────────────
    novel_boxes = [
        (9.8, 7.3, "Conformal Prediction\n★ Novel",   "MAPIE, α=0.10\n90% coverage intervals"),
        (9.8, 5.7, "SHAP Spatial\nAnalysis",           "Global + district\n+ factor map"),
        (9.8, 4.1, "Infrastructure\nRisk Overlay",     "Roads,bridges,hydro\nSettlements"),
    ]
    for x, y, lbl, sub in novel_boxes:
        box(ax, x, y, 2.3, 0.95, lbl, sub, PHASE_COLOURS["novel"])

    # ── Outputs ───────────────────────────────────────────────────────────────
    out_boxes = [
        (12.5, 7.5, "Susceptibility\nMaps (4 class)",  "30m, HP-wide"),
        (12.5, 6.3, "Uncertainty\nWidth Maps",          "90% CI per pixel"),
        (12.5, 5.1, "SHAP Factor\nMaps",                "Dominant factor/district"),
        (12.5, 3.9, "Infrastructure\nExposure Report",  "Roads, bridges, hydro"),
    ]
    for x, y, lbl, sub in out_boxes:
        box(ax, x, y, 2.3, 0.75, lbl, sub, PHASE_COLOURS["output"])

    # ── Arrows ────────────────────────────────────────────────────────────────
    # Data → Proc
    arrow(ax, 2.65, 7.5, 3.05, 7.5)
    arrow(ax, 2.65, 6.5, 3.05, 6.5)
    arrow(ax, 2.65, 5.5, 3.05, 5.5)
    arrow(ax, 2.65, 4.5, 3.05, 4.5)
    arrow(ax, 2.65, 3.5, 3.05, 3.5)
    # Proc → Model
    arrow(ax, 5.35, 7.5, 5.85, 7.3)
    arrow(ax, 5.35, 6.2, 5.85, 5.7)
    arrow(ax, 5.35, 2.5, 5.85, 5.7)   # graph → GNN
    # Model → Novel
    arrow(ax, 8.15, 7.3, 8.65, 7.3)
    arrow(ax, 8.15, 5.7, 8.65, 5.7)
    arrow(ax, 8.15, 4.1, 8.65, 4.1)
    # Novel → Output
    arrow(ax, 10.95, 7.3, 11.35, 7.5)
    arrow(ax, 10.95, 7.3, 11.35, 6.3)
    arrow(ax, 10.95, 5.7, 11.35, 5.1)
    arrow(ax, 10.95, 4.1, 11.35, 3.9)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=PHASE_COLOURS["data"],   label="Data Sources"),
        mpatches.Patch(color=PHASE_COLOURS["proc"],   label="Preprocessing"),
        mpatches.Patch(color=PHASE_COLOURS["model"],  label="ML Models"),
        mpatches.Patch(color=PHASE_COLOURS["novel"],  label="Novel Contributions (★)"),
        mpatches.Patch(color=PHASE_COLOURS["output"], label="Outputs"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              ncol=5, fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_title(
        "Methodological Workflow — Flash Flood Susceptibility Mapping",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    out = PAPER_DIR / "fig02_workflow.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Workflow diagram -> {out}")


if __name__ == "__main__":
    main()
