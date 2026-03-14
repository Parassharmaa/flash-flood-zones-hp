"""
Generate Fig 1: Study area map (HP state boundary + districts + river basins).
Uses the downloaded boundaries GeoJSON directly.

Output: results/paper_figures/fig01_study_area.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import PAPER_DIR, BOUNDARIES_DIR  # noqa: E402

# River basin colours
BASIN_COLOURS = {
    "Beas":   "#4ECDC4",
    "Satluj": "#45B7D1",
    "Chenab": "#96CEB4",
    "Ravi":   "#FFEAA7",
    "Yamuna": "#DDA0DD",
}

# Major locations (lon, lat, label, district)
LOCATIONS = [
    (77.189, 32.224, "Shimla\n(capital)",   "right"),
    (77.103, 32.100, "Manali",              "right"),
    (77.111, 31.828, "Kullu",               "left"),
    (76.918, 31.709, "Mandi",               "right"),
    (76.520, 32.100, "Dharamshala",         "right"),
    (77.583, 31.633, "Rampur",              "right"),
    (78.533, 31.583, "Reckong Peo",         "left"),
    (76.133, 32.556, "Chamba",              "right"),
    (77.800, 32.700, "Keylong",             "right"),
    (78.133, 31.833, "Kaza",               "left"),
]


def load_geojson(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_polygon(ax, geometry, **kwargs):
    """Plot a GeoJSON geometry (Polygon or MultiPolygon) onto ax."""
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    polys = []
    if geometry["type"] == "Polygon":
        coords = geometry["coordinates"][0]
        polys.append(MplPolygon(coords, closed=True))
    elif geometry["type"] == "MultiPolygon":
        for part in geometry["coordinates"]:
            polys.append(MplPolygon(part[0], closed=True))

    col = PatchCollection(polys, **kwargs)
    ax.add_collection(col)
    return col


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 12))

    # ── HP state outline ─────────────────────────────────────────────────────
    state_path = BOUNDARIES_DIR / "hp_state.geojson"
    if state_path.exists():
        state_gj = load_geojson(state_path)
        feats = state_gj.get("features", [state_gj])
        for feat in feats:
            geom = feat.get("geometry", feat)
            plot_polygon(ax, geom,
                         facecolor="#F5F5DC", edgecolor="black",
                         linewidth=1.5, alpha=0.3, zorder=1)

    # ── Districts ─────────────────────────────────────────────────────────────
    dist_path = BOUNDARIES_DIR / "hp_districts_raw.json"
    if dist_path.exists():
        dist_gj = load_geojson(dist_path)
        feats = dist_gj.get("features", [])
        for feat in feats:
            geom = feat.get("geometry", {})
            if not geom:
                continue
            plot_polygon(ax, geom,
                         facecolor="none", edgecolor="#888888",
                         linewidth=0.5, alpha=0.8, zorder=2)

    # ── River corridors (schematic lines, not real polylines) ─────────────────
    rivers = {
        "Beas":   [(77.12, 32.70), (77.10, 32.10), (76.85, 31.60)],
        "Satluj": [(78.50, 31.85), (77.60, 31.65), (76.40, 31.20)],
        "Chenab": [(77.90, 32.90), (76.90, 32.50), (76.00, 33.10)],
        "Ravi":   [(76.50, 32.65), (76.10, 32.20), (75.70, 32.10)],
        "Yamuna": [(78.00, 31.10), (77.60, 30.60), (77.20, 30.35)],
    }
    for basin, pts in rivers.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=BASIN_COLOURS[basin],
                linewidth=2.5, alpha=0.85, zorder=3,
                label=f"{basin} basin",
                path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])

    # ── Location markers ──────────────────────────────────────────────────────
    for lon, lat, label, ha in LOCATIONS:
        ax.plot(lon, lat, "o", color="#333333", markersize=4, zorder=5)
        ax.annotate(
            label, xy=(lon, lat),
            xytext=(4 if ha == "right" else -4, 0),
            textcoords="offset points",
            ha=ha, va="center", fontsize=7.5,
            color="#222222",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # ── NH highways (schematic) ───────────────────────────────────────────────
    nh3  = [(77.20, 32.24), (77.10, 32.55), (77.08, 32.90), (77.12, 33.10)]   # Manali-Leh
    nh21 = [(76.82, 30.70), (77.00, 31.40), (77.10, 31.83), (77.10, 32.10)]   # Chandigarh-Manali
    for pts, label, col in [(nh3, "NH-3 (Manali-Leh)", "#CC4444"),
                             (nh21, "NH-21 (Chd-Manali)", "#CC4444")]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "--", color=col, linewidth=1.2, alpha=0.7, zorder=4)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(75.3, 79.2)
    ax.set_ylim(30.1, 33.5)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title(
        "Himachal Pradesh — Five Major River Basins",
        fontsize=12, fontweight="bold",
    )

    # ── Inset: India locator ──────────────────────────────────────────────────
    axin = ax.inset_axes([0.02, 0.02, 0.22, 0.22])
    india_rect = plt.Rectangle((68, 6), 29, 30,
                                linewidth=1, edgecolor="black",
                                facecolor="#DDDDDD")
    hp_rect = plt.Rectangle((75.5, 30.2), 3.5, 3.1,
                             linewidth=1.5, edgecolor="red",
                             facecolor="#FF6B6B", alpha=0.7)
    axin.add_patch(india_rect)
    axin.add_patch(hp_rect)
    axin.set_xlim(65, 100)
    axin.set_ylim(5, 40)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_title("India", fontsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=c, label=b) for b, c in BASIN_COLOURS.items()
    ]
    handles.append(mpatches.Patch(color="white", label=""))
    handles.append(plt.Line2D([0], [0], linestyle="--", color="#CC4444",
                               label="National Highway"))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, title="River Basins / Roads", title_fontsize=8)

    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    out = PAPER_DIR / "fig01_study_area.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Fig 1 saved -> {out}")


if __name__ == "__main__":
    main()
