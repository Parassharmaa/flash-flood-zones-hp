"""
Generate Fig 1: Study area map (HP state boundary + districts + river basins).
Uses the downloaded boundaries GeoJSON directly.

Output: results/paper_figures/fig01_study_area.png
"""

import sys
import json
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))
from config import PAPER_DIR, BOUNDARIES_DIR  # noqa: E402

# River basin colours
BASIN_COLOURS = {
    "Beas":   "#4ECDC4",
    "Sutlej": "#45B7D1",
    "Chenab": "#96CEB4",
    "Ravi":   "#FFEAA7",
    "Yamuna": "#DDA0DD",
}

def get_india_polygons() -> list[list[tuple]]:
    """
    Load India's boundary from local cache (Natural Earth 50m, downloaded once).
    Returns all rings: mainland + Andaman & Nicobar + Lakshadweep.
    """
    cache_path = BOUNDARIES_DIR / "india_outline_50m.json"

    if cache_path.exists():
        rings = json.loads(cache_path.read_text())
        return [[(c[0], c[1]) for c in ring] for ring in rings]

    # Download Natural Earth 50m (includes all islands)
    ne_url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
              "/master/geojson/ne_50m_admin_0_countries.geojson")
    print("  Fetching India boundary from Natural Earth 50m (one-time download)...")
    with urllib.request.urlopen(ne_url, timeout=30) as resp:
        data = json.loads(resp.read())
    india = next(
        f for f in data["features"]
        if f["properties"].get("ISO_A3") == "IND"
           or f["properties"].get("NAME") == "India"
    )
    geom = india["geometry"]
    if geom["type"] == "Polygon":
        rings = [geom["coordinates"][0]]
    else:  # MultiPolygon — mainland + all islands
        rings = [p[0] for p in geom["coordinates"]]
    cache_path.write_text(json.dumps(rings))
    print(f"  Cached {len(rings)} rings → {cache_path}")
    return [[(c[0], c[1]) for c in ring] for ring in rings]


def get_disputed_polygons() -> list[list[tuple]]:
    """
    Load India's disputed territories (POK / Aksai Chin) from Natural Earth.
    Tries 10m first (has explicit disputed_areas layer), falls back to empty list.
    """
    cache_path = BOUNDARIES_DIR / "india_disputed_10m.json"

    if cache_path.exists():
        rings = json.loads(cache_path.read_text())
        return [[(c[0], c[1]) for c in ring] for ring in rings]

    ne_url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
              "/master/geojson/ne_10m_admin_0_disputed_areas.geojson")
    print("  Fetching disputed areas from Natural Earth 10m...")
    try:
        with urllib.request.urlopen(ne_url, timeout=45) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Could not fetch disputed areas ({e}); skipping.")
        cache_path.write_text("[]")
        return []

    # Features administered by Pakistan (POK) or China (Aksai Chin)
    # that India claims
    keep_admins = {"Pakistan", "China"}
    # rough bounding box of India's disputed northern territories
    # lon 71–81°E, lat 32–37.5°N
    rings = []
    for feat in data["features"]:
        props = feat.get("properties", {})
        admin = props.get("ADMIN", "") or props.get("admin", "") or ""
        geom = feat.get("geometry")
        if not geom or admin not in keep_admins:
            continue
        # filter by bbox — only features inside the disputed northern region
        coords_flat = []
        if geom["type"] == "Polygon":
            coords_flat = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            for p in geom["coordinates"]:
                coords_flat.extend(p[0])
        lons = [c[0] for c in coords_flat]
        lats = [c[1] for c in coords_flat]
        if not lons:
            continue
        # Keep only features in the north-west Himalayan disputed region
        if min(lats) > 25 and max(lats) < 40 and min(lons) > 68 and max(lons) < 82:
            if geom["type"] == "Polygon":
                rings.append(geom["coordinates"][0])
            elif geom["type"] == "MultiPolygon":
                for p in geom["coordinates"]:
                    rings.append(p[0])

    cache_path.write_text(json.dumps(rings))
    print(f"  Cached {len(rings)} disputed rings → {cache_path}")
    return [[(c[0], c[1]) for c in ring] for ring in rings]

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
        "Sutlej": [(78.50, 31.85), (77.60, 31.65), (76.40, 31.20)],
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

    # ── NH-3: Chandigarh–Manali–Leh (single combined route after 2010 renumbering) ──
    nh3 = [
        (76.82, 30.70), (77.00, 31.40), (77.10, 31.83),  # Chandigarh→Manali
        (77.10, 32.10), (77.10, 32.55), (77.08, 32.90), (77.12, 33.10),  # Manali→Leh
    ]
    xs = [p[0] for p in nh3]
    ys = [p[1] for p in nh3]
    ax.plot(xs, ys, "--", color="#CC4444", linewidth=1.2, alpha=0.7, zorder=4,
            label="NH-3 (Chandigarh--Leh)")

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
    axin = ax.inset_axes([0.02, 0.02, 0.25, 0.28])
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    # Union all India rings (mainland + islands + disputed territories) so only
    # the outer boundary is drawn — no internal lines between regions
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    india_rings = get_india_polygons()
    disputed_rings = get_disputed_polygons()
    all_polys = [ShapelyPolygon(r) for r in india_rings + disputed_rings if len(r) >= 3]
    india_union = unary_union(all_polys)

    union_geoms = (list(india_union.geoms)
                   if india_union.geom_type == "MultiPolygon"
                   else [india_union])
    india_patches = [MplPolygon(list(g.exterior.coords), closed=True)
                     for g in union_geoms]
    india_col = PatchCollection(india_patches, facecolor="#D5D5D5",
                                edgecolor="#555555", linewidth=0.6, zorder=1)
    axin.add_collection(india_col)

    # HP highlight — actual state polygon
    state_path = BOUNDARIES_DIR / "hp_state.geojson"
    if state_path.exists():
        hp_gj = load_geojson(state_path)
        hp_feats = hp_gj.get("features", [hp_gj])
        hp_patches = []
        for feat in hp_feats:
            geom = feat.get("geometry", feat)
            if geom.get("type") == "Polygon":
                hp_patches.append(MplPolygon(geom["coordinates"][0], closed=True))
            elif geom.get("type") == "MultiPolygon":
                for part in geom["coordinates"]:
                    hp_patches.append(MplPolygon(part[0], closed=True))
        hp_col = PatchCollection(hp_patches, facecolor="#FF4444",
                                 edgecolor="#CC0000", linewidth=1.0,
                                 alpha=0.9, zorder=2)
        axin.add_collection(hp_col)
    axin.text(77.4, 31.7, "HP", ha="center", va="center",
              fontsize=6, fontweight="bold", color="white", zorder=3)
    axin.set_xlim(63, 100)
    axin.set_ylim(5, 40)   # extended north to ~40°N to include POK/Aksai Chin
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_title("India", fontsize=7, pad=2)
    axin.set_aspect("equal")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=c, label=b) for b, c in BASIN_COLOURS.items()]
    handles.append(plt.Line2D([0], [0], linestyle="--", color="#CC4444",
                               linewidth=1.5, label="NH-3 (Chandigarh–Leh)"))
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
