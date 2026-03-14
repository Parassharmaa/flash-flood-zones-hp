"""
Phase 2 — Assemble all conditioning factors into a single analysis-ready stack.

1. Resample all rasters to common grid (30m, UTM 43N, HP extent)
2. Multicollinearity check (Pearson correlation + VIF)
3. Export final factor stack as multi-band GeoTIFF
4. Aggregate factors to watershed level for GNN node features

Inputs:  data/processed/terrain/*.tif + data/raw/{lulc,rainfall,soil}/*.tif
Outputs:
  data/processed/conditioning_factors/factor_stack.tif  — pixel-level stack
  data/processed/conditioning_factors/factor_names.json
  data/processed/conditioning_factors/correlation_matrix.png
  data/processed/conditioning_factors/vif_results.csv
  data/processed/watershed_graph/node_features.csv      — watershed-level factors
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    TERRAIN_DIR, FACTORS_DIR, GRAPH_DIR,
    LULC_DIR, RAINFALL_DIR, SOIL_DIR, RESULTS,
    HP_CRS_UTM, HP_PIXEL_M,
)

# Factor definitions: (name, source_path_glob, band_index)
FACTOR_SOURCES = [
    ("elevation",         TERRAIN_DIR / "dem_hp.tif",           1),
    ("slope",             TERRAIN_DIR / "slope.tif",             1),
    ("aspect",            TERRAIN_DIR / "aspect.tif",            1),
    ("plan_curvature",    TERRAIN_DIR / "plan_curvature.tif",    1),
    ("profile_curvature", TERRAIN_DIR / "profile_curvature.tif", 1),
    ("twi",               TERRAIN_DIR / "twi.tif",               1),
    ("spi",               TERRAIN_DIR / "spi.tif",               1),
    ("tri",               TERRAIN_DIR / "tri.tif",               1),
    # These are added after GEE download:
    ("rainfall_mean",     RAINFALL_DIR / "rainfall_mean_annual_gpm.tif", 1),
    ("rainfall_extreme",  RAINFALL_DIR / "rainfall_extreme_p95_gpm.tif", 1),
    ("lulc",              LULC_DIR / "lulc_hp.tif",              1),
    ("soil_clay",         SOIL_DIR / "clay_0_30cm_hp.tif",       1),
]

VIF_THRESHOLD = 10.0
CORR_THRESHOLD = 0.80


def resample_to_reference(src_path: Path, ref_path: Path, out_path: Path) -> None:
    """Resample a raster to match the reference grid."""
    with rasterio.open(ref_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_width     = ref.width
        dst_height    = ref.height
        dst_nodata    = -9999.0

    with rasterio.open(src_path) as src:
        kwargs = {
            "driver":    "GTiff",
            "dtype":     "float32",
            "width":     dst_width,
            "height":    dst_height,
            "count":     1,
            "crs":       dst_crs,
            "transform": dst_transform,
            "nodata":    dst_nodata,
            "compress":  "lzw",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata,
            )


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each column."""
    from sklearn.linear_model import LinearRegression

    vifs = {}
    cols = df.columns.tolist()
    for i, col in enumerate(cols):
        X = df.drop(columns=[col]).values
        y = df[col].values
        lr = LinearRegression().fit(X, y)
        r2 = lr.score(X, y)
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return pd.DataFrame({"factor": list(vifs.keys()),
                         "vif":    list(vifs.values())}).sort_values("vif", ascending=False)


def multicollinearity_check(data: dict[str, np.ndarray]) -> list[str]:
    """
    Remove factors with |r| > CORR_THRESHOLD or VIF > VIF_THRESHOLD.
    Returns list of retained factor names.
    """
    print("\nMulticollinearity check...")
    names = list(data.keys())
    arrays = list(data.values())

    # Sample 10,000 valid pixels
    n_px = arrays[0].size
    idx  = np.random.choice(n_px, min(10_000, n_px), replace=False)
    mat  = np.column_stack([a.ravel()[idx] for a in arrays])

    # Remove rows with any NaN / nodata
    valid = np.all(np.isfinite(mat) & (mat != -9999), axis=1)
    mat   = mat[valid]

    df = pd.DataFrame(mat, columns=names)

    # Step 1: Pearson correlation
    corr = df.corr()
    corr_path = FACTORS_DIR / "correlation_matrix.png"
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Conditioning factor correlation matrix (n=10,000 pixels)\nThreshold = ±{CORR_THRESHOLD}")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Correlation matrix → {corr_path}")

    # Remove one from each highly-correlated pair
    to_drop = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr.iloc[i, j]) > CORR_THRESHOLD and names[j] not in to_drop:
                to_drop.add(names[j])
                print(f"  Drop (corr={corr.iloc[i,j]:.2f}): {names[j]} (correlated with {names[i]})")

    retained = [n for n in names if n not in to_drop]
    df_ret   = df[retained]

    # Step 2: VIF
    vif_df = compute_vif(df_ret)
    vif_path = FACTORS_DIR / "vif_results.csv"
    vif_df.to_csv(vif_path, index=False)
    print(f"  VIF results → {vif_path}")

    while True:
        high_vif = vif_df[vif_df["vif"] > VIF_THRESHOLD]
        if high_vif.empty:
            break
        worst = high_vif.iloc[0]["factor"]
        print(f"  Drop (VIF={high_vif.iloc[0]['vif']:.1f}): {worst}")
        retained.remove(worst)
        df_ret = df[retained]
        vif_df = compute_vif(df_ret)

    print(f"\n  Final factors ({len(retained)}): {retained}")
    return retained


def build_factor_stack(retained: list[str], factor_data: dict[str, np.ndarray],
                       ref_path: Path) -> Path:
    """Write multi-band factor stack GeoTIFF."""
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()

    meta.update({
        "count":   len(retained),
        "dtype":   "float32",
        "nodata":  -9999,
        "compress": "lzw",
    })
    stack_path = FACTORS_DIR / "factor_stack.tif"
    with rasterio.open(stack_path, "w", **meta) as dst:
        for i, name in enumerate(retained, start=1):
            dst.write(factor_data[name].astype(np.float32), i)
            dst.update_tags(i, name=name)

    (FACTORS_DIR / "factor_names.json").write_text(
        json.dumps({"factors": retained, "n_factors": len(retained)}, indent=2)
    )
    print(f"  Factor stack ({len(retained)} bands) → {stack_path}")
    return stack_path


def aggregate_to_watersheds(retained: list[str], factor_data: dict[str, np.ndarray],
                             ref_path: Path) -> None:
    """Compute mean factor value per watershed for GNN node features."""
    ws_path = GRAPH_DIR / "watersheds.geojson"
    if not ws_path.exists():
        print("  Watershed file not found — skipping watershed aggregation")
        return

    print("\nAggregating factors to watershed level...")
    from rasterio.features import rasterize

    watersheds = gpd.read_file(ws_path)
    with rasterio.open(ref_path) as src:
        transform = src.transform
        h, w      = src.height, src.width

    rows = []
    from shapely.geometry import mapping
    for _, ws in watersheds.iterrows():
        mask = rasterize(
            [(mapping(ws.geometry), 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0, dtype=np.uint8,
        )
        row = {"watershed_id": ws["watershed_id"]}
        for name in retained:
            vals = factor_data[name][mask == 1]
            valid_vals = vals[(vals != -9999) & np.isfinite(vals)]
            row[f"mean_{name}"] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
        rows.append(row)

    node_df = pd.DataFrame(rows)
    out = GRAPH_DIR / "node_features.csv"
    node_df.to_csv(out, index=False)
    print(f"  Node features ({len(node_df)} watersheds × {len(retained)} factors) → {out}")


def main() -> None:
    print("=" * 60)
    print("Phase 2: Assemble conditioning factors")
    print("=" * 60)

    # Reference grid = DEM
    ref_path = TERRAIN_DIR / "dem_hp.tif"
    if not ref_path.exists():
        raise FileNotFoundError(f"DEM not found: {ref_path}. Run 04_preprocess_terrain.py first.")

    # Load available factors
    factor_data = {}
    for name, src_path, band in FACTOR_SOURCES:
        if not src_path.exists():
            print(f"  SKIP (not yet downloaded): {name} ({src_path.name})")
            continue
        resampled = FACTORS_DIR / f"{name}_30m.tif"
        if not resampled.exists():
            resample_to_reference(src_path, ref_path, resampled)
        with rasterio.open(resampled) as src:
            arr = src.read(band).astype(np.float32)
        factor_data[name] = arr
        print(f"  Loaded: {name} {arr.shape}")

    if len(factor_data) < 4:
        print(f"\nWARNING: Only {len(factor_data)} factors available.")
        print("Download more data first (run 02_download_rasters.py + GEE exports).")
        print("Saving available factors and continuing...")

    if factor_data:
        retained = multicollinearity_check(factor_data)
        stack    = build_factor_stack(retained, factor_data, ref_path)
        aggregate_to_watersheds(retained, factor_data, ref_path)
    else:
        print("No factor data available. Exiting.")
        return

    print("\nFactor assembly complete. Next: run 07_build_flood_inventory.py")


if __name__ == "__main__":
    main()
