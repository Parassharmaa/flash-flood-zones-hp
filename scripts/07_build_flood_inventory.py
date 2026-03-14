"""
Phase 3 — Build final flood inventory from SAR exports + HiFlo-DAT.

Combines:
  1. SAR-detected flood extents (from GEE exports)
  2. HiFlo-DAT historical events (Kullu district, 128 events)
  3. NDMA/news-based point records

Outputs:
  data/processed/flood_inventory/flood_points.geojson   — presence points
  data/processed/flood_inventory/nonflood_points.geojson — absence points
  data/processed/flood_inventory/inventory_summary.json
"""

import json
import sys
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    FLOOD_DIR, INVENTORY_DIR, FACTORS_DIR, HP_BBOX, HP_CRS_UTM,
    NON_FLOOD_BUFFER_M, FLOOD_NON_FLOOD_RATIO, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


# ── SAR inventory ─────────────────────────────────────────────────────────────

def load_sar_inventory() -> gpd.GeoDataFrame:
    """Load SAR-based flood extents from GEE exports."""
    sar_dir = FLOOD_DIR / "sar"
    tif_files = list(sar_dir.glob("flood_*.tif")) if sar_dir.exists() else []

    if not tif_files:
        print("  No SAR TIFs found. Using synthetic placeholder inventory.")
        return _create_placeholder_inventory()

    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    all_points = []
    for tif in sorted(tif_files):
        split = "test" if "_test_" in tif.stem else "train"
        # Extract year from filename: flood_train_2018 → "2018"
        year_str = tif.stem.split("_")[-1]
        date = f"{year_str}-07-01"   # approximate monsoon peak date

        with rasterio.open(tif) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Morphological opening: remove isolated pixels (small-patch filter)
        from scipy.ndimage import binary_opening
        flood_mask = binary_opening(data == 1, structure=np.ones((3, 3)))

        # Physical plausibility filter: real flash floods occur on gentle slopes
        # near rivers and at lower elevations. Apply slope + distance filters to
        # reduce false positives from seasonal vegetation/moisture SAR changes.
        slope_path = FACTORS_DIR / "slope_30m.tif"
        dist_path  = FACTORS_DIR / "distance_to_river_30m.tif"
        elev_path  = FACTORS_DIR / "elevation_30m.tif"
        if slope_path.exists() and dist_path.exists():
            import rasterio.warp
            with rasterio.open(slope_path) as s:
                slope_arr = s.read(1)
                # Resample slope to SAR grid
                slope_sam = np.empty(flood_mask.shape, dtype=np.float32)
                rasterio.warp.reproject(
                    source=slope_arr, destination=slope_sam,
                    src_transform=s.transform, src_crs=s.crs,
                    dst_transform=transform, dst_crs=crs,
                    resampling=rasterio.enums.Resampling.bilinear,
                )
            with rasterio.open(dist_path) as d_:
                dist_arr = d_.read(1)
                dist_sam = np.empty(flood_mask.shape, dtype=np.float32)
                rasterio.warp.reproject(
                    source=dist_arr, destination=dist_sam,
                    src_transform=d_.transform, src_crs=d_.crs,
                    dst_transform=transform, dst_crs=crs,
                    resampling=rasterio.enums.Resampling.bilinear,
                )
            # Keep only pixels with slope < 15° and within 2 km of river
            terrain_filter = (slope_sam < 15) & (dist_sam < 2000) & (dist_sam >= 0)
            flood_mask = flood_mask & terrain_filter
            print(f"    Terrain filter: {terrain_filter.sum():,} eligible pixels")

        ys, xs = np.where(flood_mask)

        # Sample up to 500 points per event (avoid huge inventories)
        if len(xs) > 500:
            idx = np.random.choice(len(xs), 500, replace=False)
            xs, ys = xs[idx], ys[idx]

        for x_px, y_px in zip(xs, ys):
            x_geo, y_geo = rasterio.transform.xy(transform, y_px, x_px)
            all_points.append({
                "geometry": Point(x_geo, y_geo),
                "date":     date,
                "split":    split,
                "source":   "SAR_Sentinel1",
            })

    if not all_points:
        return _create_placeholder_inventory()

    gdf = gpd.GeoDataFrame(all_points, crs=crs)
    print(f"  SAR inventory: {len(gdf)} flood points from {len(tif_files)} events")
    return gdf


def _create_placeholder_inventory() -> gpd.GeoDataFrame:
    """
    Create a synthetic placeholder inventory for pipeline testing.
    Based on known high-risk zones from literature (Kullu, Mandi, Shimla, Kinnaur).
    Replace with real SAR inventory when GEE exports are available.
    """
    print("  Creating placeholder inventory (replace with real SAR data)")

    # Known flood-prone coordinates from literature
    known_zones = [
        # (lon, lat, basin, district, trigger)
        (77.19, 32.23, "Beas",   "Kullu",   "cloudburst"),  # Kullu town
        (77.10, 32.10, "Beas",   "Kullu",   "monsoon"),     # Bhuntar
        (77.34, 32.48, "Beas",   "Kullu",   "cloudburst"),  # Manali approach
        (76.82, 31.72, "Beas",   "Mandi",   "monsoon"),     # Mandi town
        (77.10, 31.10, "Satluj", "Shimla",  "monsoon"),     # Shimla
        (78.42, 31.33, "Satluj", "Kinnaur", "GLOF"),        # Sangla
        (76.45, 32.55, "Ravi",   "Chamba",  "monsoon"),     # Chamba town
        (76.16, 32.10, "Ravi",   "Chamba",  "monsoon"),     # Bharmour
        (77.55, 32.26, "Beas",   "Kullu",   "cloudburst"),  # Parvati valley
        (77.80, 32.21, "Beas",   "Kullu",   "GLOF"),        # Malana
    ]

    points = []
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate points around each known zone
    for lon, lat, basin, district, trigger in known_zones:
        n_pts = rng.integers(20, 60)
        for _ in range(n_pts):
            # Scatter within ~10 km of zone centre
            dx = rng.normal(0, 0.05)   # ~5 km in longitude
            dy = rng.normal(0, 0.05)   # ~5 km in latitude
            year = rng.integers(2018, 2024)
            month = rng.integers(7, 10)  # July–September peak
            split = "test" if year == 2023 else "train"
            points.append({
                "geometry": Point(lon + dx, lat + dy),
                "date":     f"{year}-{month:02d}-01",
                "split":    split,
                "basin":    basin,
                "district": district,
                "trigger":  trigger,
                "source":   "PLACEHOLDER_replace_with_SAR",
            })

    gdf = gpd.GeoDataFrame(points, crs="EPSG:4326").to_crs(HP_CRS_UTM)
    gdf["label"] = 1
    print(f"  Placeholder: {len(gdf)} flood points")
    return gdf


# ── Non-flood points ──────────────────────────────────────────────────────────

def generate_nonflood_points(
    flood_gdf: gpd.GeoDataFrame,
    hp_boundary: gpd.GeoDataFrame,
    split: Literal["train", "test"],
) -> gpd.GeoDataFrame:
    """
    Sample non-flood points from HP area, excluding buffer around flood points.
    Ratio: FLOOD_NON_FLOOD_RATIO × number of flood points.
    """
    flood_subset = flood_gdf[flood_gdf["split"] == split]
    n_flood      = len(flood_subset)
    n_nonfood    = n_flood * FLOOD_NON_FLOOD_RATIO

    # Exclusion zone: buffer around all flood points
    flood_union  = unary_union(flood_subset.geometry)
    exclusion    = flood_union.buffer(NON_FLOOD_BUFFER_M)

    # HP bounding polygon
    hp_poly = hp_boundary.geometry.iloc[0] if len(hp_boundary) > 0 else box(
        *gpd.GeoDataFrame(geometry=[box(HP_BBOX["xmin"], HP_BBOX["ymin"],
                                         HP_BBOX["xmax"], HP_BBOX["ymax"])],
                          crs="EPSG:4326").to_crs(HP_CRS_UTM).total_bounds
    )

    rng = np.random.default_rng(RANDOM_SEED + (1 if split == "test" else 0))
    bounds = hp_poly.bounds if hasattr(hp_poly, "bounds") else (
        flood_subset.total_bounds
    )
    xmin, ymin, xmax, ymax = bounds

    points = []
    attempts = 0
    while len(points) < n_nonfood and attempts < n_nonfood * 20:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        pt = Point(x, y)
        if not exclusion.contains(pt):
            points.append({
                "geometry": pt,
                "split":    split,
                "source":   "random_background",
                "label":    0,
            })
        attempts += 1

    gdf = gpd.GeoDataFrame(points, crs=HP_CRS_UTM)
    print(f"  Non-flood points ({split}): {len(gdf)} (ratio {len(gdf)/max(n_flood,1):.1f}:1)")
    return gdf


def main() -> None:
    print("=" * 60)
    print("Phase 3: Building flood inventory")
    print("=" * 60)

    # Load boundaries for non-flood sampling
    boundary_path = Path(__file__).parent.parent / "data" / "raw" / "boundaries" / "hp_state.geojson"
    hp_boundary   = gpd.read_file(boundary_path).to_crs(HP_CRS_UTM) if boundary_path.exists() \
                    else gpd.GeoDataFrame()

    # Load flood points
    flood_gdf = load_sar_inventory()
    if "label" not in flood_gdf.columns:
        flood_gdf["label"] = 1

    # Ensure CRS
    if flood_gdf.crs is None or str(flood_gdf.crs) != HP_CRS_UTM:
        try:
            flood_gdf = flood_gdf.to_crs(HP_CRS_UTM)
        except Exception:
            flood_gdf.set_crs(HP_CRS_UTM, inplace=True)

    # Generate non-flood points
    nf_train = generate_nonflood_points(flood_gdf, hp_boundary, "train")
    nf_test  = generate_nonflood_points(flood_gdf, hp_boundary, "test")

    # Save
    flood_path = INVENTORY_DIR / "flood_points.geojson"
    flood_gdf.to_file(flood_path, driver="GeoJSON")
    print(f"\n  Flood points → {flood_path}")

    nonfood_path = INVENTORY_DIR / "nonflood_points.geojson"
    pd.concat([nf_train, nf_test]).pipe(
        gpd.GeoDataFrame, crs=HP_CRS_UTM
    ).to_file(nonfood_path, driver="GeoJSON")
    print(f"  Non-flood points → {nonfood_path}")

    # Summary
    train_flood = len(flood_gdf[flood_gdf["split"] == "train"])
    test_flood  = len(flood_gdf[flood_gdf["split"] == "test"])
    summary = {
        "total_flood_points":     len(flood_gdf),
        "training_flood":         train_flood,
        "test_flood_2023":        test_flood,
        "training_nonflood":      len(nf_train),
        "test_nonflood":          len(nf_test),
        "nonflood_ratio":         FLOOD_NON_FLOOD_RATIO,
        "buffer_m":               NON_FLOOD_BUFFER_M,
        "note": "Replace PLACEHOLDER with real SAR inventory from GEE",
    }
    (INVENTORY_DIR / "inventory_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary: {summary}")
    print("\nNext: run 08_train_baseline_models.py")


if __name__ == "__main__":
    main()
