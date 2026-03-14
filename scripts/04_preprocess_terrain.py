"""
Phase 2 — Terrain preprocessing: derive all conditioning factors from DEM.

Inputs:
  data/raw/dem/*.tif               — Copernicus GLO-30 tiles

Outputs (all at 30m, UTM 43N):
  data/processed/terrain/dem_hp.tif
  data/processed/terrain/slope.tif
  data/processed/terrain/aspect.tif
  data/processed/terrain/plan_curvature.tif
  data/processed/terrain/profile_curvature.tif
  data/processed/terrain/twi.tif
  data/processed/terrain/spi.tif
  data/processed/terrain/tri.tif
  data/processed/terrain/drainage_density.tif
  data/processed/terrain/distance_to_river.tif
  data/processed/watershed_graph/watersheds.geojson
  data/processed/watershed_graph/graph_edges.csv
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DEM_DIR, TERRAIN_DIR, HP_BBOX, HP_CRS_UTM, HP_PIXEL_M, GRAPH_DIR
)


def merge_dem_tiles() -> Path:
    """Merge Copernicus GLO-30 tiles and reproject to UTM 43N."""
    print("Merging DEM tiles...")
    tiles = list(DEM_DIR.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(
            f"No DEM tiles in {DEM_DIR}. Run 02_download_rasters.py first."
        )

    # Merge tiles
    datasets = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                     "width": mosaic.shape[2], "transform": out_transform})
    for ds in datasets:
        ds.close()

    merged_path = TERRAIN_DIR / "dem_merged.tif"
    with rasterio.open(merged_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    print(f"  Merged {len(tiles)} tiles → {merged_path}")

    # Reproject to UTM 43N at target resolution
    utm_path = TERRAIN_DIR / "dem_hp.tif"
    with rasterio.open(merged_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, HP_CRS_UTM, src.width, src.height, *src.bounds,
            resolution=HP_PIXEL_M,
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": HP_CRS_UTM,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": "lzw",
        })
        with rasterio.open(utm_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=HP_CRS_UTM,
                resampling=Resampling.bilinear,
            )
    print(f"  Reprojected to UTM 43N → {utm_path}")
    merged_path.unlink()  # clean up intermediate
    return utm_path


def compute_slope_aspect(dem_path: Path) -> tuple[Path, Path]:
    """Compute slope (degrees) and aspect from DEM using numpy gradient."""
    print("Computing slope and aspect...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        res = src.res[0]  # pixel size in metres (UTM)

    # Replace nodata
    dem[dem == meta.get("nodata", -9999)] = np.nan

    # Gradient in x and y
    dy, dx = np.gradient(dem, res, res)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    # Aspect in degrees (0 = north, clockwise)
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360

    meta.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})

    slope_path = TERRAIN_DIR / "slope.tif"
    aspect_path = TERRAIN_DIR / "aspect.tif"
    for arr, path in [(slope, slope_path), (aspect, aspect_path)]:
        arr = np.where(np.isnan(arr), -9999, arr).astype(np.float32)
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr[np.newaxis, :, :])

    print(f"  Slope → {slope_path}")
    print(f"  Aspect → {aspect_path}")
    return slope_path, aspect_path


def compute_curvature(dem_path: Path) -> tuple[Path, Path]:
    """Compute plan and profile curvature from DEM."""
    print("Computing curvature...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        meta = src.meta.copy()
        res = float(src.res[0])

    nodata = float(meta.get("nodata", -9999))
    dem[dem == nodata] = np.nan

    # Second-order derivatives using central differences
    dx  = np.gradient(dem, axis=1) / res
    dy  = np.gradient(dem, axis=0) / res
    dxx = np.gradient(dx, axis=1) / res
    dyy = np.gradient(dy, axis=0) / res
    dxy = np.gradient(dx, axis=0) / res

    p = dx**2 + dy**2
    q = p + 1

    # Plan curvature (contour curvature) — flow divergence
    plan = -(dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / (p * np.sqrt(q) + 1e-10)
    # Profile curvature — flow acceleration
    profile = -(dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / (p * np.sqrt(q**3) + 1e-10)

    meta.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})

    plan_path    = TERRAIN_DIR / "plan_curvature.tif"
    profile_path = TERRAIN_DIR / "profile_curvature.tif"
    for arr, path in [(plan, plan_path), (profile, profile_path)]:
        arr = np.where(np.isnan(arr), -9999, arr).astype(np.float32)
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr[np.newaxis, :, :])

    print(f"  Plan curvature → {plan_path}")
    print(f"  Profile curvature → {profile_path}")
    return plan_path, profile_path


def compute_twi_spi(dem_path: Path, slope_path: Path) -> tuple[Path, Path]:
    """
    Compute TWI (Topographic Wetness Index) and SPI (Stream Power Index).

    TWI = ln(A / tan(slope))  where A = specific catchment area
    SPI = A * tan(slope)

    Uses flow accumulation from pysheds for specific catchment area.
    Falls back to a simplified approximation if pysheds unavailable.
    """
    print("Computing TWI and SPI...")

    try:
        from pysheds.grid import Grid

        grid = Grid.from_raster(str(dem_path))
        dem_arr = grid.read_raster(str(dem_path))

        # Condition DEM
        pit_filled = grid.fill_pits(dem_arr)
        flooded    = grid.fill_depressions(pit_filled)
        inflated   = grid.resolve_flats(flooded)

        # Flow direction and accumulation (D8)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        fdir   = grid.flowdir(inflated, dirmap=dirmap)
        acc    = grid.accumulation(fdir, dirmap=dirmap)

        # Specific catchment area (pixels → m²)
        res = dem_path  # we'll read resolution
        with rasterio.open(dem_path) as src:
            cell_size = src.res[0]
        sca = acc * cell_size  # m² per unit width

        # Read slope for TWI/SPI
        with rasterio.open(slope_path) as src:
            slope_arr = src.read(1).astype(np.float64)
            meta = src.meta.copy()

        slope_rad  = np.radians(np.clip(slope_arr, 0.1, 89))  # avoid 0/inf
        twi_arr    = np.log(sca / np.tan(slope_rad) + 1e-10)
        spi_arr    = sca * np.tan(slope_rad)

        use_pysheds = True

    except ImportError:
        print("  pysheds not available — using simplified TWI approximation")
        with rasterio.open(dem_path) as src:
            dem_arr  = src.read(1).astype(np.float64)
            meta     = src.meta.copy()
            cell_size = src.res[0]
        with rasterio.open(slope_path) as src:
            slope_arr = src.read(1).astype(np.float64)

        # Simplified: use elevation rank as proxy for accumulation
        slope_rad = np.radians(np.clip(slope_arr, 0.1, 89))
        # Elevation-based drainage area proxy (higher area at lower elevations)
        elev_norm = (dem_arr.max() - dem_arr) / (dem_arr.max() - dem_arr.min() + 1)
        sca       = elev_norm * cell_size * 100
        twi_arr   = np.log(sca / np.tan(slope_rad) + 1e-10)
        spi_arr   = sca * np.tan(slope_rad)
        use_pysheds = False

    meta.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})
    twi_path = TERRAIN_DIR / "twi.tif"
    spi_path = TERRAIN_DIR / "spi.tif"

    for arr, path in [(twi_arr, twi_path), (spi_arr, spi_path)]:
        arr = np.where(np.isnan(arr) | np.isinf(arr), -9999, arr).astype(np.float32)
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr[np.newaxis, :, :])

    method = "pysheds D8" if use_pysheds else "elevation proxy"
    print(f"  TWI ({method}) → {twi_path}")
    print(f"  SPI ({method}) → {spi_path}")
    return twi_path, spi_path


def compute_tri(dem_path: Path) -> Path:
    """Terrain Ruggedness Index: mean absolute difference from 3×3 neighbourhood."""
    print("Computing TRI...")
    from scipy.ndimage import uniform_filter

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        meta = src.meta.copy()
    nodata = float(meta.get("nodata", -9999))
    dem[dem == nodata] = np.nan

    # TRI = mean absolute deviation from centre in 3×3 window
    dem_mean = uniform_filter(dem, size=3)
    tri = np.abs(dem - dem_mean)

    meta.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})
    tri_path = TERRAIN_DIR / "tri.tif"
    tri = np.where(np.isnan(tri), -9999, tri).astype(np.float32)
    with rasterio.open(tri_path, "w", **meta) as dst:
        dst.write(tri[np.newaxis, :, :])
    print(f"  TRI → {tri_path}")
    return tri_path


def compute_distance_to_river(dem_path: Path) -> Path:
    """
    Euclidean distance (metres) from each pixel to nearest stream channel.
    Streams defined by flow accumulation > threshold using pysheds.
    Falls back to elevation-based channel approximation if pysheds unavailable.
    """
    print("Computing distance to river...")
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        cell_size = src.res[0]

    try:
        from pysheds.grid import Grid

        grid    = Grid.from_raster(str(dem_path))
        dem_arr = grid.read_raster(str(dem_path))

        pit_filled = grid.fill_pits(dem_arr)
        flooded    = grid.fill_depressions(pit_filled)
        inflated   = grid.resolve_flats(flooded)

        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        fdir   = grid.flowdir(inflated, dirmap=dirmap)
        acc    = grid.accumulation(fdir, dirmap=dirmap)

        # Channels: accumulation > 5 km² / cell_size²
        threshold_px = int(5e6 / (cell_size**2))
        is_channel   = (acc > threshold_px).astype(bool)

    except ImportError:
        # Fallback: low-elevation + high-accumulation proxy
        with rasterio.open(dem_path) as src:
            dem_arr = src.read(1).astype(np.float32)
        # Approximate stream channels as lowest 5th percentile elevation
        elev_thresh = np.nanpercentile(dem_arr[dem_arr > -9998], 5)
        is_channel  = dem_arr < elev_thresh

    # EDT: distance in pixels → multiply by cell_size for metres
    dist_px  = distance_transform_edt(~is_channel)
    dist_m   = (dist_px * cell_size).astype(np.float32)
    dist_m   = np.where(~np.isfinite(dist_m), -9999, dist_m)

    meta.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})
    dist_path = TERRAIN_DIR / "distance_to_river.tif"
    with rasterio.open(dist_path, "w", **meta) as dst:
        dst.write(dist_m[np.newaxis, :, :])
    print(f"  Distance to river → {dist_path}")
    return dist_path


def main() -> None:
    print("=" * 60)
    print("Phase 2: Terrain preprocessing")
    print("=" * 60)

    dem_path = TERRAIN_DIR / "dem_hp.tif"
    if not dem_path.exists():
        dem_path = merge_dem_tiles()
    else:
        print(f"DEM already exists: {dem_path}")

    slope_path, aspect_path = compute_slope_aspect(dem_path)
    plan_path, profile_path = compute_curvature(dem_path)
    twi_path, spi_path      = compute_twi_spi(dem_path, slope_path)
    tri_path                = compute_tri(dem_path)
    dist_path               = compute_distance_to_river(dem_path)

    outputs = [dem_path, slope_path, aspect_path, plan_path,
               profile_path, twi_path, spi_path, tri_path, dist_path]
    print(f"\nTerrain factors computed: {len(outputs)}")
    print("Next: run 05_watershed_delineation.py to build the GNN graph")


if __name__ == "__main__":
    main()
