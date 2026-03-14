"""
Phase 1 — Download raster conditioning factors.

Sources:
  DEM:      Copernicus GLO-30 (30m, free via AWS)
  Rainfall: GPM IMERG via GEE export OR direct download
  LULC:     ESA WorldCover 2021 (10m) via direct download
  Soil:     SoilGrids REST API (250m)

Run AFTER 01_download_boundaries.py.

NOTE: GEE-based downloads (Sentinel-1 SAR, GPM) require
`earthengine authenticate` to be run once before this script.
"""

import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DEM_DIR, LULC_DIR, RAINFALL_DIR, SOIL_DIR,
    HP_BBOX,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download a file with progress bar."""
    if dest.exists():
        print(f"  Already exists, skipping: {dest.name}")
        return
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


# ── DEM: Copernicus GLO-30 ────────────────────────────────────────────────────

def download_dem_copernicus() -> None:
    """
    Download Copernicus GLO-30 DEM tiles covering HP bbox.
    Tiles are 1°×1° at 1 arc-second (~30m).
    Free access via AWS S3 (no auth required).
    """
    print("\nDownloading Copernicus GLO-30 DEM tiles...")

    # HP spans lat 30–34, lon 75–79 → tiles needed
    import math
    lats = range(math.floor(HP_BBOX["ymin"]), math.ceil(HP_BBOX["ymax"]))
    lons = range(math.floor(HP_BBOX["xmin"]), math.ceil(HP_BBOX["xmax"]))

    # Copernicus GLO-30 on AWS S3 (public, no auth)
    # URL pattern: .../Copernicus_DSM_COG_10_N30_00_E075_00_DEM/...tif
    aws_base = "https://copernicus-dem-30m.s3.amazonaws.com"

    downloaded = 0
    for lat in lats:
        for lon in lons:
            lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            stem = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
            filename = f"{stem}.tif"
            url = f"{aws_base}/{stem}/{filename}"
            dest = DEM_DIR / filename
            try:
                download_file(url, dest, desc=f"DEM {lat_str}{lon_str}")
                downloaded += 1
            except Exception as e:
                print(f"  WARNING: could not download {filename}: {e}")

    print(f"  Downloaded {downloaded} DEM tiles to {DEM_DIR}")


# ── LULC: ESA WorldCover 2021 ─────────────────────────────────────────────────

def download_esa_worldcover() -> None:
    """
    ESA WorldCover 2021 (10m, free).
    Download HP-covering tiles from AWS S3.
    Tile naming: ESA_WorldCover_10m_2021_v200_<tile>.tif
    """
    print("\nDownloading ESA WorldCover 2021 LULC tiles...")

    # WorldCover tiles are 3°×3° (e.g., N30E075)
    import math
    lat_starts = range(
        (math.floor(HP_BBOX["ymin"]) // 3) * 3,
        math.ceil(HP_BBOX["ymax"]),
        3,
    )
    lon_starts = range(
        (math.floor(HP_BBOX["xmin"]) // 3) * 3,
        math.ceil(HP_BBOX["xmax"]),
        3,
    )

    base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"

    for lat in lat_starts:
        for lon in lon_starts:
            lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            filename = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
            url = f"{base}{filename}"
            dest = LULC_DIR / filename
            try:
                download_file(url, dest, desc=f"LULC {lat_str}{lon_str}")
            except Exception as e:
                print(f"  WARNING: {filename}: {e}")


# ── Soil: SoilGrids REST API ──────────────────────────────────────────────────

def download_soilgrids() -> None:
    """
    Download soil texture / type from SoilGrids 2.0 via WCS REST API.
    Gets clay content 0-30cm as a proxy for infiltration capacity.
    """
    print("\nDownloading SoilGrids clay content (0-30cm)...")

    # SoilGrids WCS endpoint
    wcs_url = "https://maps.isric.org/mapserv?map=/map/clay.map"
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": "clay_0-30cm_mean",
        "FORMAT": "image/tiff",
        "SUBSET": f"X({HP_BBOX['xmin']},{HP_BBOX['xmax']})",
        "SUBSETTING_CRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "OUTPUT_CRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }
    # Add lat subset separately (SoilGrids uses Y for latitude)
    params["SUBSET2"] = f"Y({HP_BBOX['ymin']},{HP_BBOX['ymax']})"

    dest = SOIL_DIR / "clay_0_30cm_hp.tif"
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    try:
        resp = requests.get(wcs_url, params=params, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  SoilGrids clay → {dest}")
    except Exception as e:
        print(f"  WARNING: SoilGrids download failed: {e}")
        # Write manual download note
        note = SOIL_DIR / "MANUAL_SOILGRIDS.txt"
        note.write_text(
            "Manual download: https://soilgrids.org\n"
            "Layer: clay_0-30cm_mean\n"
            f"Bbox: {HP_BBOX}\n"
            "Resolution: 250m\n"
        )


# ── Instructions for GEE-based downloads ──────────────────────────────────────

def write_gee_instructions() -> None:
    """Write instructions for datasets that require GEE."""
    note = RAINFALL_DIR / "DOWNLOAD_INSTRUCTIONS.txt"
    note.write_text(
        "GPM IMERG Rainfall — GEE Export\n"
        "================================\n"
        "Run: uv run python scripts/03_gee_exports.py\n\n"
        "Requires: earthengine authenticate (run once)\n\n"
        "Exports to Google Drive then download to data/raw/rainfall/\n\n"
        "Layers:\n"
        "  1. Mean annual precipitation 2001-2023\n"
        "  2. 95th percentile daily precipitation (extreme events)\n"
        "  3. Annual precipitation 2018-2024 (flood inventory period)\n"
    )
    print(f"\n  GEE rainfall instructions → {note}")


def main() -> None:
    print("=" * 60)
    print("Phase 1b: Downloading raster conditioning factors")
    print("=" * 60)
    download_dem_copernicus()
    download_esa_worldcover()
    download_soilgrids()
    write_gee_instructions()
    print("\nDone. Run scripts/03_gee_exports.py for GEE-based data.")


if __name__ == "__main__":
    main()
