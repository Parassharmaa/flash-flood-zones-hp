"""
Phase 3 — Build SAR-based flood inventory using Sentinel-1 via GEE Python API.

Authentication (one-time setup):
  uv run earthengine authenticate
  → opens browser, saves credentials to ~/.config/earthengine/

Then run:
  uv run python scripts/03_gee_sar_inventory.py

What it does:
  1. Authenticates with the GEE Python API.
  2. For each known flood event, computes pre/post SAR backscatter change.
  3. Submits export tasks to Google Drive (folder: hp_flood_inventory).
  4. Polls until all tasks complete (can take 10–30 min per task).
  5. Downloads completed TIFs from Drive to data/raw/flood_inventory/sar/.
     (Requires: pip install google-auth google-auth-httplib2 google-api-python-client)

Outputs:
  data/raw/flood_inventory/sar/flood_train_YYYYMMDD.tif  (per training event)
  data/raw/flood_inventory/sar/flood_test_YYYYMMDD.tif   (per 2023 test event)
  data/raw/rainfall/rainfall_mean_annual_gpm.tif
  data/raw/rainfall/rainfall_extreme_p95_gpm.tif
  data/raw/flood_inventory/known_events.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import FLOOD_DIR, HP_BBOX, SAR_START_DATE, SAR_END_DATE  # noqa: E402

# Known major HP flood events
# Source: HiFlo-DAT, NDMA, Kumar 2022, NHESS 2026 analysis
KNOWN_FLOOD_EVENTS = [
    {"date": "2018-07-25", "basin": "Beas",   "district": "Kullu",   "trigger": "cloudburst"},
    {"date": "2018-08-18", "basin": "Satluj", "district": "Shimla",  "trigger": "monsoon"},
    {"date": "2019-07-31", "basin": "Beas",   "district": "Mandi",   "trigger": "cloudburst"},
    {"date": "2019-08-14", "basin": "Chenab", "district": "Chamba",  "trigger": "monsoon"},
    {"date": "2020-07-28", "basin": "Beas",   "district": "Kullu",   "trigger": "cloudburst"},
    {"date": "2021-07-26", "basin": "Beas",   "district": "Kullu",   "trigger": "monsoon"},
    {"date": "2021-08-17", "basin": "Satluj", "district": "Kinnaur", "trigger": "monsoon"},
    {"date": "2022-07-13", "basin": "Beas",   "district": "Mandi",   "trigger": "cloudburst"},
    {"date": "2022-08-24", "basin": "Ravi",   "district": "Chamba",  "trigger": "monsoon"},
    # 2023 — temporal test set (held out from training)
    {"date": "2023-07-09", "basin": "Beas",   "district": "Kullu",   "trigger": "cloudburst"},
    {"date": "2023-07-14", "basin": "Satluj", "district": "Shimla",  "trigger": "monsoon"},
    {"date": "2023-08-02", "basin": "Beas",   "district": "Mandi",   "trigger": "cloudburst"},
    {"date": "2023-08-13", "basin": "Chenab", "district": "Chamba",  "trigger": "monsoon"},
    {"date": "2023-09-01", "basin": "Satluj", "district": "Kinnaur", "trigger": "monsoon"},
    # 2024
    {"date": "2024-07-31", "basin": "Beas",   "district": "Kullu",   "trigger": "cloudburst"},
    {"date": "2024-08-01", "basin": "Satluj", "district": "Mandi",   "trigger": "monsoon"},
]

HP_REGION = [HP_BBOX[0], HP_BBOX[2], HP_BBOX[1], HP_BBOX[3]]  # [west, south, east, north]
DRIVE_FOLDER = "hp_flood_inventory"


def authenticate() -> None:
    """Initialise GEE with credentials from earthengine authenticate."""
    import ee
    try:
        ee.Initialize(project="ee-paras")  # replace with your GEE project ID
    except Exception:
        ee.Initialize()
    print("  GEE authenticated and initialised.")


def build_flood_image(date_str: str, pre_days: int = 20, post_days: int = 5):
    """
    Build a binary flood mask image for a single event using Sentinel-1 VV change detection.
    Returns an ee.Image (0/1 byte, EPSG:32643, 30m).
    """
    import ee

    date     = ee.Date(date_str)
    pre_img  = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee.Geometry.Rectangle(HP_REGION))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select("VV")
        .filterDate(date.advance(-(pre_days + 10), "day"), date.advance(-2, "day"))
        .median()
    )
    post_img = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee.Geometry.Rectangle(HP_REGION))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select("VV")
        .filterDate(date.advance(1, "day"), date.advance(post_days, "day"))
        .median()
    )

    diff        = post_img.subtract(pre_img)
    flood_raw   = diff.lt(-3)  # -3 dB threshold
    flood_clean = flood_raw.focal_mode(3, "square", "pixels")

    # Remove permanent water
    perm_water  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(90)
    flood_only  = flood_clean.And(perm_water.Not())

    # Remove small patches
    connected   = flood_only.connectedPixelCount(100, False)
    large       = flood_only.And(connected.gte(6))

    return large.toByte().set("event_date", date_str)


def submit_export_task(image, description: str, folder: str) -> object:
    """Submit a GEE export-to-Drive task and return the Task object."""
    import ee

    task = ee.batch.Export.image.toDrive(
        image       = image,
        description = description,
        folder      = folder,
        region      = ee.Geometry.Rectangle(HP_REGION),
        scale       = 30,
        crs         = "EPSG:32643",
        maxPixels   = int(1e10),
        fileFormat  = "GeoTIFF",
    )
    task.start()
    print(f"  Submitted: {description}")
    return task


def submit_rainfall_exports() -> list:
    """Submit GPM mean annual and p95 extreme rainfall export tasks."""
    import ee

    hp_geom = ee.Geometry.Rectangle(HP_REGION)

    # Mean annual rainfall (GPM IMERG monthly)
    rainfall_mean = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
        .filterDate("2001-01-01", "2023-12-31")
        .select("precipitation")
        .mean()
        .multiply(24 * 30.44 * 12)  # mm/hr → mm/year
    )
    task_mean = ee.batch.Export.image.toDrive(
        image       = rainfall_mean,
        description = "rainfall_mean_annual_gpm",
        folder      = DRIVE_FOLDER,
        region      = hp_geom,
        scale       = 1000,
        crs         = "EPSG:32643",
        maxPixels   = int(1e9),
    )
    task_mean.start()
    print("  Submitted: rainfall_mean_annual_gpm")

    # 95th percentile of daily rainfall (extreme events)
    rainfall_p95 = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
        .filterDate("2001-01-01", "2023-12-31")
        .select("precipitation")
        .reduce(ee.Reducer.percentile([95]))
    )
    task_p95 = ee.batch.Export.image.toDrive(
        image       = rainfall_p95,
        description = "rainfall_extreme_p95_gpm",
        folder      = DRIVE_FOLDER,
        region      = hp_geom,
        scale       = 1000,
        crs         = "EPSG:32643",
        maxPixels   = int(1e9),
    )
    task_p95.start()
    print("  Submitted: rainfall_extreme_p95_gpm")

    return [task_mean, task_p95]


def poll_tasks(tasks: list, poll_interval: int = 30) -> None:
    """Poll GEE task status until all complete or fail."""
    import ee

    print(f"\n  Polling {len(tasks)} task(s) every {poll_interval}s "
          f"(this may take 10–60 minutes)...")
    pending = list(tasks)
    while pending:
        still_pending = []
        for task in pending:
            status = task.status()
            state  = status["state"]
            desc   = status.get("description", "?")
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                icon = "✓" if state == "COMPLETED" else "✗"
                print(f"    {icon} {desc}: {state}")
            else:
                still_pending.append(task)
        pending = still_pending
        if pending:
            print(f"    Still running: {len(pending)} task(s)... (sleeping {poll_interval}s)")
            time.sleep(poll_interval)
    print("  All tasks finished.")


def download_from_drive(out_dir: Path) -> None:
    """
    Download completed GEE exports from Google Drive to out_dir.
    Requires: google-auth google-auth-httplib2 google-api-python-client
    Install with: uv add google-auth google-auth-httplib2 google-api-python-client
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import io
    except ImportError:
        print("\n  Google Drive download skipped — install google-api-python-client:")
        print("    uv add google-auth google-auth-httplib2 google-api-python-client")
        print(f"  Manually download from Drive folder '{DRIVE_FOLDER}' → {out_dir}")
        return

    # Use earthengine credentials (already authenticated)
    import ee
    creds = ee.ServiceAccountCredentials(None, None)  # uses default credentials
    service = build("drive", "v3", credentials=creds)

    # List files in the folder
    query = f"name contains 'flood_' or name contains 'rainfall_'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files   = results.get("files", [])

    if not files:
        print(f"  No files found in Drive. Check folder '{DRIVE_FOLDER}' manually.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        fid, fname = f["id"], f["name"]
        dest = out_dir / fname
        if dest.exists():
            print(f"  Already exists: {fname}")
            continue
        request = service.files().get_media(fileId=fid)
        buf = io.FileIO(dest, "wb")
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        print(f"  Downloaded: {fname}")


def save_event_metadata() -> None:
    """Save known flood events as JSON for reference."""
    out  = FLOOD_DIR / "known_events.json"
    data = {
        "source":           "HiFlo-DAT + NDMA + Kumar 2022 + NHESS 2026",
        "total_events":     len(KNOWN_FLOOD_EVENTS),
        "training_events":  [e for e in KNOWN_FLOOD_EVENTS if "2023" not in e["date"]],
        "test_events":      [e for e in KNOWN_FLOOD_EVENTS if "2023" in e["date"]],
        "note":             "2023 events held out as temporal validation set",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"  Event metadata → {out}")


def main() -> None:
    print("=" * 60)
    print("Phase 3: SAR Flood Inventory via GEE Python API")
    print("=" * 60)

    save_event_metadata()

    # Try GEE Python API
    try:
        import ee
    except ImportError:
        print("  earthengine-api not installed. Run: uv sync")
        return

    print("\n  Authenticating with GEE...")
    try:
        authenticate()
    except Exception as exc:
        print(f"  GEE authentication failed: {exc}")
        print("  Run once: uv run earthengine authenticate")
        print("  Then re-run this script.")
        return

    # Submit flood export tasks
    print("\n  Submitting SAR flood export tasks...")
    flood_tasks = []
    for event in KNOWN_FLOOD_EVENTS:
        d    = event["date"]
        split = "test" if "2023" in d else "train"
        tag  = d.replace("-", "")
        img  = build_flood_image(d)
        task = submit_export_task(img, f"flood_{split}_{tag}", DRIVE_FOLDER)
        flood_tasks.append(task)

    # Submit rainfall export tasks
    print("\n  Submitting rainfall export tasks...")
    rain_tasks = submit_rainfall_exports()

    all_tasks = flood_tasks + rain_tasks
    print(f"\n  Total tasks submitted: {len(all_tasks)}")
    print(f"  Results will appear in Google Drive → {DRIVE_FOLDER}/")

    # Poll until done
    poll_tasks(all_tasks, poll_interval=30)

    # Download results
    sar_dir    = FLOOD_DIR / "sar"
    rain_dir   = Path(__file__).parent.parent / "data" / "raw" / "rainfall"
    print(f"\n  Downloading SAR TIFs → {sar_dir}")
    download_from_drive(sar_dir)
    print(f"\n  Downloading rainfall TIFs → {rain_dir}")
    download_from_drive(rain_dir)

    print("\nDone. Next: run 04_preprocess_terrain.py")


if __name__ == "__main__":
    main()
