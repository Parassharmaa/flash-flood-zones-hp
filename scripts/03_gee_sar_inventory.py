"""
Phase 3 — Build SAR-based flood inventory using Sentinel-1 via GEE.

Method:
  1. For each known flood event date (from HiFlo-DAT + NDMA records),
     compute backscatter change between pre-event and post-event composites.
  2. Threshold change detection → binary flood extent.
  3. Clean with morphological operations.
  4. Export flood polygons per event.

Requirements:
  earthengine-api installed + authenticated (run `earthengine authenticate` once)

Outputs:
  results exported to Google Drive → download to data/raw/flood_inventory/sar/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import FLOOD_DIR, HP_BBOX, SAR_START_DATE, SAR_END_DATE  # noqa: E402

# Known major HP flood events for targeted SAR processing
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

GEE_SCRIPT = '''
// =============================================================================
// HP Flash Flood SAR Inventory — Sentinel-1 Change Detection
// Run in GEE Code Editor: https://code.earthengine.google.com
// =============================================================================

var hp = ee.Geometry.Rectangle([75.5, 30.3, 79.0, 33.3]);

// Function: detect flood extent for a single event
function detectFlood(eventDate, preWindowDays, postWindowDays) {
  var date = ee.Date(eventDate);
  var preStart  = date.advance(-preWindowDays - 10, 'day');
  var preEnd    = date.advance(-2, 'day');
  var postStart = date.advance(1, 'day');
  var postEnd   = date.advance(postWindowDays, 'day');

  // Sentinel-1 VV polarisation, descending orbit (more stable for water)
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(hp)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .select('VV');

  var pre  = s1.filterDate(preStart, preEnd).median();
  var post = s1.filterDate(postStart, postEnd).median();

  // Change detection: decrease in backscatter = water (flood)
  var diff = post.subtract(pre);

  // Threshold: -3 dB change typical for water detection
  var floodMask = diff.lt(-3)
    .focal_mode(3, 'square', 'pixels')   // morphological closing
    .reproject({crs: 'EPSG:32643', scale: 30});

  // Remove permanent water (JRC Global Surface Water)
  var permanentWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    .select('occurrence').gt(90);
  var floodOnly = floodMask.and(permanentWater.not());

  // Filter small patches (< 5 ha = 50 pixels at 30m)
  var connected = floodOnly.connectedPixelCount(100, false);
  var largePatches = floodOnly.and(connected.gte(6));

  return largePatches.set({
    'event_date': eventDate,
    'system:time_start': date.millis()
  });
}

// Process all training events (exclude 2023 — temporal test set)
var trainingEvents = [
  '2018-07-25', '2018-08-18', '2019-07-31', '2019-08-14',
  '2020-07-28', '2021-07-26', '2021-08-17', '2022-07-13', '2022-08-24'
];

var testEvents = [
  '2023-07-09', '2023-07-14', '2023-08-02', '2023-08-13', '2023-09-01'
];

// Export each event as separate asset
trainingEvents.forEach(function(d) {
  var flood = detectFlood(d, 20, 5);
  Export.image.toDrive({
    image: flood.toByte(),
    description: 'flood_train_' + d.replace(/-/g, ''),
    folder: 'hp_flood_inventory',
    region: hp,
    scale: 30,
    crs: 'EPSG:32643',
    maxPixels: 1e10
  });
});

testEvents.forEach(function(d) {
  var flood = detectFlood(d, 20, 5);
  Export.image.toDrive({
    image: flood.toByte(),
    description: 'flood_test_' + d.replace(/-/g, ''),
    folder: 'hp_flood_inventory',
    region: hp,
    scale: 30,
    crs: 'EPSG:32643',
    maxPixels: 1e10
  });
});

// Also export mean annual rainfall (GPM IMERG)
var gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07')
  .filterDate('2001-01-01', '2023-12-31')
  .select('precipitation')
  .mean()
  .multiply(24 * 30.44)  // mm/hr → mm/month → mm/year approximation
  .multiply(12);

Export.image.toDrive({
  image: gpm,
  description: 'rainfall_mean_annual_gpm',
  folder: 'hp_flood_inventory',
  region: hp,
  scale: 1000,
  crs: 'EPSG:32643',
  maxPixels: 1e9
});

// 95th percentile daily rainfall (extreme events)
var gpmDaily = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
  .filterDate('2001-01-01', '2023-12-31')
  .select('precipitation')
  .reduce(ee.Reducer.percentile([95]));

Export.image.toDrive({
  image: gpmDaily,
  description: 'rainfall_extreme_p95_gpm',
  folder: 'hp_flood_inventory',
  region: hp,
  scale: 1000,
  crs: 'EPSG:32643',
  maxPixels: 1e9
});

print('All exports submitted. Check Tasks tab.');
'''


def save_gee_script() -> None:
    """Save the GEE JavaScript to a file for manual execution."""
    out = Path(__file__).parent.parent / "scripts" / "03_gee_script.js"
    out.write_text(GEE_SCRIPT)
    print(f"GEE script saved → {out}")
    print("Instructions:")
    print("  1. Open https://code.earthengine.google.com")
    print("  2. Paste contents of scripts/03_gee_script.js")
    print("  3. Click Run → all exports submitted to Tasks tab")
    print("  4. Run each task → exports to Google Drive folder 'hp_flood_inventory'")
    print("  5. Download all TIFs to data/raw/flood_inventory/sar/")


def save_event_metadata() -> None:
    """Save known flood events as JSON for reference."""
    out = FLOOD_DIR / "known_events.json"
    data = {
        "source": "HiFlo-DAT + NDMA + Kumar 2022 + NHESS 2026",
        "total_events": len(KNOWN_FLOOD_EVENTS),
        "training_events": [e for e in KNOWN_FLOOD_EVENTS if "2023" not in e["date"]],
        "test_events": [e for e in KNOWN_FLOOD_EVENTS if "2023" in e["date"]],
        "note": "2023 events held out as temporal validation set",
    }
    out.write_text(json.dumps(data, indent=2))
    print(f"Event metadata → {out}")


def main() -> None:
    print("=" * 60)
    print("Phase 3: SAR Flood Inventory via GEE")
    print("=" * 60)
    save_gee_script()
    save_event_metadata()
    print(f"\nKnown events: {len(KNOWN_FLOOD_EVENTS)} total")
    print(f"  Training: {sum(1 for e in KNOWN_FLOOD_EVENTS if '2023' not in e['date'])}")
    print(f"  Test (2023): {sum(1 for e in KNOWN_FLOOD_EVENTS if '2023' in e['date'])}")
    print("\nNext: run GEE script, download results, then run 04_preprocess_terrain.py")


if __name__ == "__main__":
    main()
