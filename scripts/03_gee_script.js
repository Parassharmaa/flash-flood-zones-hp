// HP Flash Flood SAR Inventory - Sentinel-1 Flood Detection
// GEE Code Editor: https://code.earthengine.google.com
// All ASCII only

var hp = ee.Geometry.Rectangle([75.5, 30.3, 79.0, 33.3]);

// Build Sentinel-1 IW VV collection for a date range
function getS1(startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(hp)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV']);
}

// Long-term dry season reference (Nov-Feb 2016-2022, median)
var dryRef = getS1('2016-11-01', '2022-02-28').median();

// Permanent water mask
var permWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence').gt(50).unmask(0);

// Seasonal flood map: monsoon minimum vs dry reference
function seasonalFlood(year) {
  var start    = year + '-06-01';
  var end      = year + '-10-15';
  var monsoon  = getS1(start, end);
  var minBack  = monsoon.min();
  var diff     = minBack.subtract(dryRef);
  var flooded  = diff.lt(-3).unmask(0);
  var noPermW  = flooded.where(permWater, 0);
  var connected = noPermW.selfMask().connectedPixelCount(50, false);
  var cleaned  = noPermW.where(connected.lt(6), 0);
  return cleaned.toByte().rename('flood').set('year', year);
}

// --- Training exports (2018-2022) ---
Export.image.toDrive({
  image: seasonalFlood(2018),
  description: 'flood_train_2018',
  folder: 'hp_flood_inventory',
  region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
});

Export.image.toDrive({
  image: seasonalFlood(2019),
  description: 'flood_train_2019',
  folder: 'hp_flood_inventory',
  region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
});

Export.image.toDrive({
  image: seasonalFlood(2020),
  description: 'flood_train_2020',
  folder: 'hp_flood_inventory',
  region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
});

Export.image.toDrive({
  image: seasonalFlood(2021),
  description: 'flood_train_2021',
  folder: 'hp_flood_inventory',
  region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
});

Export.image.toDrive({
  image: seasonalFlood(2022),
  description: 'flood_train_2022',
  folder: 'hp_flood_inventory',
  region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
});

// --- Test export (2023 - held out) ---
Export.image.toDrive({
  image: seasonalFlood(2023),
  description: 'flood_test_2023',
  folder: 'hp_flood_inventory',
  region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
});

// --- Rainfall: mean annual (GPM IMERG Monthly, mm/year) ---
var gpmCollection = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07')
  .filterDate('2001-01-01', '2023-12-31')
  .select('precipitation');

var gpmMean = gpmCollection.mean().multiply(8766).rename('rainfall_mm_yr');

Export.image.toDrive({
  image: gpmMean,
  description: 'rainfall_mean_annual_gpm',
  folder: 'hp_flood_inventory',
  region: hp, scale: 5000, crs: 'EPSG:4326', maxPixels: 1e9
});

// --- Rainfall: 95th percentile monthly (extreme months proxy) ---
var gpmP95 = gpmCollection.max().rename("rainfall_max_monthly");

Export.image.toDrive({
  image: gpmP95,
  description: 'rainfall_extreme_p95_gpm',
  folder: 'hp_flood_inventory',
  region: hp, scale: 5000, crs: 'EPSG:4326', maxPixels: 1e9
});

print('8 export tasks queued. Open Tasks tab and click RUN on each.');
