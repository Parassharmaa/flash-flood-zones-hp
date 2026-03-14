// HP Flash Flood SAR Inventory - Sentinel-1 Flood Detection
// GEE Code Editor: https://code.earthengine.google.com
// No CRS string in exports - uses each image's native projection

var hp = ee.Geometry.Rectangle([75.5, 30.3, 79.0, 33.3]);

function getS1(startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(hp)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select(['VV']);
}

// Long-term dry-season median (Nov-Feb, 2016-2022) - used as flood reference
var dryRef = getS1('2016-11-01', '2022-02-28').median();

// Permanent water mask (JRC, >50% occurrence)
var permWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence').gt(50).unmask(0);

// Per-year seasonal flood mask
// monsoon minimum backscatter vs dry-season reference
function seasonalFlood(year) {
  var start   = year + '-06-01';
  var end     = year + '-10-15';
  var monsoon = getS1(start, end);
  var minBack = monsoon.min();
  var diff    = minBack.subtract(dryRef);
  var flooded = diff.lt(-3).unmask(0);
  var result  = flooded.where(permWater, 0);
  return result.toByte().rename('flood').set('year', year);
}

// Helper: export one flood image (no crs arg - uses native S1 projection)
function exportFlood(img, desc) {
  Export.image.toDrive({
    image: img,
    description: desc,
    folder: 'hp_flood_inventory',
    region: hp,
    scale: 30,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
}

// Training years (2018-2022)
exportFlood(seasonalFlood(2018), 'flood_train_2018');
exportFlood(seasonalFlood(2019), 'flood_train_2019');
exportFlood(seasonalFlood(2020), 'flood_train_2020');
exportFlood(seasonalFlood(2021), 'flood_train_2021');
exportFlood(seasonalFlood(2022), 'flood_train_2022');

// Test year (2023 - held out for temporal validation)
exportFlood(seasonalFlood(2023), 'flood_test_2023');

// GPM IMERG Monthly mean annual rainfall (mm/year)
var gpmColl = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07')
  .filterDate('2001-01-01', '2023-12-31')
  .select('precipitation');

Export.image.toDrive({
  image: gpmColl.mean().multiply(8766).rename('rainfall_mm_yr'),
  description: 'rainfall_mean_annual_gpm',
  folder: 'hp_flood_inventory',
  region: hp,
  scale: 11000,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

// GPM max monthly rainfall (extreme proxy, mm/hr)
Export.image.toDrive({
  image: gpmColl.max().rename('rainfall_max_monthly'),
  description: 'rainfall_max_monthly_gpm',
  folder: 'hp_flood_inventory',
  region: hp,
  scale: 11000,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

print('8 tasks submitted. Click RUN on each in the Tasks tab.');
