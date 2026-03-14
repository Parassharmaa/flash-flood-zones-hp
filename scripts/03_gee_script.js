// HP Flash Flood SAR Inventory - Sentinel-1 Change Detection
// Run in GEE Code Editor: https://code.earthengine.google.com
// NOTE: All ASCII only - no special chars to avoid GEE parser issues

var hp = ee.Geometry.Rectangle([75.5, 30.3, 79.0, 33.3]);

// Detect flood extent for a single event date string (e.g. '2018-07-25')
function detectFlood(eventDate) {
  var t = ee.Date(eventDate);

  var preStart  = t.advance(-30, 'day');
  var preEnd    = t.advance(-2,  'day');
  var postStart = t.advance(1,   'day');
  var postEnd   = t.advance(6,   'day');

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(hp)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .select('VV');

  var pre  = s1.filterDate(preStart, preEnd).median();
  var post = s1.filterDate(postStart, postEnd).median();

  var diff       = post.subtract(pre);
  var floodRaw   = diff.lt(-3);
  var floodClean = floodRaw.focal_mode({radius: 1, kernelType: 'square', units: 'pixels'});
  var permWater  = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(90);
  var floodOnly  = floodClean.and(permWater.not());
  var connected  = floodOnly.connectedPixelCount(100, false);
  var largePatch = floodOnly.and(connected.gte(6));

  return largePatch.toByte().set('event_date', eventDate);
}

var trainEvents = [
  '2018-07-25','2018-08-18','2019-07-31','2019-08-14',
  '2020-07-28','2021-07-26','2021-08-17','2022-07-13','2022-08-24'
];

var testEvents = [
  '2023-07-09','2023-07-14','2023-08-02','2023-08-13','2023-09-01'
];

for (var i = 0; i < trainEvents.length; i++) {
  var d = trainEvents[i];
  Export.image.toDrive({
    image: detectFlood(d),
    description: 'flood_train_' + d.replace(/-/g,''),
    folder: 'hp_flood_inventory',
    region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
  });
}

for (var j = 0; j < testEvents.length; j++) {
  var dt = testEvents[j];
  Export.image.toDrive({
    image: detectFlood(dt),
    description: 'flood_test_' + dt.replace(/-/g,''),
    folder: 'hp_flood_inventory',
    region: hp, scale: 30, crs: 'EPSG:32643', maxPixels: 1e10
  });
}

var rainfallMean = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V07')
  .filterDate('2001-01-01','2023-12-31')
  .select('precipitation').mean().multiply(8766);

Export.image.toDrive({
  image: rainfallMean, description: 'rainfall_mean_annual_gpm',
  folder: 'hp_flood_inventory', region: hp, scale: 1000, crs: 'EPSG:32643', maxPixels: 1e9
});

var rainfallP95 = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
  .filterDate('2001-01-01','2023-12-31')
  .select('precipitation').reduce(ee.Reducer.percentile([95]));

Export.image.toDrive({
  image: rainfallP95, description: 'rainfall_extreme_p95_gpm',
  folder: 'hp_flood_inventory', region: hp, scale: 1000, crs: 'EPSG:32643', maxPixels: 1e9
});

print('Tasks submitted - click RUN on each in the Tasks tab.');
