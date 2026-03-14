# Flash Flood Zone Mapping — Implementation Plan

## Objective
Map flash flood risk zones across Himachal Pradesh using terrain, hydrology,
land cover, and climate data. Identify high-risk areas and analyse drivers.

## Progress Tracker
- [ ] Phase 1: Data collection and preprocessing
- [ ] Phase 2: Terrain and hydrological analysis (DEM derivatives)
- [ ] Phase 3: Flash flood conditioning factors
- [ ] Phase 4: Flood susceptibility modelling
- [ ] Phase 5: Validation and uncertainty analysis
- [ ] Phase 6: Results and report

## Phases (to be detailed as work progresses)

### Phase 1 — Data
- DEM (SRTM/ALOS) at target resolution
- Rainfall data (IMD gridded / CHIRPS)
- Land cover (ESA CCI / MODIS)
- Soil type / lithology
- Historical flood event records (NDMA, HPSDMA, news archives)
- HP district/watershed boundaries

### Phase 2 — Terrain Analysis
- Slope, aspect, curvature from DEM
- Flow accumulation, drainage density
- Topographic Wetness Index (TWI)
- Stream Power Index (SPI)

### Phase 3 — Conditioning Factors
- NDVI / vegetation cover
- Distance to streams
- Rainfall intensity (return period analysis)
- Land use change

### Phase 4 — Susceptibility Modelling
- Frequency Ratio / Weights of Evidence (statistical baseline)
- Machine learning: Random Forest, XGBoost
- Ensemble approach
- SHAP values for interpretability

### Phase 5 — Validation
- ROC-AUC on held-out flood events
- Spatial cross-validation
- Comparison with published flood inventories

### Phase 6 — Output
- Flash flood susceptibility map (HP-wide, district-level)
- High-risk zone delineation
- Conservation/policy recommendations
