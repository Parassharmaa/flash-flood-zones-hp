# Flash Flood HP — Learnings & Decisions Log

## Phase 0 — Foundation (Complete)

### Project goals confirmed
- Version B: useful output first (decision tool for HP SDMA/PWD/tourism), paper second
- Target journal: NHESS or Remote Sensing (Q1)
- Completion order: paper → dashboard (dashboard visualises validated paper outputs)

### Novel approach confirmed
1. **SAR-based flood inventory** (Sentinel-1 GEE, 2018–2024) — replicable, cloud-free
2. **Graph Neural Network (GraphSAGE)** on watershed connectivity graph — no prior flash flood susceptibility paper uses GNNs; captures upstream→downstream flood propagation
3. **Conformal Prediction (MAPIE)** — first uncertainty-quantified susceptibility map for HP
4. **Spatial block CV (leave-one-basin-out)** — addresses #1 methodological flaw in existing literature
5. **SHAP spatial analysis** — "factor importance map" per district

### Literature review key findings
- Only ONE peer-reviewed ML susceptibility paper for HP: Saha et al. 2023 (Beas basin, AUC=0.88, book chapter)
- No journal paper covers multiple HP basins
- No study applies spatial block CV for HP
- No study uses GLOF-specific conditioning factors (glacial lake proximity)
- 2023 monsoon season (404 deaths, Rs 9,905 crore) = ideal temporal test set

---

## Phase 1 — Data Collection (Scripts ready, data pending)

### Data acquisition status
| Dataset | Script | Status | Notes |
|---------|--------|--------|-------|
| HP boundaries | 01_download_boundaries.py | Script ready | OSM Overpass |
| DEM (GLO-30) | 02_download_rasters.py | Script ready | Copernicus tiles |
| LULC (ESA WorldCover) | 02_download_rasters.py | Script ready | AWS S3 |
| Soil (SoilGrids) | 02_download_rasters.py | Script ready | WCS API |
| SAR flood inventory | 03_gee_sar_inventory.py | **Manual step needed** | GEE Code Editor |
| Rainfall (GPM) | 03_gee_sar_inventory.py | **Manual step needed** | GEE export |

### Key decisions
- **CRS**: UTM Zone 43N (EPSG:32643) — appropriate for HP, minimises distortion
- **Resolution**: 30m target — balances detail with computational feasibility
- **Study area bbox**: 75.5–79.0°E, 30.3–33.3°N (full HP + small buffer)
- **Temporal split**: train = 2018–2022 events, test = 2023 July–Sept season

---

## Phase 2 — Preprocessing (Scripts ready)

### Terrain factors pipeline
DEM → slope, aspect, plan/profile curvature, TWI, SPI, TRI → factor stack
Uses pysheds for flow accumulation (D8 algorithm); numpy fallback if unavailable.

### Watershed delineation
- Target: 300–500 sub-watersheds for HP (MIN_CATCHMENT_AREA_KM2 = 150)
- GNN graph: directed edges upstream → downstream based on mean elevation difference
- Adjacency matrix saved as scipy sparse for efficiency

### Multicollinearity
- Pearson threshold: |r| > 0.80 (stricter than typical 0.75)
- VIF threshold: 10
- Expected removals: BIO-type correlations; TWI and drainage density likely correlated

---

## Phase 3 — Flood Inventory (SAR pipeline ready)

### 16 known events identified
- 10 training events (2018–2022)
- 5 test events (2023 — temporal validation)
- Plus 2024 training events
- Sources: HiFlo-DAT (Kullu), NDMA, Kumar 2022, NHESS 2026 analysis

### SAR processing approach
- Sentinel-1 VV polarisation, descending orbit (more stable over water)
- Change detection: pre/post backscatter difference, threshold -3 dB
- Morphological cleaning: focal mode filter 3×3
- Permanent water removal: JRC Global Surface Water mask
- Minimum patch size: 6 connected pixels (0.54 ha at 30m)

### Non-flood sampling
- Buffer: 1,000m around all flood points
- Ratio: 5:1 (non-flood:flood) — avoids excessive imbalance
- Spatial distribution: random within HP extent outside exclusion zone

---

## Phase 4 — ML Pipeline (Scripts ready)

### Baseline models
- RF: n=500, max_depth=6, balanced class weights
- XGBoost: n=500, max_depth=6, scale_pos_weight=5
- LightGBM: n=500, max_depth=6, balanced class weights
- Stacking: RF+XGB+LGB → Logistic Regression meta-learner

### GNN architecture decision
- Chose GraphSAGE over GAT for initial implementation
  - Reason: GraphSAGE more stable on irregular graphs; GAT adds attention overhead
  - Plan: compare both in ablation study for paper
- PyTorch Geometric implementation with fallback neighbourhood aggregation proxy
- 3 layers, 64 hidden dim, 200 epochs, LR=1e-3, dropout=0.3
- Bidirectional message passing (reverse edges added for stability)

### Conformal prediction
- MAPIE split-conformal (inductive conformal prediction)
- α = 0.10 → 90% prediction intervals
- Calibration set: 20% of training data held out after model fitting
- Manual fallback implemented if MAPIE not installed

### Key methodological insight (from literature)
Random train-test splits inflate AUC by 5–15% due to spatial autocorrelation.
All models evaluated with leave-one-basin-out spatial block CV.
Benchmark: Saha et al. 2023 (AUC=0.88) — target to beat.

---

## Pipeline Status (2026-03-14)

### Completed (Session 2)
- All 16 DEM tiles downloaded and merged to dem_hp.tif (15020×12865px, UTM 43N)
- All terrain factors computed: slope, aspect, plan/profile curvature, TWI, SPI, TRI, distance_to_river
- Watershed graph: 460 nodes, 1700 directed edges (20km regular grid proxy — pysheds fallback)
- LULC: 4 ESA WorldCover 2021 tiles merged to lulc_hp.tif
- SoilGrids clay: clay_0_30cm_hp.tif downloaded (WCS 1.0.0, clay_0-5cm_mean)
- Factor stack: 11 bands, BigTIFF, 15020×12865 px (real terrain + LULC + soil; rainfall pending GEE)
- Conformal prediction: coverage 93.1% @ α=0.10 (valid); susceptibility maps written (773MB each)
- SHAP analysis: global importance, dependence plots, spatial factor map (downsampled 1/30)
- Paper PDF compiles cleanly (1.98 MB)
- Full pipeline scripts 04–12 all complete without errors

### Performance Fixes (Session 2)
- `10_conformal_prediction.py`: downsample TIFs 1/10 before matplotlib imshow (was hanging)
- `11_shap_analysis.py`: compute spatial SHAP map at 1/30 scale (1.9M → 21k pixels)
- `generate_susceptibility_map`: processes in 256-row blocks for memory efficiency (BigTIFF)

### GEE Issues & Fixes
- **Error**: `Image.select: Band pattern 'VV' did not match any bands. Available bands: [HH, HV, angle]`
  - **Cause**: Some Sentinel-1B scenes over HP use HH/HV instead of VV/VH
  - **Fix**: Added `.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))` to `getS1()`
  - **Action**: User must cancel failed tasks and re-run with updated script

### Key Code Fixes (Session 1)
- Fixed DEM download URL: OpenTopography → Copernicus AWS S3
- Fixed sys.path in all scripts (parent not parent.parent)
- Fixed date parsing in flood inventory (year only, not YYYYMMDD)
- Added scipy morphological opening to SAR flood mask processing
- Added `distance_to_river` terrain factor (was in docs but not implemented)
- Fixed `rainfall_extreme` filename (max_monthly not p95)

## Next Steps (Priority Order — BLOCKING on GEE data)
1. **User: cancel all failed GEE tasks, re-run** `scripts/03_gee_script.js` (with VV filter) in GEE Code Editor
2. **User: download GEE outputs from Google Drive** → `data/raw/flood_inventory/sar/` (flood_*.tif) + `data/raw/rainfall/` (rainfall_*.tif)
3. Re-run `uv run python scripts/06_assemble_factors.py` (will pick up rainfall_mean + rainfall_extreme)
4. Re-run `uv run python scripts/07_build_flood_inventory.py` (will use real SAR TIFs instead of synthetic)
5. Re-run scripts 08–12 with real flood inventory data
6. Update paper `XX` placeholders in `paper/chapters/04_results.tex` with real AUC/F1/Kappa values
7. Abstract needs real numbers too
