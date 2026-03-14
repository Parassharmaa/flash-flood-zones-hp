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

### Completed
- Full pipeline runs on synthetic data (scripts 07-12 all pass)
- Paper PDF compiles cleanly (2.12 MB, tectonic)
- DEM downloading from Copernicus GLO-30 AWS (10/16 tiles as of now)
- LULC tiles complete (4 tiles, ESA WorldCover)
- GEE tasks running (user submitted, 2/8 failed due to HH polarisation)

### GEE Issues & Fixes
- **Error**: `Image.select: Band pattern 'VV' did not match any bands. Available bands: [HH, HV, angle]`
  - **Cause**: Some Sentinel-1B scenes over HP use HH/HV instead of VV/VH
  - **Fix**: Added `.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))` to `getS1()`
  - **Action**: User must cancel failed tasks and re-run with updated script

### Key Code Fixes This Session
- Fixed DEM download URL: OpenTopography → Copernicus AWS S3
- Fixed sys.path in all scripts (parent not parent.parent)
- Fixed date parsing in flood inventory (year only, not YYYYMMDD)
- Added scipy morphological opening to SAR flood mask processing
- Added `distance_to_river` terrain factor (was in docs but not implemented)
- Fixed `rainfall_extreme` filename (max_monthly not p95)

## Next Steps (Priority Order)
1. Wait for DEM download to complete (remaining 6 tiles)
2. User: cancel failed GEE tasks, re-run with updated script (VV filter)
3. User: download GEE outputs from Google Drive → data/raw/flood_inventory/sar/ and data/raw/rainfall/
4. Run `uv run python scripts/04_preprocess_terrain.py` (needs all 16 DEM tiles)
5. Run `uv run python scripts/05_watershed_delineation.py`
6. Run `uv run python scripts/06_assemble_factors.py`
7. Run `uv run python scripts/07_build_flood_inventory.py` (will use real SAR TIFs)
8. Run `uv run python scripts/08_train_baseline_models.py`
9. Run `uv run python scripts/09_train_gnn.py`
10. Run `uv run python scripts/10_conformal_prediction.py`
11. Run `uv run python scripts/11_shap_analysis.py`
12. Update paper XX placeholders with real results
