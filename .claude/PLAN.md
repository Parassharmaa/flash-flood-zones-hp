# Flash Flood Risk Intelligence — HP
## Full Implementation Plan

**Goal:** Version B — Useful output first, paper second.
"A decision tool that tells HP SDMA / tourism board / PWD which roads, villages,
and valleys face highest risk this monsoon season and why."

**Novel approach:** GNN (Graph Neural Network on watershed graph) + Conformal Prediction
(uncertainty-quantified susceptibility) + SAR-derived flood inventory

**Target paper journal:** NHESS or Remote Sensing (Q1)

---

## Progress Tracker

### Phase 0 — Foundation ✅ IN PROGRESS
- [x] Literature review (33 papers, synthesis, gaps)
- [x] Monorepo structure planned
- [ ] Turborepo + Next.js dashboard scaffold
- [ ] Python ML environment setup
- [ ] GitHub repo (private)

### Phase 1 — Data Collection
- [ ] DEM: ALOS PALSAR 12.5m (JAXA) or Copernicus GLO-30 for HP extent
- [ ] Flood inventory: Sentinel-1 SAR via Google Earth Engine (2018–2024)
- [ ] Flood inventory: HiFlo-DAT (Kullu, 128 events) — import + clean
- [ ] Rainfall: GPM-IMERG monthly + extreme event data
- [ ] LULC: ESA WorldCover 2021 (10m)
- [ ] Soil: SoilGrids 250m
- [ ] Lithology: GSI geological map
- [ ] Glacial lakes: ICIMOD inventory
- [ ] Infrastructure: OpenStreetMap (roads, bridges, settlements)
- [ ] HP boundaries: district + watershed polygons (HUC-equivalent)
- [ ] Tourism facilities: OSM + HP tourism data

### Phase 2 — Preprocessing & Conditioning Factors
- [ ] Clip all rasters to HP extent + reproject to UTM 43N
- [ ] Derive terrain factors from DEM: slope, aspect, curvature, TWI, SPI, TRI, drainage density
- [ ] Compute distance-to-river (Euclidean + cost distance)
- [ ] Multicollinearity check: Pearson correlation + VIF (remove >0.8 / VIF >10)
- [ ] Delineate sub-watersheds (catchment polygons) for GNN graph construction
- [ ] Build watershed adjacency graph (upstream→downstream edges)
- [ ] Final factor set: target 12–16 conditioning factors

### Phase 3 — Flood Inventory Construction
- [ ] SAR-based: process Sentinel-1 GRD scenes (2018–2024) in GEE
  - Change detection: pre/post event backscatter difference
  - Threshold + morphological cleaning
  - Export flood extent polygons per event
- [ ] Combine SAR inventory + HiFlo-DAT + NDMA records
- [ ] Clean: remove duplicates, assign event metadata (date, basin, trigger type)
- [ ] Split: training (pre-2023) / temporal test (July–September 2023)
- [ ] Non-flood points: random sampling with ≥5x buffer from flood points

### Phase 4 — ML Pipeline (Core Science)

#### 4a — Baseline Models (benchmark)
- [ ] Random Forest (scikit-learn)
- [ ] XGBoost
- [ ] LightGBM
- [ ] Stacking ensemble (RF + XGB + LGB → Logistic Regression meta-learner)
- [ ] Validation: spatial block CV (leave-one-watershed-out) + AUC, F1, Kappa

#### 4b — Novel Model: Graph Neural Network (PRIMARY CONTRIBUTION)
- [ ] Build watershed graph using PyTorch Geometric
  - Nodes: sub-watersheds (~200–500 catchment polygons)
  - Node features: aggregated conditioning factors per catchment
  - Edges: directed river connectivity (upstream → downstream)
  - Edge weights: relative flow volume / catchment area ratio
- [ ] GNN architecture: GraphSAGE or GAT (Graph Attention Network)
  - GraphSAGE: aggregates neighbour features (good for irregular graphs)
  - GAT: learns attention weights on edges (learns which upstream nodes matter more)
- [ ] Train: flood occurrence per catchment (binary)
- [ ] Compare: GNN vs. pixel-based RF/XGBoost — does connectivity improve accuracy?
- [ ] SHAP for GNN: node-level feature attribution

#### 4c — Uncertainty Quantification: Conformal Prediction (SECOND CONTRIBUTION)
- [ ] Wrap best-performing model with MAPIE (conformal prediction library)
- [ ] Output: susceptibility score + 90% prediction interval per pixel/catchment
- [ ] Produce uncertainty map (wide intervals = high uncertainty zones)
- [ ] Key output: "HP SDMA should act on high-susceptibility + low-uncertainty zones first"

#### 4d — SHAP Spatial Analysis
- [ ] Global SHAP: which factors matter most across HP
- [ ] Local SHAP: which factors drive risk in each district
- [ ] SHAP dependence plots for top 5 factors
- [ ] "Factor importance map" — most important factor per pixel (novel viz)

### Phase 5 — Susceptibility Mapping & Risk Layer

#### 5a — Susceptibility Maps
- [ ] Current susceptibility: 4 classes (low / moderate / high / very high)
- [ ] Uncertainty map: conformal prediction interval widths
- [ ] District-level summary statistics

#### 5b — Infrastructure Risk Overlay
- [ ] Road network (Manali-Leh highway, Chandigarh-Manali highway, district roads)
- [ ] Bridges and crossings
- [ ] Hydroelectric projects (HP has 27 major projects)
- [ ] Settlements: village centroids + population estimates
- [ ] Tourist facilities and peak-season hotspots

#### 5c — Seasonal Risk Briefing
- [ ] Monsoon-season risk score (using seasonal rainfall anomaly)
- [ ] District-level briefing template: top 5 high-risk corridors, recommended actions
- [ ] Output: district PDF briefings + dashboard data

### Phase 6 — Dashboard (Tool)

#### 6a — Turborepo + Next.js Scaffold
- [ ] Init turborepo (pnpm workspaces)
- [ ] apps/web: Next.js 15 App Router
- [ ] packages/ui: shadcn/ui + Tailwind
- [ ] packages/api: tRPC
- [ ] Biome linting config

#### 6b — Dashboard Features
- [ ] **Interactive map** (MapLibre GL JS)
  - HP district boundaries
  - Susceptibility layer (choropleth)
  - Toggle: roads / settlements / hydro projects
  - Click any zone → district briefing panel
- [ ] **District briefing panel**
  - Risk score + uncertainty band
  - Top 3 driving factors (SHAP)
  - Infrastructure at risk (count of roads/villages/hydro in high-risk zone)
  - Comparison vs. state average
- [ ] **Monsoon risk indicator**
  - Current season risk status per district
- [ ] **Factor explorer**
  - Select a conditioning factor → see its contribution map
  - SHAP spatial map overlay
- [ ] **Export**
  - Download district PDF briefing
  - Download GeoJSON susceptibility layer

### Phase 7 — Validation & Paper

#### 7a — Validation
- [ ] Spatial block CV results table (all models)
- [ ] Temporal validation: 2023 monsoon events as independent test
- [ ] Comparison table: GNN vs. RF/XGB/LGB/stacking
- [ ] Conformal prediction coverage analysis (does 90% interval actually contain 90%?)

#### 7b — Paper
- [x] Abstract (placeholder values for XX fields; complete once real data is in)
- [x] Introduction + Research Gap (5 gaps addressed)
- [x] Study Area (HP geography, flood history, 2023 event)
- [x] Data & Methods
  - Conditioning factors + multicollinearity
  - SAR inventory construction
  - GNN architecture
  - Conformal prediction framework
  - Spatial CV design
- [x] Results (skeleton with XX placeholders; fill once real pipeline runs)
- [x] Discussion
- [x] Conclusion
- [ ] Fill XX placeholders with real results (requires real SAR inventory + GEE run)
- [ ] Add figure files (susceptibility map, uncertainty map, SHAP global/spatial)
- [ ] LaTeX compilation check (tectonic)

---

## Architecture Decision Notes

### Why GNN?
Standard ML models (RF, XGBoost) treat every pixel/catchment as independent.
But water flows from upstream to downstream — a high-snowmelt catchment in Lahaul-Spiti
directly raises flood risk 200 km downstream in Kullu. GNNs encode this graph structure.
No published flash flood susceptibility paper uses GNNs as of March 2026.

### Why Conformal Prediction?
Every existing susceptibility map gives a point estimate with no uncertainty.
MAPIE (conformal prediction) provides statistically guaranteed coverage:
"this zone is high-susceptibility with 90% confidence, ±X interval."
HP SDMA can prioritize zones that are high-risk AND high-certainty.
This directly addresses the #1 reviewer criticism of ML susceptibility papers.

### Why SAR inventory?
Optical imagery (Sentinel-2) is cloud-contaminated during monsoon peak.
Sentinel-1 SAR sees through clouds. GEE has full archive from 2018.
SAR-based inventory = more events, more reliable labels, replicable methodology.

### Study Area Decision
Primary: Beas + Satluj basins (highest data availability, HiFlo-DAT, best-documented)
Secondary: Full HP state (extrapolated susceptibility map)
Temporal validation: July–September 2023 (worst HP flood season on record)

---

## Key References
- Valavi et al. 2021: spatial block CV for SDM (applies to flood ML too)
- Saha et al. 2023: only peer-reviewed ML paper for HP (Beas, AUC=0.88) — we beat this
- HiFlo-DAT 2025: Kullu flood database (training data)
- MAPIE library: conformal prediction for scikit-learn compatible models
- PyTorch Geometric: GNN implementation
