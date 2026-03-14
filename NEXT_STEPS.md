# Next Steps — Data Integration

## When GEE Tasks Complete

### Step 1: Download from Google Drive
Open Google Drive → `hp_flood_inventory/` folder and download all 8 TIF files:
- `flood_train_2018.tif`
- `flood_train_2019.tif`
- `flood_train_2020.tif`
- `flood_train_2021.tif`
- `flood_train_2022.tif`
- `flood_test_2023.tif`
- `rainfall_mean_annual_gpm.tif`
- `rainfall_max_monthly_gpm.tif`

### Step 2: Place files in project
```
data/raw/flood_inventory/sar/
    flood_train_2018.tif
    flood_train_2019.tif
    flood_train_2020.tif
    flood_train_2021.tif
    flood_train_2022.tif
    flood_test_2023.tif

data/raw/rainfall/
    rainfall_mean_annual_gpm.tif
    rainfall_max_monthly_gpm.tif
```

### Step 3: Run pipeline
```bash
# (DEM tiles download automatically — already running)
# Wait for DEM + GEE data, then:
uv run python scripts/04_preprocess_terrain.py   # terrain factors from DEM
uv run python scripts/05_watershed_delineation.py # watershed graph
uv run python scripts/06_assemble_factors.py     # combine all factors
uv run python scripts/07_build_flood_inventory.py # flood point dataset
uv run python scripts/08_train_baseline_models.py # RF/XGB/LGB/Stacking
uv run python scripts/09_train_gnn.py            # GraphSAGE GNN
uv run python scripts/10_conformal_prediction.py  # uncertainty intervals
uv run python scripts/11_shap_analysis.py        # SHAP importance maps
uv run python scripts/12_generate_paper_figures.py # paper figures
# Then fill paper/chapters/*.tex with real numbers
```

## DEM Status
- 16 tiles needed (lat 30-33, lon 75-78)
- Downloading from Copernicus GLO-30 AWS (no auth)
- Check: `ls data/raw/dem/*.tif | wc -l`  (expect 16 when done)
