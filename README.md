# Flash Flood Susceptibility Mapping — Himachal Pradesh

State-wide flash flood susceptibility assessment using Graph Neural Networks and Conformal Prediction. First multi-basin, uncertainty-aware susceptibility map for HP.

**Paper:** [`paper/main.pdf`](paper/main.pdf)

## Key Results

| Model | AUC-ROC (LOBO spatial CV) |
|-------|--------------------------|
| GNN-GraphSAGE | **0.995 ± 0.004** |
| Stacking Ensemble | 0.901 ± 0.004 |
| Random Forest | 0.900 ± 0.004 |
| Saha 2023 benchmark | 0.880 (no spatial CV) |

- 15,785 km² (14.3% of HP) classified as High or Very High susceptibility
- 82.9% conformal coverage on 2023 temporal test set
- 1,457 km highways and 2,759 bridges exposed in high-risk zones

## Method

1. **Flood inventory**: 3,000 points from Sentinel-1 SAR seasonal composites (2018–2022 train; 2023 test)
2. **12 conditioning factors**: terrain (GLO-30 DEM), rainfall (GPM-IMERG), LULC (ESA WorldCover), soil (SoilGrids)
3. **GNN-GraphSAGE** on a 460-node directed watershed graph — captures upstream→downstream flood propagation
4. **Conformal prediction** (split-conformal, α=0.10) — uncertainty-quantified susceptibility maps
5. **Leave-one-basin-out** spatial block CV across 5 HP river basins

## Dashboard

```bash
uv sync
uv run streamlit run dashboard/app.py   # interactive Streamlit app
uv run python dashboard/generate_html_dashboard.py  # static HTML version
```

## Run Pipeline

```bash
uv sync
uv run python scripts/08_train_baseline_models.py
uv run python scripts/09_train_gnn.py
uv run python scripts/10_conformal_prediction.py
uv run python scripts/11_shap_analysis.py
```

Compile paper: `cd paper && tectonic main.tex`

## Repository

```
scripts/          # pipeline (00–17)
paper/            # LaTeX manuscript → main.pdf
results/          # model outputs (gitignored)
data/             # raw + processed data (gitignored)
dashboard/        # Streamlit + HTML dashboard
```
