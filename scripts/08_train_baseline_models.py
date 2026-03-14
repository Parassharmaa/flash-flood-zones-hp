"""
Phase 4a — Train baseline ML models: RF, XGBoost, LightGBM + stacking ensemble.

Validation: spatial block cross-validation (leave-one-basin-out).
Metrics: AUC-ROC, F1, Kappa, Precision, Recall.

Outputs:
  results/models/rf_model.pkl
  results/models/xgb_model.pkl
  results/models/lgb_model.pkl
  results/models/stacking_model.pkl
  results/validation/baseline_cv_results.csv
  results/validation/baseline_comparison.png
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, cohen_kappa_score, f1_score,
    precision_score, recall_score, roc_curve,
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    INVENTORY_DIR, FACTORS_DIR, GRAPH_DIR, MODELS_DIR, VALIDATION_DIR,
    RANDOM_SEED, N_ESTIMATORS, MAX_DEPTH, CV_FOLDS,
)

np.random.seed(RANDOM_SEED)

# Basins for spatial block CV (leave-one-basin-out)
BASINS = ["Beas", "Satluj", "Chenab", "Ravi", "Yamuna"]


def load_training_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load factor values + labels at flood/non-flood points."""
    factor_names_path = FACTORS_DIR / "factor_names.json"
    stack_path        = FACTORS_DIR / "factor_stack.tif"

    if not stack_path.exists() or not factor_names_path.exists():
        print("Factor stack not found — generating synthetic data for pipeline testing")
        return _synthetic_data()

    import rasterio
    from rasterio.sample import sample_gen

    factor_names = json.loads(factor_names_path.read_text())["factors"]

    flood_gdf    = gpd.read_file(INVENTORY_DIR / "flood_points.geojson")
    nonflood_gdf = gpd.read_file(INVENTORY_DIR / "nonflood_points.geojson")

    # Training split only
    flood_train    = flood_gdf[flood_gdf["split"] == "train"]
    nonflood_train = nonflood_gdf[nonflood_gdf["split"] == "train"]
    all_points     = pd.concat([flood_train, nonflood_train], ignore_index=True)

    # Sample raster values at points — reproject to raster CRS first
    with rasterio.open(stack_path) as src:
        raster_crs = src.crs
        all_points_proj = all_points.to_crs(raster_crs)
        coords = [(geom.x, geom.y) for geom in all_points_proj.geometry]
        sampled = list(sample_gen(src, coords))
    X = np.array(sampled, dtype=np.float32)
    y = all_points["label"].values

    # Remove nodata rows
    valid = np.all(X != -9999, axis=1) & np.all(np.isfinite(X), axis=1)
    X, y  = X[valid], y[valid]
    meta  = all_points[valid].reset_index(drop=True)
    meta.columns = [c for c in meta.columns]

    print(f"Training data: {X.shape[0]} samples × {X.shape[1]} factors")
    print(f"  Flood: {y.sum()}, Non-flood: {(y==0).sum()}")
    return X, y, meta


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Synthetic data for pipeline testing before real data is available."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_flood    = 300
    n_nonflood = 1500

    # Flood points: high slope, high rainfall, low elevation, close to river
    X_flood = rng.normal(
        loc=[500, 30, 180, -0.1, -0.2, 8, 1000, 5, 1200, 80, 3, 40, 200],
        scale=[200, 10, 60,  0.3,  0.3, 2, 500,  2, 300,  20, 1, 15, 100],
        size=(n_flood, 13),
    )
    X_nonflood = rng.normal(
        loc=[2000, 20, 180, 0.0, 0.0, 6, 300, 4, 900, 50, 5, 30, 1500],
        scale=[500, 8,  60, 0.3, 0.3, 2, 200, 2, 200, 15, 2, 10, 500],
        size=(n_nonflood, 13),
    )
    X = np.vstack([X_flood, X_nonflood]).astype(np.float32)
    y = np.array([1]*n_flood + [0]*n_nonflood)

    factor_names = [
        "elevation", "slope", "aspect", "plan_curvature", "profile_curvature",
        "twi", "spi", "tri", "rainfall_mean", "rainfall_extreme",
        "lulc", "soil_clay", "distance_to_river",
    ]
    basins_pool = ["Beas", "Satluj", "Chenab", "Ravi"]
    meta = pd.DataFrame({
        "label": y,
        "basin": rng.choice(basins_pool, len(y)),
        "split": ["train"] * len(y),
    })
    (FACTORS_DIR / "factor_names.json").write_text(
        json.dumps({"factors": factor_names, "n_factors": len(factor_names)}, indent=2)
    )
    print(f"Synthetic data: {len(y)} samples × {X.shape[1]} factors (PLACEHOLDER)")
    return X, y, meta


def spatial_block_cv(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """
    Leave-one-basin-out spatial cross-validation.
    Each fold withholds all points from one basin as test set.
    """
    print("\nSpatial block CV (leave-one-basin-out)...")

    if "basin" not in meta.columns:
        print("  No basin column — falling back to random 5-fold CV")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        folds = [(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)]
        fold_labels = [f"fold_{i}" for i in range(CV_FOLDS)]
    else:
        basins_present = meta["basin"].unique()
        folds = []
        fold_labels = []
        for basin in basins_present:
            test_mask  = (meta["basin"] == basin).values
            train_mask = ~test_mask
            if test_mask.sum() < 5:
                continue
            folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
            fold_labels.append(basin)

    results = []
    for fold_name, (train_idx, test_idx) in zip(fold_labels, folds):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_te.sum() < 3:
            continue  # skip folds with too few positive samples

        for model_name, model in models.items():
            model.fit(X_tr, y_tr)
            proba  = model.predict_proba(X_te)[:, 1]
            preds  = (proba >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc_val = auc(fpr, tpr)
            results.append({
                "model":     model_name,
                "fold":      fold_name,
                "auc":       round(auc_val, 4),
                "f1":        round(f1_score(y_te, preds, zero_division=0), 4),
                "kappa":     round(cohen_kappa_score(y_te, preds), 4),
                "precision": round(precision_score(y_te, preds, zero_division=0), 4),
                "recall":    round(recall_score(y_te, preds, zero_division=0), 4),
                "n_test":    len(y_te),
                "n_pos":     int(y_te.sum()),
            })
            print(f"  {model_name:20s} | {fold_name:12s} | AUC={auc_val:.3f}")

    return pd.DataFrame(results)


def build_models() -> dict:
    """Instantiate all baseline models."""
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        scale_pos_weight=5,  # handle class imbalance
        random_state=RANDOM_SEED, eval_metric="logloss",
        verbosity=0,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        class_weight="balanced", random_state=RANDOM_SEED,
        verbose=-1,
    )
    stacking = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_model), ("lgb", lgb_model)],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False, cv=3, n_jobs=-1,
    )
    return {"RF": rf, "XGBoost": xgb_model, "LightGBM": lgb_model, "Stacking": stacking}


def plot_cv_results(results_df: pd.DataFrame) -> None:
    """Plot AUC distribution per model across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of AUC per model
    model_names = results_df["model"].unique()
    auc_by_model = [results_df[results_df["model"] == m]["auc"].values for m in model_names]
    axes[0].boxplot(auc_by_model, labels=model_names)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("AUC Distribution — Spatial Block CV")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].axhline(0.88, color="gray", linestyle="--", alpha=0.5,
                    label="Saha et al. 2023 benchmark (0.88)")
    axes[0].legend(fontsize=8)

    # Heatmap: model × fold
    pivot = results_df.pivot_table(values="auc", index="model", columns="fold")
    im = axes[1].imshow(pivot.values, vmin=0.6, vmax=1.0, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=axes[1], label="AUC")
    axes[1].set_title("AUC by Model × Basin Fold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out = VALIDATION_DIR / "baseline_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot → {out}")


def main() -> None:
    print("=" * 60)
    print("Phase 4a: Training baseline models (RF, XGB, LGB, Stacking)")
    print("=" * 60)

    X, y, meta = load_training_data()
    models     = build_models()

    # Cross-validation
    cv_results = spatial_block_cv(X, y, meta, models)
    cv_path    = VALIDATION_DIR / "baseline_cv_results.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"\nCV results → {cv_path}")

    # Summary table
    summary = cv_results.groupby("model")[["auc", "f1", "kappa"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    print("\nCV Summary:")
    print(summary.round(4).to_string())

    # Plot
    plot_cv_results(cv_results)

    # Fit final models on all training data
    print("\nFitting final models on full training set...")
    for name, model in models.items():
        model.fit(X, y)
        out = MODELS_DIR / f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, out)
        print(f"  Saved {name} → {out}")

    # Save factor names for later use
    factor_names_path = FACTORS_DIR / "factor_names.json"
    if factor_names_path.exists():
        summary_dict = {
            "models_trained": list(models.keys()),
            "cv_mean_auc": cv_results.groupby("model")["auc"].mean().round(4).to_dict(),
            "benchmark_auc_saha2023": 0.88,
            "note": "Spatial block CV (leave-one-basin-out)",
        }
        (VALIDATION_DIR / "baseline_summary.json").write_text(
            json.dumps(summary_dict, indent=2)
        )

    print("\nBaseline models complete. Next: run 09_train_gnn.py")


if __name__ == "__main__":
    main()
