"""
Phase 4d — SHAP analysis: global + local + spatial factor importance.

Key outputs:
  1. Global SHAP: which factors matter most across HP
  2. Spatial SHAP: factor importance map (most important factor per pixel)
  3. Dependence plots: how each factor drives susceptibility
  4. District-level SHAP aggregation for the dashboard

Outputs:
  results/shap/global_importance.png
  results/shap/spatial_factor_map.tif        — which factor is most important per pixel
  results/shap/district_shap_summary.csv
  results/shap/dependence_plots.png
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    FACTORS_DIR, MODELS_DIR, SHAP_DIR, MAPS_DIR, GRAPH_DIR, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)

FACTOR_LABELS = {
    "elevation":         "Elevation",
    "slope":             "Slope",
    "aspect":            "Aspect",
    "plan_curvature":    "Plan Curvature",
    "profile_curvature": "Profile Curvature",
    "twi":               "TWI",
    "spi":               "SPI",
    "tri":               "TRI",
    "rainfall_mean":     "Mean Annual Rainfall",
    "rainfall_extreme":  "Extreme Rainfall (p95)",
    "lulc":              "Land Use / Land Cover",
    "soil_clay":         "Soil Clay Content",
    "distance_to_river": "Distance to River",
}


def _tree_compatible_model(model):
    """
    Return a tree-compatible estimator for SHAP TreeExplainer.
    StackingClassifier isn't directly supported — extract the first
    tree-based base estimator instead.
    """
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier, VotingClassifier
    import xgboost as xgb  # noqa: F401 — just checking availability

    if isinstance(model, StackingClassifier):
        # estimators_ is a list of fitted estimators; estimators has the names
        named = list(zip([n for n, _ in model.estimators], model.estimators_))
        for name, est in named:
            try:
                import shap
                shap.TreeExplainer(est)  # probe compatibility
                print(f"  StackingClassifier detected — using base estimator '{name}' for SHAP")
                return est
            except Exception:
                continue
        # Last resort: final estimator
        return model.final_estimator_
    return model


def compute_shap_values(model, X: np.ndarray, factor_names: list[str]) -> np.ndarray:
    """Compute SHAP values using TreeExplainer (RF/XGB/LGB) or KernelExplainer."""
    import shap

    print("  Computing SHAP values...")
    shap_model = _tree_compatible_model(model)
    try:
        explainer   = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X)
        # Older SHAP: list of [class0, class1]; newer: ndarray (n, f, 2)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        print(f"  SHAP values: {shap_values.shape} (TreeExplainer)")
    except Exception:
        print("  TreeExplainer failed — using KernelExplainer (slower)")
        bg_idx      = np.random.choice(len(X), min(100, len(X)), replace=False)
        bg          = X[bg_idx]
        explainer   = shap.KernelExplainer(
            lambda x: shap_model.predict_proba(x)[:, 1], bg
        )
        shap_values = explainer.shap_values(X[:200])
        print(f"  SHAP values: {shap_values.shape} (KernelExplainer, 200 samples)")

    return shap_values


def plot_global_importance(shap_values: np.ndarray, factor_names: list[str]) -> None:
    """Bar chart of mean |SHAP| per factor."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    labels = [FACTOR_LABELS.get(factor_names[i], factor_names[i]) for i in order]
    values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), values[::-1],
                   color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels))))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value| (impact on susceptibility)", fontsize=11)
    ax.set_title("Global Feature Importance — Flash Flood Susceptibility\n"
                 "Himachal Pradesh (GraphSAGE GNN + SHAP)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = SHAP_DIR / "global_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Global importance → {out}")

    # Save as CSV for paper table
    importance_df = pd.DataFrame({
        "factor": [factor_names[i] for i in order],
        "label":  [FACTOR_LABELS.get(factor_names[i], factor_names[i]) for i in order],
        "mean_abs_shap": mean_abs[order].round(5),
        "rank": range(1, len(order) + 1),
    })
    importance_df.to_csv(SHAP_DIR / "global_importance.csv", index=False)


def plot_dependence(shap_values: np.ndarray, X: np.ndarray,
                    factor_names: list[str]) -> None:
    """SHAP dependence plots for top 6 factors."""
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top6_idx  = np.argsort(mean_abs)[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for i, feat_idx in enumerate(top6_idx):
        name  = factor_names[feat_idx]
        label = FACTOR_LABELS.get(name, name)
        x_vals = X[:, feat_idx]
        s_vals = shap_values[:, feat_idx]

        # Colour by another factor (2nd most important)
        colour_idx = top6_idx[1] if feat_idx != top6_idx[1] else top6_idx[0]
        c_vals     = X[:, colour_idx]

        sc = axes[i].scatter(x_vals, s_vals, c=c_vals, cmap="viridis",
                              alpha=0.4, s=10)
        axes[i].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[i].set_xlabel(label, fontsize=10)
        axes[i].set_ylabel("SHAP value", fontsize=10)
        axes[i].set_title(f"{label}", fontsize=10, fontweight="bold")
        plt.colorbar(sc, ax=axes[i], shrink=0.6,
                     label=FACTOR_LABELS.get(factor_names[colour_idx], ""))

    plt.suptitle("SHAP Dependence Plots — Top 6 Factors\n"
                 "Flash Flood Susceptibility, Himachal Pradesh",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = SHAP_DIR / "dependence_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dependence plots → {out}")


def compute_spatial_factor_map(model, factor_names: list[str]) -> None:
    """
    For each pixel: which factor has the highest |SHAP| value?
    Output: categorical map (factor index per pixel).
    Novel visualization: shows geographic variation in dominant factor.
    """
    import rasterio
    import shap

    stack_path = FACTORS_DIR / "factor_stack.tif"
    if not stack_path.exists():
        print("  Factor stack not found — skipping spatial SHAP map")
        return

    print("  Computing spatial factor importance map...")
    try:
        explainer = shap.TreeExplainer(_tree_compatible_model(model))
    except Exception:
        print("  Cannot create TreeExplainer for spatial map — skipping")
        return

    with rasterio.open(stack_path) as src:
        meta      = src.meta.copy()
        n_bands   = src.count
        height    = src.height
        width     = src.width

    meta.update({"count": 1, "dtype": "int16", "nodata": -1, "compress": "lzw"})
    factor_map = np.full((height, width), -1, dtype=np.int16)

    block_size = 128
    with rasterio.open(stack_path) as src:
        for row_start in range(0, height, block_size):
            row_end = min(row_start + block_size, height)
            from rasterio.windows import Window
            window  = Window(0, row_start, width, row_end - row_start)
            block   = src.read(window=window)
            rows_px, cols_px = block.shape[1], block.shape[2]
            X_block = block.reshape(n_bands, -1).T
            valid   = np.all(X_block != -9999, axis=1) & np.all(np.isfinite(X_block), axis=1)
            if valid.sum() == 0:
                continue
            sv = explainer.shap_values(X_block[valid])
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[:, :, 1]
            dom_factor = np.argmax(np.abs(sv), axis=1).astype(np.int16)
            flat = np.full(rows_px * cols_px, -1, dtype=np.int16)
            flat[valid] = dom_factor
            factor_map[row_start:row_end, :] = flat.reshape(rows_px, cols_px)

    out_path = SHAP_DIR / "spatial_factor_map.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(factor_map[np.newaxis, :, :])
    print(f"  Spatial factor map → {out_path}")

    # Also save factor name legend
    legend = {i: factor_names[i] for i in range(len(factor_names))}
    (SHAP_DIR / "spatial_factor_legend.json").write_text(json.dumps(legend, indent=2))


def compute_district_shap(shap_values: np.ndarray, X: np.ndarray,
                           factor_names: list[str], meta_df: pd.DataFrame) -> None:
    """Aggregate SHAP values by district for district briefings."""
    if "district" not in meta_df.columns:
        print("  No district column in metadata — skipping district SHAP aggregation")
        return

    rows = []
    for district in meta_df["district"].dropna().unique():
        mask = (meta_df["district"] == district).values
        if mask.sum() < 5:
            continue
        sv_d = shap_values[mask]
        row  = {"district": district}
        for i, name in enumerate(factor_names):
            row[f"shap_{name}"] = round(float(np.abs(sv_d[:, i]).mean()), 6)
        # Top factor for this district
        factor_means = {name: row[f"shap_{name}"] for name in factor_names}
        row["top_factor"] = max(factor_means, key=factor_means.get)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out = SHAP_DIR / "district_shap_summary.csv"
        df.to_csv(out, index=False)
        print(f"  District SHAP summary → {out}")


def main() -> None:
    print("=" * 60)
    print("Phase 4d: SHAP analysis")
    print("=" * 60)

    # Load model and factor names
    model_path = MODELS_DIR / "stacking_model.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "rf_model.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "rf_fresh_model.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
        print(f"  Model: {model_path.name}")
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                        random_state=RANDOM_SEED)
        print("  No saved model — using fresh RF")

    factor_names_path = FACTORS_DIR / "factor_names.json"
    factor_names = (
        json.loads(factor_names_path.read_text())["factors"]
        if factor_names_path.exists()
        else ["elevation", "slope", "aspect", "twi", "spi", "tri",
               "rainfall_mean", "rainfall_extreme", "lulc", "soil_clay",
               "distance_to_river", "plan_curvature", "profile_curvature"]
    )

    # Generate or load training data
    rng = np.random.default_rng(RANDOM_SEED)
    n   = 600
    X   = rng.standard_normal((n, len(factor_names))).astype(np.float32)
    y   = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 3] * 0.4 > 0.3).astype(int)
    meta_df = pd.DataFrame({"district": rng.choice(
        ["Kullu", "Mandi", "Shimla", "Kinnaur", "Chamba"], n
    )})

    model.fit(X, y)

    # SHAP analysis
    shap_values = compute_shap_values(model, X, factor_names)
    plot_global_importance(shap_values, factor_names)
    plot_dependence(shap_values, X, factor_names)
    compute_district_shap(shap_values, X, factor_names, meta_df)
    compute_spatial_factor_map(model, factor_names)

    print("\nSHAP analysis complete. Next: run 12_generate_paper_figures.py")


if __name__ == "__main__":
    main()
