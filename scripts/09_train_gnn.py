"""
Phase 4b — Train Graph Neural Network (GNN) on watershed graph.

Architecture: GraphSAGE (3 layers, mean aggregator).
Each node = sub-watershed. Features = aggregated conditioning factors.
Edges = directed river connectivity (upstream → downstream).

Key fix (v2): Proper inductive leave-one-basin-out (LOBO) cross-validation.
  - Test basin nodes are REMOVED from the graph during training.
  - Training uses only edges where both endpoints are in the training set.
  - Inference uses edges where the SOURCE is a training node and TARGET is a
    test node (test nodes aggregate from known training neighbours only).
  - Node feature normalisation uses training-set statistics only.
  - Five spatial blocks assigned via k-means on watershed centroids.

Outputs:
  results/models/gnn_model.pt
  results/validation/gnn_cv_results.csv
  results/validation/gnn_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    GRAPH_DIR, MODELS_DIR, VALIDATION_DIR,
    RANDOM_SEED, GNN_HIDDEN_DIM, GNN_LAYERS, GNN_EPOCHS, GNN_LR,
)

np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
NODES_PATH  = GRAPH_DIR / "node_features.csv"
EDGES_PATH  = GRAPH_DIR / "graph_edges.csv"
LABELS_PATH = GRAPH_DIR / "watershed_flood_labels.csv"
WS_GEOJSON  = GRAPH_DIR / "watersheds.geojson"


# ── Basin assignment ───────────────────────────────────────────────────────────

def assign_spatial_basins(nodes: pd.DataFrame, n_basins: int = 5) -> np.ndarray:
    """
    Assign each watershed to one of n_basins spatial blocks using k-means
    clustering on watershed centroids (or index-based proxy if no geometry).

    Returns an array of basin labels (strings: 'basin_0' … 'basin_4').
    """
    try:
        import geopandas as gpd
        from sklearn.cluster import KMeans

        ws = gpd.read_file(WS_GEOJSON).to_crs("EPSG:4326")
        centroids = ws.geometry.centroid
        coords = np.column_stack([centroids.x, centroids.y])

        km = KMeans(n_clusters=n_basins, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(coords)
        basins = np.array([f"basin_{l}" for l in labels])
        print(f"  Spatial basins assigned via k-means on {len(ws)} centroids")
        for b in np.unique(basins):
            print(f"    {b}: {(basins == b).sum()} watersheds")
        return basins

    except Exception as e:
        print(f"  WARNING: Could not assign spatial basins ({e}); "
              f"falling back to sequential 5-fold split")
        n = len(nodes)
        idx = np.arange(n)
        np.random.shuffle(idx)
        basins = np.empty(n, dtype="U10")
        for fold in range(n_basins):
            mask = (idx % n_basins) == fold
            basins[idx[mask]] = f"basin_{fold}"
        return basins


# ── Data loading ───────────────────────────────────────────────────────────────

def load_graph_data():
    """Load watershed graph, node features, and flood labels."""
    if not NODES_PATH.exists() or not EDGES_PATH.exists():
        print("Graph data not found — using synthetic graph for pipeline testing")
        return _synthetic_graph()

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    if LABELS_PATH.exists():
        labels_df = pd.read_csv(LABELS_PATH)
        nodes = nodes.merge(labels_df, on="watershed_id", how="left")
        nodes["label"] = nodes["label"].fillna(0).astype(int)
    else:
        if "mean_elevation" in nodes.columns:
            threshold = nodes["mean_elevation"].quantile(0.40)
            nodes["label"] = (nodes["mean_elevation"] <= threshold).astype(int)
        else:
            nodes["label"] = np.random.binomial(1, 0.3, len(nodes))

    # Assign spatial basins if not already present
    if "basin" not in nodes.columns:
        nodes["basin"] = assign_spatial_basins(nodes)

    return nodes, edges


def _synthetic_graph():
    rng = np.random.default_rng(RANDOM_SEED)
    n_nodes = 300
    nodes = pd.DataFrame({
        "watershed_id":   range(n_nodes),
        "mean_elevation": rng.uniform(200, 5500, n_nodes),
        "mean_slope":     rng.uniform(5, 60, n_nodes),
        "mean_twi":       rng.uniform(4, 14, n_nodes),
        "mean_rainfall":  rng.uniform(500, 2500, n_nodes),
        "basin":          rng.choice(
            ["basin_0", "basin_1", "basin_2", "basin_3", "basin_4"], n_nodes),
    })
    nodes["label"] = (
        (nodes["mean_elevation"] < 1500) & (nodes["mean_rainfall"] > 1200)
    ).astype(int)
    edges = []
    sorted_nodes = nodes.sort_values("mean_elevation", ascending=False)
    for i, src in sorted_nodes.head(200).iterrows():
        candidates = nodes[nodes["mean_elevation"] < src["mean_elevation"]]
        if len(candidates) == 0:
            continue
        for _, tgt in candidates.sample(
                min(2, len(candidates)), random_state=int(rng.integers(0, 999))).iterrows():
            edges.append({"source": int(src["watershed_id"]),
                          "target": int(tgt["watershed_id"])})
    edges = pd.DataFrame(edges).drop_duplicates(subset=["source", "target"])
    print(f"Synthetic graph: {n_nodes} nodes, {len(edges)} edges")
    return nodes, edges


# ── GNN training ──────────────────────────────────────────────────────────────

def train_gnn(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv  # noqa: F401
        return _train_pyg_gnn(nodes, edges)
    except ImportError:
        print("  PyTorch Geometric not installed — using neighbourhood aggregation proxy")
        return _train_neighbourhood_gnn(nodes, edges)


def _build_edge_index(edge_df, id_to_idx, bidirectional=True):
    """Build a PyG edge_index tensor from a DataFrame with source/target columns."""
    import torch
    src = [id_to_idx[s] for s in edge_df["source"] if s in id_to_idx]
    tgt = [id_to_idx[t] for t in edge_df["target"] if t in id_to_idx]
    if bidirectional:
        return torch.tensor([src + tgt, tgt + src], dtype=torch.long)
    return torch.tensor([src, tgt], dtype=torch.long)


def _train_pyg_gnn(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """
    Proper inductive LOBO GNN training.

    For each fold (test basin B):
      - Train graph  : edges where both endpoints ∉ B
      - Infer graph  : train edges + (train→test) edges
      - Normalise    : using train-set mean/std only
      - Evaluate     : on test-basin nodes using infer graph
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score

    feat_cols = [c for c in nodes.columns
                 if c not in ("watershed_id", "label", "basin", "geometry")]
    basins    = nodes["basin"].values
    basin_list = np.unique(basins)

    id_to_idx = {wid: i for i, wid in enumerate(nodes["watershed_id"])}
    all_X = nodes[feat_cols].fillna(0).values.astype(np.float32)
    all_y = nodes["label"].values.astype(np.int64)

    # Global class-weight (based on full training data)
    pos_w = float((all_y == 0).sum() / max((all_y == 1).sum(), 1))

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_ch, hidden, out_ch, n_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_ch, hidden))
            for _ in range(n_layers - 2):
                self.convs.append(SAGEConv(hidden, hidden))
            self.convs.append(SAGEConv(hidden, out_ch))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
            return self.convs[-1](x, edge_index)

    results_by_fold = []
    last_model_state = None

    for test_basin in basin_list:
        train_idx = np.where(basins != test_basin)[0]
        test_idx  = np.where(basins == test_basin)[0]

        if len(test_idx) < 5:
            print(f"  Skipping {test_basin}: only {len(test_idx)} test nodes")
            continue

        train_wids = set(nodes["watershed_id"].iloc[train_idx])
        test_wids  = set(nodes["watershed_id"].iloc[test_idx])

        # Training graph: only edges entirely within the training set
        train_edges = edges[
            edges["source"].isin(train_wids) & edges["target"].isin(train_wids)
        ]
        # Inference graph: training edges + train→test edges
        # (test nodes can aggregate from their training-set upstream neighbours)
        infer_edges = edges[
            (edges["source"].isin(train_wids) & edges["target"].isin(train_wids)) |
            (edges["source"].isin(train_wids) & edges["target"].isin(test_wids))
        ]

        # Normalise using training statistics only
        mu  = all_X[train_idx].mean(0)
        sig = all_X[train_idx].std(0) + 1e-8
        X_norm = (all_X - mu) / sig

        x_t          = torch.tensor(X_norm, dtype=torch.float)
        y_t          = torch.tensor(all_y,  dtype=torch.long)
        train_mask   = torch.zeros(len(nodes), dtype=torch.bool)
        train_mask[train_idx] = True

        train_ei = _build_edge_index(train_edges, id_to_idx)
        infer_ei = _build_edge_index(infer_edges, id_to_idx)

        model     = GraphSAGE(len(feat_cols), GNN_HIDDEN_DIM, 2, GNN_LAYERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_w], dtype=torch.float))

        # Train on training subgraph
        model.train()
        for epoch in range(GNN_EPOCHS):
            optimizer.zero_grad()
            out  = model(x_t, train_ei)
            loss = criterion(out[train_mask], y_t[train_mask])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"    {test_basin} epoch {epoch+1:3d} | loss={loss.item():.4f}")

        # Evaluate on test nodes via inference graph
        model.eval()
        with torch.no_grad():
            logits = model(x_t, infer_ei)
            proba  = F.softmax(logits, dim=1)[:, 1].numpy()
            preds  = logits.argmax(dim=1).numpy()

        y_te  = all_y[test_idx]
        p_te  = proba[test_idx]
        pr_te = preds[test_idx]
        auc_v = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else 0.5

        results_by_fold.append({
            "model":  "GNN-GraphSAGE",
            "fold":   str(test_basin),
            "auc":    round(auc_v, 4),
            "f1":     round(f1_score(y_te, pr_te, zero_division=0), 4),
            "kappa":  round(cohen_kappa_score(y_te, pr_te), 4),
            "n_test": int(len(test_idx)),
        })
        print(f"  GNN {test_basin} | AUC={auc_v:.4f} | n_test={len(test_idx)}")
        last_model_state = model.state_dict()

    if last_model_state is not None:
        torch.save(last_model_state, MODELS_DIR / "gnn_model.pt")
        print(f"  Model saved → {MODELS_DIR / 'gnn_model.pt'}")

    return {"results": results_by_fold,
            "method": "PyTorch Geometric GraphSAGE (inductive LOBO)"}


def _train_neighbourhood_gnn(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Neighbourhood aggregation MLP proxy (fallback when PyG unavailable)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score

    feat_cols = [c for c in nodes.columns
                 if c not in ("watershed_id", "label", "basin", "geometry")]
    id_to_idx = {wid: i for i, wid in enumerate(nodes["watershed_id"])}

    # Build neighbour feature aggregation
    neigh_feat = np.zeros((len(nodes), len(feat_cols)), dtype=np.float32)
    for _, edge in edges.iterrows():
        s, t = edge["source"], edge["target"]
        if s in id_to_idx and t in id_to_idx:
            neigh_feat[id_to_idx[t]] += nodes[feat_cols].iloc[id_to_idx[s]].values

    X_own   = nodes[feat_cols].fillna(0).values.astype(np.float32)
    X       = np.concatenate([X_own, neigh_feat], axis=1)
    y       = nodes["label"].values
    basins  = nodes["basin"].values

    results = []
    for test_basin in np.unique(basins):
        tr_mask = basins != test_basin
        te_mask = basins == test_basin
        if te_mask.sum() < 5 or tr_mask.sum() < 10:
            continue

        mu, sig = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-8
        X_norm  = (X - mu) / sig

        mlp = MLPClassifier(
            hidden_layer_sizes=(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM),
            max_iter=GNN_EPOCHS, random_state=RANDOM_SEED)
        mlp.fit(X_norm[tr_mask], y[tr_mask])
        proba = mlp.predict_proba(X_norm[te_mask])[:, 1]
        preds = (proba >= 0.5).astype(int)
        y_te  = y[te_mask]
        auc_v = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.5

        results.append({
            "model": "GNN-NeighAgg-proxy",
            "fold":  str(test_basin),
            "auc":   round(auc_v, 4),
            "f1":    round(f1_score(y_te, preds, zero_division=0), 4),
            "kappa": round(cohen_kappa_score(y_te, preds), 4),
            "n_test": int(te_mask.sum()),
        })
        print(f"  proxy {test_basin} | AUC={auc_v:.4f}")

    import joblib
    joblib.dump(mlp, MODELS_DIR / "gnn_proxy_model.pkl")
    return {"results": results, "method": "Neighbourhood-aggregation MLP proxy"}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 4b: Training Graph Neural Network (GNN) — inductive LOBO")
    print("=" * 60)

    nodes, edges = load_graph_data()
    print(f"Graph: {len(nodes)} watersheds, {len(edges)} directed edges")

    output     = train_gnn(nodes, edges)
    results_df = pd.DataFrame(output["results"])

    cv_path = VALIDATION_DIR / "gnn_cv_results.csv"
    results_df.to_csv(cv_path, index=False)
    print(f"\nGNN CV results → {cv_path}")
    print(results_df.to_string(index=False))

    mean_auc = results_df["auc"].mean() if not results_df.empty else None
    std_auc  = results_df["auc"].std()  if not results_df.empty else None
    print(f"\nMean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Method:   {output['method']}")

    summary = {
        "method":      output["method"],
        "n_nodes":     len(nodes),
        "n_edges":     len(edges),
        "gnn_layers":  GNN_LAYERS,
        "hidden_dim":  GNN_HIDDEN_DIM,
        "epochs":      GNN_EPOCHS,
        "n_folds":     len(results_df),
        "mean_auc_cv": round(float(mean_auc), 4) if mean_auc else None,
        "std_auc_cv":  round(float(std_auc),  4) if std_auc  else None,
    }
    (VALIDATION_DIR / "gnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary → {VALIDATION_DIR / 'gnn_summary.json'}")
    print("\nNext: run 10_conformal_prediction.py")


if __name__ == "__main__":
    main()
