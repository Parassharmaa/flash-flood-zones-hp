"""
Phase 2 — Delineate sub-watersheds and build the GNN watershed graph.

This is the key preprocessing step for the Graph Neural Network.
Each sub-watershed becomes a node; river connections become directed edges.

Outputs:
  data/processed/watershed_graph/watersheds.geojson  — sub-watershed polygons
  data/processed/watershed_graph/graph_nodes.csv     — node features
  data/processed/watershed_graph/graph_edges.csv     — directed edges (upstream→downstream)
  data/processed/watershed_graph/adjacency.npz       — sparse adjacency matrix
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

# pysheds 0.5 uses np.in1d which was removed in NumPy 2.0
if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, **kw: np.isin(ar1, ar2, **kw).ravel()

import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
from config import TERRAIN_DIR, GRAPH_DIR, HP_CRS_UTM  # noqa: E402


# Target: ~300–500 sub-watersheds for HP
# Adjust MIN_CATCHMENT_AREA_KM2 to get desired number
MIN_CATCHMENT_AREA_KM2 = 150   # minimum watershed area in km²


def delineate_watersheds(dem_path: Path) -> gpd.GeoDataFrame:
    """
    Delineate sub-watersheds using D8 flow accumulation.
    Uses pysheds if available; falls back to simplified approach.
    """
    print("Delineating sub-watersheds...")

    try:
        from pysheds.grid import Grid

        grid    = Grid.from_raster(str(dem_path))
        dem_arr = grid.read_raster(str(dem_path))

        # Condition DEM
        pit_filled = grid.fill_pits(dem_arr)
        flooded    = grid.fill_depressions(pit_filled)
        inflated   = grid.resolve_flats(flooded)

        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        fdir   = grid.flowdir(inflated, dirmap=dirmap)
        acc    = grid.accumulation(fdir, dirmap=dirmap)

        with rasterio.open(dem_path) as src:
            cell_size = src.res[0]
            transform = src.transform
            crs       = src.crs

        # Sample pour points: regular grid of stream cells spaced by target spacing
        # Each pour point → one sub-watershed
        acc_arr = np.array(acc)
        threshold_px = int((MIN_CATCHMENT_AREA_KM2 * 1e6) / (cell_size**2))

        # Pour points: stream pixels, sampled on regular grid to control density
        spacing = max(1, int(np.sqrt(threshold_px)))
        rows_idx, cols_idx = np.where(acc_arr > threshold_px)
        # Subsample to regular grid
        mask = ((rows_idx % spacing == 0) & (cols_idx % spacing == 0))
        rows_idx, cols_idx = rows_idx[mask], cols_idx[mask]
        print(f"  {len(rows_idx)} pour points sampled (spacing={spacing}px)")

        # For each pour point, get catchment using pysheds
        features = []
        from rasterio.transform import rowcol, xy as rast_xy

        for r, c in zip(rows_idx[:500], cols_idx[:500]):   # cap at 500
            x, y = rasterio.transform.xy(transform, r, c)
            try:
                catch = grid.catchment(x, y, fdir=fdir, dirmap=dirmap, xytype='coordinate')
                catch_mask = np.array(catch).astype(np.uint8)
                area_px = int(catch_mask.sum())
                area_km2 = area_px * (cell_size ** 2) / 1e6
                if area_km2 < MIN_CATCHMENT_AREA_KM2 * 0.1:
                    continue
                results = list(shapes(catch_mask, mask=catch_mask, transform=transform))
                for geom, _ in results:
                    features.append({
                        "geometry": shape(geom),
                        "catchment_id": len(features),
                        "area_km2": round(area_km2, 2),
                    })
            except Exception:
                continue

        print(f"  Found {len(features)} raw catchments (before area filter)")
        if len(features) < 10:
            raise ValueError(f"Too few catchments ({len(features)}), falling back")

        gdf = gpd.GeoDataFrame(features, crs=crs)
        method = "pysheds D8 pour-point"

    except (ImportError, ValueError, Exception) as e:
        print(f"  pysheds catchment delineation failed ({e}) — using grid proxy")
        gdf, method = _simple_watershed_proxy(dem_path)

    # Filter by minimum area
    gdf = gdf[gdf["area_km2"] >= MIN_CATCHMENT_AREA_KM2 * 0.5].reset_index(drop=True)
    gdf["watershed_id"] = range(len(gdf))

    print(f"  Method: {method}")
    print(f"  Watersheds after area filter: {len(gdf)}")
    return gdf


def _simple_watershed_proxy(dem_path: Path) -> tuple[gpd.GeoDataFrame, str]:
    """
    Fallback: divide HP into a regular grid of sub-areas as watershed proxies.
    Used only when pysheds is unavailable.
    """
    import shapely.geometry as sg

    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        crs    = src.crs

    # ~20×20 km grid → ~300-400 cells for HP
    cell_size_m = 20_000
    xs = np.arange(bounds.left,  bounds.right, cell_size_m)
    ys = np.arange(bounds.bottom, bounds.top,  cell_size_m)

    features = []
    cid = 0
    for x in xs:
        for y in ys:
            poly = sg.box(x, y, x + cell_size_m, y + cell_size_m)
            features.append({
                "geometry": poly,
                "catchment_id": cid,
                "area_km2": (cell_size_m / 1000)**2,
            })
            cid += 1

    return gpd.GeoDataFrame(features, crs=crs), "regular-grid proxy"


def build_graph(watersheds: gpd.GeoDataFrame, dem_path: Path) -> pd.DataFrame:
    """
    Build directed graph edges: upstream_id → downstream_id.

    Logic: for each watershed, the neighbouring watershed with the lowest
    mean elevation is the downstream neighbour.
    """
    print("Building watershed connectivity graph...")

    # Compute mean elevation per watershed
    with rasterio.open(dem_path) as src:
        elev_data = src.read(1).astype(np.float32)
        transform = src.transform

    from rasterio.features import rasterize

    mean_elevs = {}
    for _, row in watersheds.iterrows():
        wid = row["watershed_id"]
        mask = rasterize(
            [(mapping(row.geometry), 1)],
            out_shape=elev_data.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        vals = elev_data[mask == 1]
        mean_elevs[wid] = float(np.nanmean(vals)) if len(vals) > 0 else 9999.0

    watersheds["mean_elevation"] = watersheds["watershed_id"].map(mean_elevs)

    # Find neighbours (shared boundary)
    edges = []
    for idx, row in watersheds.iterrows():
        wid    = row["watershed_id"]
        elev_i = mean_elevs[wid]
        # Buffer slightly to find touching neighbours
        buffered = row.geometry.buffer(100)
        neighbours = watersheds[
            (watersheds.index != idx) &
            (watersheds.geometry.intersects(buffered))
        ]
        for _, nb in neighbours.iterrows():
            nb_id   = nb["watershed_id"]
            elev_nb = mean_elevs[nb_id]
            # Edge direction: higher elevation → lower elevation
            if elev_i > elev_nb:
                edges.append({
                    "source": wid,
                    "target": nb_id,
                    "elev_diff": round(elev_i - elev_nb, 1),
                })

    edge_df = pd.DataFrame(edges).drop_duplicates(subset=["source", "target"])
    print(f"  Graph: {len(watersheds)} nodes, {len(edge_df)} directed edges")
    return edge_df


def save_outputs(watersheds: gpd.GeoDataFrame, edges: pd.DataFrame) -> None:
    """Save watershed polygons, node attributes, and edge list."""
    # Watershed polygons
    ws_path = GRAPH_DIR / "watersheds.geojson"
    watersheds.to_file(ws_path, driver="GeoJSON")
    print(f"  Watersheds → {ws_path}")

    # Node features CSV
    node_cols = [c for c in watersheds.columns if c != "geometry"]
    nodes_path = GRAPH_DIR / "graph_nodes.csv"
    watersheds[node_cols].to_csv(nodes_path, index=False)
    print(f"  Node features → {nodes_path}")

    # Edge list CSV
    edges_path = GRAPH_DIR / "graph_edges.csv"
    edges.to_csv(edges_path, index=False)
    print(f"  Edge list → {edges_path}")

    # Sparse adjacency matrix
    n = len(watersheds)
    id_to_idx = {wid: i for i, wid in enumerate(watersheds["watershed_id"])}
    rows = [id_to_idx[s] for s in edges.get("source", []) if s in id_to_idx]
    cols = [id_to_idx[t] for t in edges.get("target", []) if t in id_to_idx]
    data = np.ones(len(rows))
    from scipy.sparse import csr_matrix
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj_path = GRAPH_DIR / "adjacency.npz"
    from scipy.sparse import save_npz
    save_npz(str(adj_path), adj)
    print(f"  Adjacency matrix ({n}×{n}) → {adj_path}")

    # Summary
    summary = {
        "n_watersheds": n,
        "n_edges": len(edges),
        "mean_area_km2": round(float(watersheds["area_km2"].mean()), 1),
        "total_area_km2": round(float(watersheds["area_km2"].sum()), 0),
        "min_elevation": round(float(watersheds["mean_elevation"].min()), 0),
        "max_elevation": round(float(watersheds["mean_elevation"].max()), 0),
    }
    (GRAPH_DIR / "graph_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Summary: {summary}")


def main() -> None:
    print("=" * 60)
    print("Phase 2: Watershed delineation + GNN graph construction")
    print("=" * 60)

    dem_path = TERRAIN_DIR / "dem_hp.tif"
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM not found at {dem_path}. Run 04_preprocess_terrain.py first."
        )

    watersheds = delineate_watersheds(dem_path)
    edges      = build_graph(watersheds, dem_path)
    save_outputs(watersheds, edges)
    print("\nWatershed graph ready. Next: run 06_assemble_factors.py")


if __name__ == "__main__":
    main()
