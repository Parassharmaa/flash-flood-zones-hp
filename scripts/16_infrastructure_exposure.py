"""
Script 16: Infrastructure Exposure Analysis
Downloads OSM infrastructure data and overlays with susceptibility map.
Fills in the XX placeholders in paper/chapters/04_results.tex.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import geometry_mask
from shapely.geometry import box, shape

sys.path.insert(0, str(Path(__file__).parent))
from config import BOUNDARIES_DIR, INFRA_DIR, MAPS_DIR, RESULTS

# ── Output file ───────────────────────────────────────────────────────────────
OUTPUT_JSON = RESULTS / "infrastructure_exposure.json"

# ── Susceptibility thresholds ─────────────────────────────────────────────────
HIGH_THRESH   = 0.50
VHIGH_THRESH  = 0.70


def load_hp_boundary():
    """Load HP state boundary (WGS84)."""
    for name in ["hp_boundary.geojson", "hp_boundary.gpkg", "hp_district.geojson"]:
        p = BOUNDARIES_DIR / name
        if p.exists():
            return gpd.read_file(p).to_crs("EPSG:4326")
    raise FileNotFoundError(f"No HP boundary found in {BOUNDARIES_DIR}")


def overpass_query(query: str, timeout: int = 120) -> dict:
    """Run an Overpass API query and return the JSON response."""
    url = "https://overpass-api.de/api/interpreter"
    resp = requests.post(url, data={"data": query}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_osm_roads(bbox_str: str) -> gpd.GeoDataFrame:
    """Fetch national and state highways from OSM."""
    q = f"""
[out:json][timeout:120][bbox:{bbox_str}];
(
  way["highway"="primary"];
  way["highway"="trunk"];
  way["highway"="motorway"];
);
out body geom;
"""
    print("  Fetching roads from OSM …")
    data = overpass_query(q)
    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(nd["lon"], nd["lat"]) for nd in el["geometry"]]
        if len(coords) < 2:
            continue
        from shapely.geometry import LineString
        rows.append({
            "osm_id": el["id"],
            "highway": el.get("tags", {}).get("highway", ""),
            "ref": el.get("tags", {}).get("ref", ""),
            "name": el.get("tags", {}).get("name", ""),
            "geometry": LineString(coords),
        })
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "highway", "ref", "name", "geometry"],
                                crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def fetch_osm_bridges(bbox_str: str) -> gpd.GeoDataFrame:
    """Fetch bridges from OSM."""
    q = f"""
[out:json][timeout:120][bbox:{bbox_str}];
(
  way["bridge"="yes"];
  node["man_made"="bridge"];
);
out body geom;
"""
    print("  Fetching bridges from OSM …")
    data = overpass_query(q)
    rows = []
    for el in data.get("elements", []):
        geom_type = el.get("type")
        if geom_type == "node":
            from shapely.geometry import Point
            rows.append({
                "osm_id": el["id"],
                "type": "node",
                "geometry": Point(el["lon"], el["lat"]),
            })
        elif geom_type == "way" and "geometry" in el:
            coords = [(nd["lon"], nd["lat"]) for nd in el["geometry"]]
            if len(coords) >= 2:
                from shapely.geometry import LineString
                mid = coords[len(coords) // 2]
                from shapely.geometry import Point
                rows.append({
                    "osm_id": el["id"],
                    "type": "way",
                    "geometry": Point(mid),
                })
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "type", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def fetch_osm_hydro_projects(bbox_str: str) -> gpd.GeoDataFrame:
    """Fetch hydroelectric projects from OSM."""
    q = f"""
[out:json][timeout:120][bbox:{bbox_str}];
(
  node["power"="plant"]["plant:source"="hydro"];
  way["power"="plant"]["plant:source"="hydro"];
  node["waterway"="dam"];
  way["waterway"="dam"];
);
out body geom;
"""
    print("  Fetching hydroelectric projects from OSM …")
    data = overpass_query(q)
    rows = []
    for el in data.get("elements", []):
        geom_type = el.get("type")
        if geom_type == "node":
            from shapely.geometry import Point
            rows.append({
                "osm_id": el["id"],
                "subtype": el.get("tags", {}).get("waterway", el.get("tags", {}).get("power", "")),
                "name": el.get("tags", {}).get("name", ""),
                "geometry": Point(el["lon"], el["lat"]),
            })
        elif geom_type == "way" and "geometry" in el:
            coords = [(nd["lon"], nd["lat"]) for nd in el.get("geometry", [])]
            if coords:
                from shapely.geometry import Point
                mid = coords[len(coords) // 2]
                rows.append({
                    "osm_id": el["id"],
                    "subtype": el.get("tags", {}).get("waterway", el.get("tags", {}).get("power", "")),
                    "name": el.get("tags", {}).get("name", ""),
                    "geometry": Point(mid),
                })
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "subtype", "name", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def fetch_osm_settlements(bbox_str: str) -> gpd.GeoDataFrame:
    """Fetch villages and small settlements."""
    q = f"""
[out:json][timeout:120][bbox:{bbox_str}];
(
  node["place"~"village|hamlet|town"];
);
out body;
"""
    print("  Fetching settlements from OSM …")
    data = overpass_query(q)
    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "node":
            continue
        pop = el.get("tags", {}).get("population", "0")
        try:
            pop = int(pop)
        except (ValueError, TypeError):
            pop = 0
        from shapely.geometry import Point
        rows.append({
            "osm_id": el["id"],
            "place": el.get("tags", {}).get("place", ""),
            "name": el.get("tags", {}).get("name", ""),
            "population": pop,
            "geometry": Point(el["lon"], el["lat"]),
        })
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "place", "name", "population", "geometry"],
                                crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def fetch_osm_tourism(bbox_str: str) -> gpd.GeoDataFrame:
    """Fetch tourist accommodation (hotels, guesthouses, resorts)."""
    q = f"""
[out:json][timeout:120][bbox:{bbox_str}];
(
  node["tourism"~"hotel|guest_house|hostel|motel|resort"];
  way["tourism"~"hotel|guest_house|hostel|motel|resort"];
);
out body geom;
"""
    print("  Fetching tourist accommodation from OSM …")
    data = overpass_query(q)
    rows = []
    for el in data.get("elements", []):
        geom_type = el.get("type")
        if geom_type == "node":
            from shapely.geometry import Point
            rows.append({
                "osm_id": el["id"],
                "tourism": el.get("tags", {}).get("tourism", ""),
                "name": el.get("tags", {}).get("name", ""),
                "geometry": Point(el["lon"], el["lat"]),
            })
        elif geom_type == "way" and "geometry" in el:
            coords = [(nd["lon"], nd["lat"]) for nd in el.get("geometry", [])]
            if coords:
                from shapely.geometry import Point
                mid = coords[len(coords) // 2]
                rows.append({
                    "osm_id": el["id"],
                    "tourism": el.get("tags", {}).get("tourism", ""),
                    "name": el.get("tags", {}).get("name", ""),
                    "geometry": Point(mid),
                })
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "tourism", "name", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def sample_susceptibility_at_points(gdf: gpd.GeoDataFrame, susceptibility_path: Path) -> np.ndarray:
    """Sample susceptibility raster values at point geometries."""
    import rasterio
    from rasterio.transform import rowcol

    gdf_utm = gdf.to_crs("EPSG:32643")
    with rasterio.open(susceptibility_path) as src:
        transform = src.transform
        data = src.read(1)
        nodata = src.nodata or -9999

    vals = []
    for geom in gdf_utm.geometry:
        try:
            x, y = geom.centroid.x, geom.centroid.y
            row, col = rowcol(transform, x, y)
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                v = data[row, col]
                vals.append(float(v) if v != nodata else np.nan)
            else:
                vals.append(np.nan)
        except Exception:
            vals.append(np.nan)
    return np.array(vals)


def sample_susceptibility_along_lines(gdf: gpd.GeoDataFrame,
                                      susceptibility_path: Path,
                                      sample_spacing_m: float = 500.0) -> np.ndarray:
    """Sample susceptibility along line geometries at regular intervals."""
    import rasterio
    from rasterio.transform import rowcol
    from shapely.geometry import Point

    gdf_utm = gdf.to_crs("EPSG:32643")
    with rasterio.open(susceptibility_path) as src:
        transform = src.transform
        data = src.read(1)
        nodata = src.nodata or -9999

    max_vals = []
    for geom in gdf_utm.geometry:
        length = geom.length
        n_pts = max(2, int(length / sample_spacing_m))
        pts = [geom.interpolate(i / n_pts, normalized=True) for i in range(n_pts + 1)]
        line_vals = []
        for pt in pts:
            try:
                row, col = rowcol(transform, pt.x, pt.y)
                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                    v = data[row, col]
                    if v != nodata:
                        line_vals.append(float(v))
            except Exception:
                pass
        max_vals.append(max(line_vals) if line_vals else np.nan)
    return np.array(max_vals)


def road_km_in_high_susceptibility(roads_gdf: gpd.GeoDataFrame,
                                   susceptibility_path: Path,
                                   threshold: float = HIGH_THRESH) -> dict:
    """Calculate km of roads passing through high/very high susceptibility zones."""
    import rasterio
    from rasterio.transform import rowcol

    roads_utm = roads_gdf.to_crs("EPSG:32643")
    with rasterio.open(susceptibility_path) as src:
        transform = src.transform
        data = src.read(1)
        nodata = src.nodata or -9999

    sample_spacing_m = 200.0
    total_km_high = 0.0
    total_km_vhigh = 0.0

    # NH references
    nh_refs = {}

    for _, row_data in roads_utm.iterrows():
        geom = row_data.geometry
        length = geom.length
        ref = str(row_data.get("ref", ""))

        n_pts = max(2, int(length / sample_spacing_m))
        high_count = 0
        vhigh_count = 0
        valid_count = 0

        for i in range(n_pts + 1):
            pt = geom.interpolate(i / n_pts, normalized=True)
            try:
                r, c = rowcol(transform, pt.x, pt.y)
                if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                    v = data[r, c]
                    if v != nodata and v >= 0:
                        valid_count += 1
                        if v >= HIGH_THRESH:
                            high_count += 1
                        if v >= VHIGH_THRESH:
                            vhigh_count += 1
            except Exception:
                pass

        if valid_count > 0:
            seg_km = length / 1000.0
            frac_high = high_count / valid_count
            frac_vhigh = vhigh_count / valid_count
            total_km_high += seg_km * frac_high
            total_km_vhigh += seg_km * frac_vhigh

            # Track specific NH routes
            for nh in ["NH-3", "NH-21", "NH-5", "NH-154", "NH-503"]:
                if nh in ref or nh.replace("-", "") in ref:
                    if nh not in nh_refs:
                        nh_refs[nh] = 0.0
                    nh_refs[nh] += seg_km * frac_high

    return {
        "total_km_high_vhigh": round(total_km_high),
        "total_km_vhigh": round(total_km_vhigh),
        "nh_breakdown": {k: round(v) for k, v in sorted(nh_refs.items(), key=lambda x: -x[1])},
    }


def main():
    print("=== Infrastructure Exposure Analysis ===\n")

    # Load susceptibility map
    susceptibility_path = MAPS_DIR / "susceptibility_point_estimate.tif"
    if not susceptibility_path.exists():
        raise FileNotFoundError(f"Susceptibility map not found: {susceptibility_path}")
    print(f"Using susceptibility map: {susceptibility_path}")

    # Load HP boundary for bounding box
    try:
        hp_boundary = load_hp_boundary()
        bounds = hp_boundary.total_bounds  # (minx, miny, maxx, maxy) in WGS84
    except FileNotFoundError:
        bounds = [75.5, 30.3, 79.0, 33.3]
    bbox_str = f"{bounds[1]:.4f},{bounds[0]:.4f},{bounds[3]:.4f},{bounds[2]:.4f}"
    print(f"HP bounding box (S,W,N,E): {bbox_str}\n")

    results = {}

    # ── 1. Roads ─────────────────────────────────────────────────────────────
    roads_cache = INFRA_DIR / "osm_roads.geojson"
    if roads_cache.exists():
        print("  Loading roads from cache …")
        roads_gdf = gpd.read_file(roads_cache)
    else:
        roads_gdf = fetch_osm_roads(bbox_str)
        if not roads_gdf.empty:
            roads_gdf.to_file(roads_cache, driver="GeoJSON")
    print(f"  Roads fetched: {len(roads_gdf)} segments")

    if not roads_gdf.empty:
        road_stats = road_km_in_high_susceptibility(roads_gdf, susceptibility_path)
        results["roads"] = road_stats
        print(f"  Roads in High+VHigh zones: {road_stats['total_km_high_vhigh']} km")
        print(f"  NH breakdown: {road_stats['nh_breakdown']}")
    else:
        results["roads"] = {"total_km_high_vhigh": "N/A", "nh_breakdown": {}}

    # ── 2. Bridges ───────────────────────────────────────────────────────────
    bridges_cache = INFRA_DIR / "osm_bridges.geojson"
    if bridges_cache.exists():
        print("\n  Loading bridges from cache …")
        bridges_gdf = gpd.read_file(bridges_cache)
    else:
        bridges_gdf = fetch_osm_bridges(bbox_str)
        if not bridges_gdf.empty:
            bridges_gdf.to_file(bridges_cache, driver="GeoJSON")
    print(f"  Bridges fetched: {len(bridges_gdf)}")

    if not bridges_gdf.empty:
        # Keep only point geometries for sampling
        point_mask = bridges_gdf.geometry.geom_type == "Point"
        bridges_pts = bridges_gdf[point_mask].copy()
        susc_vals = sample_susceptibility_at_points(bridges_pts, susceptibility_path)
        n_high = int(np.nansum(susc_vals >= HIGH_THRESH))
        n_vhigh = int(np.nansum(susc_vals >= VHIGH_THRESH))
        results["bridges"] = {"n_high_vhigh": n_high, "n_vhigh": n_vhigh, "total": len(bridges_pts)}
        print(f"  Bridges in High+VHigh zones: {n_high} / {len(bridges_pts)}")
    else:
        results["bridges"] = {"n_high_vhigh": "N/A", "total": "N/A"}

    # ── 3. Hydroelectric projects ─────────────────────────────────────────────
    hydro_cache = INFRA_DIR / "osm_hydro.geojson"
    if hydro_cache.exists():
        print("\n  Loading hydro from cache …")
        hydro_gdf = gpd.read_file(hydro_cache)
    else:
        hydro_gdf = fetch_osm_hydro_projects(bbox_str)
        if not hydro_gdf.empty:
            hydro_gdf.to_file(hydro_cache, driver="GeoJSON")
    print(f"  Hydro facilities fetched: {len(hydro_gdf)}")

    if not hydro_gdf.empty:
        susc_vals = sample_susceptibility_at_points(hydro_gdf, susceptibility_path)
        n_high = int(np.nansum(susc_vals >= HIGH_THRESH))
        results["hydro"] = {"n_high": n_high, "total": len(hydro_gdf)}
        print(f"  Hydro in High+VHigh zones: {n_high} / {len(hydro_gdf)}")
    else:
        results["hydro"] = {"n_high": "N/A", "total": "N/A"}

    # ── 4. Settlements ────────────────────────────────────────────────────────
    settle_cache = INFRA_DIR / "osm_settlements.geojson"
    if settle_cache.exists():
        print("\n  Loading settlements from cache …")
        settle_gdf = gpd.read_file(settle_cache)
    else:
        settle_gdf = fetch_osm_settlements(bbox_str)
        if not settle_gdf.empty:
            settle_gdf.to_file(settle_cache, driver="GeoJSON")
    print(f"  Settlements fetched: {len(settle_gdf)}")

    if not settle_gdf.empty:
        susc_vals = sample_susceptibility_at_points(settle_gdf, susceptibility_path)
        mask_vhigh = susc_vals >= VHIGH_THRESH
        n_vhigh = int(np.nansum(mask_vhigh))
        total_pop = int(settle_gdf.loc[mask_vhigh, "population"].sum())
        results["settlements"] = {
            "n_vhigh": n_vhigh,
            "total": len(settle_gdf),
            "pop_in_vhigh": total_pop,
        }
        print(f"  Villages in VHigh zones: {n_vhigh} / {len(settle_gdf)}, pop={total_pop:,}")
    else:
        results["settlements"] = {"n_vhigh": "N/A", "total": "N/A"}

    # ── 5. Tourist accommodation ──────────────────────────────────────────────
    tourism_cache = INFRA_DIR / "osm_tourism.geojson"
    if tourism_cache.exists():
        print("\n  Loading tourism from cache …")
        tourism_gdf = gpd.read_file(tourism_cache)
    else:
        tourism_gdf = fetch_osm_tourism(bbox_str)
        if not tourism_gdf.empty:
            tourism_gdf.to_file(tourism_cache, driver="GeoJSON")
    print(f"  Tourist accommodation fetched: {len(tourism_gdf)}")

    if not tourism_gdf.empty:
        susc_vals = sample_susceptibility_at_points(tourism_gdf, susceptibility_path)
        n_high = int(np.nansum(susc_vals >= HIGH_THRESH))
        results["tourism"] = {"n_high_vhigh": n_high, "total": len(tourism_gdf)}
        print(f"  Tourism in High+VHigh zones: {n_high} / {len(tourism_gdf)}")
    else:
        results["tourism"] = {"n_high_vhigh": "N/A", "total": "N/A"}

    # ── Save results ──────────────────────────────────────────────────────────
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved to {OUTPUT_JSON}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== Summary for paper ===")
    r = results.get("roads", {})
    b = results.get("bridges", {})
    h = results.get("hydro", {})
    s = results.get("settlements", {})
    t = results.get("tourism", {})

    print(f"Roads (High+VHigh zones): {r.get('total_km_high_vhigh', 'N/A')} km")
    print(f"Bridges (High+VHigh zones): {b.get('n_high_vhigh', 'N/A')}")
    print(f"Hydro projects (High zones): {h.get('n_high', 'N/A')}")
    print(f"Villages in VHigh zones: {s.get('n_vhigh', 'N/A')} (pop ~{s.get('pop_in_vhigh', 'N/A'):,})")
    print(f"Tourist units (High+VHigh): {t.get('n_high_vhigh', 'N/A')}")

    return results


if __name__ == "__main__":
    main()
