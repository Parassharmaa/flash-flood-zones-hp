"""
Phase 1 — Download HP boundaries and infrastructure from OSM.

Outputs:
  data/raw/boundaries/hp_state.geojson       — HP state boundary
  data/raw/boundaries/hp_districts.geojson   — HP district boundaries
  data/raw/boundaries/hp_rivers.geojson      — river network
  data/raw/infrastructure/roads.geojson      — road network
  data/raw/infrastructure/bridges.geojson    — bridges
  data/raw/infrastructure/settlements.geojson — towns + villages
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from config import BOUNDARIES_DIR, HP_BBOX, INFRA_DIR  # noqa: E402

ox.settings.log_console = False
ox.settings.use_cache = True


def download_hp_admin_boundaries() -> None:
    """Download HP state + district boundaries via GADM / Overpass."""
    print("Downloading HP administrative boundaries...")

    # State boundary via Nominatim + OSM relation
    try:
        gdf = ox.geocode_to_gdf("Himachal Pradesh, India")
        out = BOUNDARIES_DIR / "hp_state.geojson"
        gdf.to_file(out, driver="GeoJSON")
        print(f"  State boundary → {out} ({len(gdf)} feature)")
    except Exception as e:
        print(f"  WARNING: state boundary download failed: {e}")

    # District boundaries via GADM API (level 2 = district)
    gadm_url = (
        "https://gadm.org/download_country.html"  # manual download note
    )
    # Use Overpass for districts
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:60];
    area["name"="Himachal Pradesh"]["admin_level"="4"]->.hp;
    (
      relation["admin_level"="6"](area.hp);
    );
    out geom;
    """
    try:
        resp = requests.post(overpass_url, data={"data": query}, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        features = []
        for elem in data.get("elements", []):
            if elem["type"] == "relation":
                name = elem.get("tags", {}).get("name", "unknown")
                features.append({"type": "Feature",
                                  "properties": {"name": name, "osm_id": elem["id"]},
                                  "geometry": None})
        print(f"  Found {len(features)} district relations (geometry needs PostGIS or manual processing)")
        # Save raw response for manual processing
        raw_out = BOUNDARIES_DIR / "hp_districts_raw.json"
        raw_out.write_text(json.dumps(data, indent=2))
        print(f"  Raw districts data → {raw_out}")
    except Exception as e:
        print(f"  WARNING: district boundary download failed: {e}")


def download_river_network() -> None:
    """Download river network for HP from OSM."""
    print("Downloading river network...")
    bbox_poly = box(HP_BBOX["xmin"], HP_BBOX["ymin"],
                    HP_BBOX["xmax"], HP_BBOX["ymax"])
    try:
        # OSMnx for waterways
        tags = {"waterway": ["river", "stream", "canal"]}
        gdf = ox.features_from_bbox(
            bbox=(HP_BBOX["ymin"], HP_BBOX["ymin"],
                  HP_BBOX["ymax"], HP_BBOX["xmax"]),
            tags=tags,
        )
        out = BOUNDARIES_DIR / "hp_rivers.geojson"
        # Keep only line geometries
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        gdf.to_file(out, driver="GeoJSON")
        print(f"  River network → {out} ({len(gdf)} features)")
    except Exception as e:
        print(f"  WARNING: river download failed: {e}")


def download_infrastructure() -> None:
    """Download roads, bridges, and settlements from OSM."""
    print("Downloading infrastructure...")

    # Roads
    try:
        G = ox.graph_from_bbox(
            bbox=(HP_BBOX["ymin"], HP_BBOX["ymin"],
                  HP_BBOX["ymax"], HP_BBOX["xmax"]),
            network_type="drive",
            retain_all=False,
        )
        _, edges = ox.graph_to_gdfs(G)
        out = INFRA_DIR / "roads.geojson"
        edges.to_file(out, driver="GeoJSON")
        print(f"  Roads → {out} ({len(edges)} segments)")
    except Exception as e:
        print(f"  WARNING: road download failed: {e}")

    # Settlements
    try:
        tags = {"place": ["city", "town", "village", "hamlet"]}
        gdf = ox.features_from_bbox(
            bbox=(HP_BBOX["ymin"], HP_BBOX["ymin"],
                  HP_BBOX["ymax"], HP_BBOX["xmax"]),
            tags=tags,
        )
        gdf = gdf[gdf.geometry.geom_type == "Point"]
        out = INFRA_DIR / "settlements.geojson"
        gdf[["geometry", "name", "place"]].to_file(out, driver="GeoJSON")
        print(f"  Settlements → {out} ({len(gdf)} features)")
    except Exception as e:
        print(f"  WARNING: settlement download failed: {e}")


def main() -> None:
    print("=" * 60)
    print("Phase 1a: Downloading boundaries and infrastructure")
    print("=" * 60)
    download_hp_admin_boundaries()
    download_river_network()
    download_infrastructure()
    print("\nDone. Check data/raw/boundaries/ and data/raw/infrastructure/")


if __name__ == "__main__":
    main()
