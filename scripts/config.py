"""
Central configuration for all scripts.
All paths, parameters, and constants live here.
"""

from pathlib import Path

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Directory layout ─────────────────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
RESULTS         = ROOT / "results"
SCRIPTS         = ROOT / "scripts"

# Sub-directories (created on import)
DEM_DIR         = DATA_RAW / "dem"
RAINFALL_DIR    = DATA_RAW / "rainfall"
LULC_DIR        = DATA_RAW / "lulc"
SOIL_DIR        = DATA_RAW / "soil"
FLOOD_DIR       = DATA_RAW / "flood_inventory"
INFRA_DIR       = DATA_RAW / "infrastructure"
BOUNDARIES_DIR  = DATA_RAW / "boundaries"
GLACIAL_DIR     = DATA_RAW / "glacial_lakes"

TERRAIN_DIR     = DATA_PROCESSED / "terrain"
FACTORS_DIR     = DATA_PROCESSED / "conditioning_factors"
GRAPH_DIR       = DATA_PROCESSED / "watershed_graph"
INVENTORY_DIR   = DATA_PROCESSED / "flood_inventory"

MODELS_DIR      = RESULTS / "models"
MAPS_DIR        = RESULTS / "maps"
SHAP_DIR        = RESULTS / "shap"
VALIDATION_DIR  = RESULTS / "validation"
PAPER_DIR       = RESULTS / "paper_figures"

for d in [
    DEM_DIR, RAINFALL_DIR, LULC_DIR, SOIL_DIR, FLOOD_DIR,
    INFRA_DIR, BOUNDARIES_DIR, GLACIAL_DIR,
    TERRAIN_DIR, FACTORS_DIR, GRAPH_DIR, INVENTORY_DIR,
    MODELS_DIR, MAPS_DIR, SHAP_DIR, VALIDATION_DIR, PAPER_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Study area ────────────────────────────────────────────────────────────────
# Himachal Pradesh bounding box (WGS84)
HP_BBOX = {
    "xmin": 75.5,
    "ymin": 30.3,
    "xmax": 79.0,
    "ymax": 33.3,
}
HP_CRS      = "EPSG:4326"          # geographic
HP_CRS_UTM  = "EPSG:32643"         # UTM Zone 43N — projected for analysis
HP_PIXEL_M  = 30                   # target raster resolution in metres

# ── Conditioning factors ──────────────────────────────────────────────────────
# Final set after multicollinearity filtering (to be confirmed in Phase 2)
CANDIDATE_FACTORS = [
    "elevation",
    "slope",
    "aspect",
    "plan_curvature",
    "profile_curvature",
    "twi",              # Topographic Wetness Index
    "spi",              # Stream Power Index
    "tri",              # Terrain Ruggedness Index
    "drainage_density",
    "distance_to_river",
    "rainfall_mean_annual",
    "rainfall_extreme",  # 95th percentile daily rainfall
    "lulc",
    "ndvi",
    "soil_type",
    "lithology",
    "distance_to_glacial_lake",  # GLOF-specific
]

# ── ML parameters ────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
CV_FOLDS        = 5               # spatial block CV folds
TEST_YEAR       = 2023            # temporal hold-out (July–Sept 2023)
CONFORMAL_ALPHA = 0.10            # 90% prediction intervals

# RF / XGB / LGB shared
N_ESTIMATORS    = 500
MAX_DEPTH       = 6

# GNN
GNN_HIDDEN_DIM  = 64
GNN_LAYERS      = 3
GNN_EPOCHS      = 200
GNN_LR          = 1e-3

# ── Flood inventory ───────────────────────────────────────────────────────────
SAR_START_DATE  = "2018-01-01"
SAR_END_DATE    = "2024-12-31"
NON_FLOOD_BUFFER_M = 1000         # minimum buffer around flood points for non-flood sampling
FLOOD_NON_FLOOD_RATIO = 5         # non-flood : flood point ratio

# ── Paper metadata ────────────────────────────────────────────────────────────
PAPER_TITLE = (
    "Flash Flood Susceptibility Mapping in Himachal Pradesh Using "
    "Graph Neural Networks and Conformal Prediction: "
    "A Multi-Trigger, Uncertainty-Aware Framework"
)
TARGET_JOURNAL = "Natural Hazards and Earth System Sciences (NHESS)"
