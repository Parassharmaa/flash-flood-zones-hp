"""
Master runner — executes the full pipeline in order.
Safe to re-run: each script checks for existing outputs and skips.

Usage: uv run python scripts/13_run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

PIPELINE = [
    ("01_download_boundaries.py",  "Download HP boundaries + infrastructure (OSM)"),
    ("02_download_rasters.py",     "Download DEM, LULC, Soil rasters"),
    ("03_gee_sar_inventory.py",    "Generate GEE script for SAR flood inventory"),
    ("04_preprocess_terrain.py",   "Compute terrain factors from DEM"),
    ("05_watershed_delineation.py","Delineate watersheds + build GNN graph"),
    ("06_assemble_factors.py",     "Assemble factor stack + multicollinearity check"),
    ("07_build_flood_inventory.py","Build flood inventory (SAR + HiFlo-DAT)"),
    ("08_train_baseline_models.py","Train RF, XGBoost, LightGBM, Stacking"),
    ("09_train_gnn.py",            "Train Graph Neural Network (GraphSAGE)"),
    ("10_conformal_prediction.py", "Conformal prediction + uncertainty maps"),
    ("11_shap_analysis.py",        "SHAP global + spatial + district analysis"),
    ("12_generate_paper_figures.py","Generate all publication figures"),
]


def run_script(script_name: str, description: str) -> bool:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"  {description}")
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED: {script_name} (exit code {result.returncode})")
        return False
    print(f"\n  Done in {elapsed:.1f}s")
    return True


def main() -> None:
    print("Flash Flood HP — Full Pipeline")
    print("=" * 60)

    # Check if GEE script needs manual execution
    gee_note = Path(__file__).parent.parent / "data" / "raw" / "flood_inventory" / "sar"
    if not gee_note.exists() or not list(gee_note.glob("*.tif")):
        print("\nNOTE: SAR flood inventory not yet available.")
        print("  After running this pipeline:")
        print("  1. Open https://code.earthengine.google.com")
        print("  2. Run scripts/03_gee_script.js")
        print("  3. Download TIFs to data/raw/flood_inventory/sar/")
        print("  4. Re-run this script for real results.\n")
        print("  Continuing with placeholder data for pipeline validation...\n")

    failed = []
    for script, description in PIPELINE:
        success = run_script(script, description)
        if not success:
            failed.append(script)
            print(f"  Continuing despite failure in {script}...")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    if failed:
        print(f"\nFailed scripts ({len(failed)}):")
        for s in failed:
            print(f"  - {s}")
    else:
        print("All scripts succeeded.")
    print(f"\nOutputs:")
    print(f"  results/models/       — trained models")
    print(f"  results/maps/         — susceptibility maps")
    print(f"  results/shap/         — SHAP analysis")
    print(f"  results/validation/   — CV results + coverage analysis")
    print(f"  results/paper_figures/— publication figures")


if __name__ == "__main__":
    main()
