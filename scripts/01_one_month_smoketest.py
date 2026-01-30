# %% [markdown]
# # 01 — One-month smoke test (runner)
#
# This script is a **validation and QA runner** for the OpenWater shrinking-lake workflow.
# It executes the full NDWI-based water-area pipeline for a *single month* and is intended
# to answer one question:
#
# > “Does the end-to-end workflow behave correctly for a representative month?”
#
# ---
#
# ## What this script does
#
# For a specified (YEAR, MONTH), the script:
#
# 1) Searches Planetary Computer STAC for Sentinel-2 L2A scenes
# 2) Loads Green (B03), NIR (B08), and Scene Classification (SCL) bands
# 3) Clips imagery to the AOI polygon
# 4) Applies cloud/quality masking using the SCL
# 5) Computes per-pixel observation coverage metrics:
#    - median_valid_fraction
#    - valid_fraction_any
# 6) Computes a cloud-masked NDWI median composite
# 7) Thresholds NDWI to derive a water mask
# 8) Computes total water surface area (km²)
# 9) Optionally generates and saves QA figures:
#    - valid observation fraction map
#    - NDWI median composite
#    - water mask overlaid with AOI boundary
#
# ---
#
# ## What this script does *not* do
#
# - It does NOT loop over multiple months
# - It does NOT write time-series CSVs
# - It does NOT perform annual aggregation
#
# Those steps are handled by:
#   - `02_monthly_loop.py` (multi-month automation)
#   - `area_timeseries.py` (aggregation and plotting)
#
# ---
#
# ## Architecture note
#
# This file is intentionally **thin**.
# All scientific and geospatial logic lives in reusable modules:
#
# - `src/water/monthly.py`   → monthly workflow orchestration
# - `src/water/indices.py`   → NDWI (and future index) calculations
# - `src/water/water_mask.py`→ masking, QA metrics, area calculations
# - `src/stac/*`             → STAC search, loading, clipping
#
# This separation allows the same validated logic to be reused safely
# in batch processing and time-series analysis.
#
# ---
#
# ## When to run this script
#
# - When developing or refactoring the pipeline
# - When tuning NDWI thresholds
# - When validating behavior across contrasting seasons (e.g., Jan vs Jul)
# - Before launching long monthly loops
#
# If this script produces reasonable metrics and visually correct overlays,
# the pipeline is considered ready for automation.


# %%
from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path (Option A)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.water.monthly import (
    read_ndwi_config,
    compute_month_metrics,
    plot_month_diagnostics,
)

# %% [markdown]
# ## User knobs

# %%
YEAR = 2020
MONTH = 7  # 1-12

# Turn plots on/off (plots also save PNGs into outputs/figures when enabled)
PLOT = True

# Paths
AOI_GEOJSON = REPO_ROOT / "data" / "external" / "aoi.geojson"
CFG_NDWI = REPO_ROOT / "configs" / "ndwi.yaml"
OUT_FIGURES = REPO_ROOT / "outputs" / "figures"

# %% [markdown]
# ## Run

# %%
cfg = read_ndwi_config(CFG_NDWI)

print(f"Repo root: {REPO_ROOT}")
print(f"AOI: {AOI_GEOJSON}")
print(f"Config: {CFG_NDWI}  (exists={CFG_NDWI.exists()})")
print(f"Processing: {YEAR}-{MONTH:02d}")
print(
    f"Settings: cloud_cover_max={cfg.cloud_cover_max}, "
    f"water_thresh={cfg.water_thresh}, "
    f"resolution_m={cfg.resolution_m}, "
    f"crs={cfg.load_crs}"
)

metrics = compute_month_metrics(YEAR, MONTH, AOI_GEOJSON, cfg=cfg)

print("\n--- Monthly metrics ---")
for k in [
    "year",
    "month",
    "n_items",
    "water_area_km2",
    "median_valid_fraction",
    "valid_fraction_any",
]:
    print(f"{k:>22}: {metrics.get(k)}")

if PLOT:
    plot_month_diagnostics(metrics, cfg, out_dir=OUT_FIGURES)

# %%
