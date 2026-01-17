#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
# Repo name
# -------------------------------------------------
REPO="openwater-shrinking-lake-monitor"

# Safety: don't overwrite
if [[ -d "$REPO" ]]; then
  echo "❌ Folder '$REPO' already exists. Exiting to avoid overwriting."
  exit 1
fi

mkdir -p "$REPO"
cd "$REPO"

# -------------------------------------------------
# Directories
# -------------------------------------------------


mkdir -p \
  configs \
  data/raw data/interim data/processed data/external \
  src/stac src/optical src/water src/viz \
  notebooks \
  outputs/figures outputs/maps outputs/reports \
  docs \
  scripts \
  tests/unit tests/integration




# -------------------------------------------------
# Placeholder files
# -------------------------------------------------
touch \
  README.md \
  LICENSE \
  .gitignore \
  environment.yml \
  configs/aoi_bbox.yaml \
  configs/time_range.yaml \
  configs/collections.yaml \
  configs/ndwi.yaml \
  src/__init__.py \
  src/stac/search_items.py \
  src/stac/load_odc.py \
  src/optical/sentinel2_search_load.py \
  src/optical/masking.py \
  src/water/indices.py \
  src/water/water_mask.py \
  src/water/area_timeseries.py \
  src/water/change_maps.py \
  src/viz/plot_timeseries.py \
  src/viz/map_overlays.py \
  notebooks/01_stac_search_and_load.ipynb \
  notebooks/02_ndwi_and_thresholding.ipynb \
  notebooks/03_area_time_series.ipynb \
  notebooks/04_map_total_shrinkage.ipynb \
  notebooks/05_surface_variation_over_time.ipynb \
  docs/methods.md \
  docs/results.md

# -------------------------------------------------
# Seed AOI bbox (example: Lake Mead area)
# -------------------------------------------------
cat > configs/aoi_bbox.yaml << 'YAML'
bbox:
  min_lon: -114.9
  min_lat: 35.7
  max_lon: -114.2
  max_lat: 36.4
crs: EPSG:4326
YAML

# -------------------------------------------------
# Seed time range
# -------------------------------------------------
cat > configs/time_range.yaml << 'YAML'
start_date: "2000-01-01"
end_date: "2024-12-31"
temporal_resolution: "annual"
season: "summer"
YAML

# -------------------------------------------------
# Seed imagery collections
# -------------------------------------------------
cat > configs/collections.yaml << 'YAML'
collections:
  - sentinel-2-l2a
  - landsat-c2-l2
cloud_cover_max: 20
YAML

# -------------------------------------------------
# Seed NDWI configuration
# -------------------------------------------------
cat > configs/ndwi.yaml << 'YAML'
index: NDWI
bands:
  green: green
  nir: nir
threshold:
  value: 0.0
  description: >
    Pixels with NDWI >= threshold are classified as water.
notes:
  - Threshold may vary by sensor, season, and turbidity.
  - Sensitivity analysis recommended.
YAML

# -------------------------------------------------
# Starter README
# -------------------------------------------------
cat > README.md << 'MD'
# OpenWater – Shrinking Lake Monitor (NDWI)

## Project goal
Quantify long-term surface area changes of a shrinking lake using
optical satellite imagery and the Normalized Difference Water Index (NDWI).

This project emphasizes:
- Spatiotemporal analysis
- Exploratory spatial data analysis (ESDA)
- Transparent, interpretable methods
- Reproducible open-source workflows

## Core questions
- How has lake surface area changed over time?
- Where is shrinkage occurring spatially?
- How sensitive are results to NDWI thresholds?

## Repo layout
- configs/   : AOI, time range, imagery collections, NDWI settings
- src/       : STAC access, NDWI computation, water masking, analysis logic
- notebooks/ : Narrative analysis workflow
- outputs/   : Exported figures, maps, and reports
- docs/      : Methods and interpretation notes
- data/      : Cached imagery and intermediates (gitignored)

## Methods overview
- STAC-based imagery search and loading
- NDWI computation
- Binary water masking
- Surface area time-series calculation
- Spatial mapping of cumulative change
MD

# -------------------------------------------------
# Copy bootstrap script into repo for reproducibility
# -------------------------------------------------
SCRIPT_NAME="$(basename "$0")"
cp "$0" "scripts/$SCRIPT_NAME"



# -------------------------------------------------
echo "✅ Repo created successfully:"
echo "   $(pwd)"
echo ""
echo "Next steps:"
echo "1. Open this folder in VS Code"
echo "2. Initialize git (git init)"
echo "3. Start with notebooks/01_stac_search_and_load.ipynb"
