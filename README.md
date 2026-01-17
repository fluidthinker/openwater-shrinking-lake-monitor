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

## Quickstart (create this repo structure)
```bash
bash scripts/bootstrap_repo.sh

Run the script from the directory where you want the repo created (not inside an existing repo).


