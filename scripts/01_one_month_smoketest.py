# %% [markdown]
# # 01 — One-month smoke test (clean)
#
# Essential pipeline:
# 1) STAC search for one month of Sentinel-2 L2A (Planetary Computer)
# 2) Load B03 (green), B08 (NIR), SCL (scene classification) via ODC-STAC
# 3) Clip to AOI polygon
# 4) Build clear/valid mask from SCL and compute coverage metrics
# 5) Compute NDWI monthly median composite (cloud-masked)
# 6) Threshold NDWI -> water mask and compute water area (km²)
#
# This version:
# - Removes heavy diagnostics
# - Uses configs/ndwi.yaml for tunables
# - Returns monthly metrics as a dict (ready for looping)
# - Adds a PLOT toggle

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

# Ensure repo root is on sys.path (Option A)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stac.search_items import search_month_from_config
from src.stac.load_odc import (
    clip_to_aoi,
    load_s2_items_odc,
    read_aoi_geojson,
    scl_cloud_mask,
)

# %%
# -----------------------------
# User knobs
# -----------------------------
YEAR = 2020
MONTH = 7  # 1-12

# Turn plots on/off for batch readiness
PLOT = True

# Paths
AOI_GEOJSON = REPO_ROOT / "data" / "external" / "aoi.geojson"
CFG_NDWI = REPO_ROOT / "configs" / "ndwi.yaml"


# %%
@dataclass(frozen=True)
class NdwiConfig:
    """Configuration for NDWI monthly processing.

    Parameters
    ----------
    cloud_cover_max : int
        STAC filter for eo:cloud_cover (lenient recommended).
    water_thresh : float
        NDWI threshold; water = NDWI > water_thresh.
    resolution_m : float
        Target pixel resolution in meters.
    load_crs : str
        Projected CRS for loading/area calculations (e.g., EPSG:3857).
    chunks : dict
        Chunk sizes for x/y.
    plot_defaults : bool
        Default plotting preference from config file.
    """

    cloud_cover_max: int = 80
    water_thresh: float = 0.10
    resolution_m: float = 10.0
    load_crs: str = "EPSG:3857"
    chunks: Dict[str, int] | None = None
    plot_defaults: bool = True


def read_ndwi_config(path: Path) -> NdwiConfig:
    """Read NDWI configuration from YAML.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    NdwiConfig
        Parsed configuration with defaults filled in if file is missing.
    """
    if not path.exists():
        # Safe defaults if user hasn't created the config yet
        return NdwiConfig(chunks={"x": 2048, "y": 2048})

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    chunks = raw.get("chunks", {"x": 2048, "y": 2048})
    return NdwiConfig(
        cloud_cover_max=int(raw.get("cloud_cover_max", 80)),
        water_thresh=float(raw.get("water_thresh", 0.10)),
        resolution_m=float(raw.get("resolution_m", 10.0)),
        load_crs=str(raw.get("load_crs", "EPSG:3857")),
        chunks={"x": int(chunks.get("x", 2048)), "y": int(chunks.get("y", 2048))},
        plot_defaults=bool(raw.get("plot_defaults", True)),
    )


def compute_month_metrics(
    year: int,
    month: int,
    aoi_geojson: Path,
    cfg: NdwiConfig,
    plot: bool = False,
) -> Dict[str, Any]:
    """Run the one-month Sentinel-2 NDWI workflow and return summary metrics.

    Parameters
    ----------
    year : int
        Year to process (e.g., 2020).
    month : int
        Month to process (1-12).
    aoi_geojson : Path
        AOI polygon GeoJSON path.
    cfg : NdwiConfig
        NDWI configuration parameters.
    plot : bool, optional
        If True, renders plots (valid fraction, NDWI composite, water mask).

    Returns
    -------
    dict
        Dictionary containing:
        - year, month
        - water_area_km2
        - median_valid_fraction
        - valid_fraction_any
        - n_items
    """
    # 1) STAC search (bbox comes from configs/aoi_bbox.yaml inside search_month_from_config)
    items = search_month_from_config(
        year, month, cloud_cover_max=cfg.cloud_cover_max, limit=500
    )
    n_items = len(items)
    if n_items == 0:
        return {
            "year": year,
            "month": month,
            "water_area_km2": np.nan,
            "median_valid_fraction": np.nan,
            "valid_fraction_any": np.nan,
            "n_items": 0,
        }

    # 2) Load data
    ds = load_s2_items_odc(
        items,
        bands=["B03", "B08", "SCL"],
        crs=cfg.load_crs,
        resolution=cfg.resolution_m,
        chunks=cfg.chunks or {"x": 2048, "y": 2048},
    )

    # 3) Clip to AOI
    aoi = read_aoi_geojson(aoi_geojson)
    ds_clip = clip_to_aoi(ds, aoi)

    # 4) Valid/clear mask + metrics
    valid = scl_cloud_mask(ds_clip["SCL"]).fillna(False)

    valid_count = valid.sum(dim="time")
    total_count = ds_clip["SCL"].notnull().sum(dim="time")

    valid_fraction_map = xr.where(
        total_count > 0, valid_count / total_count, np.nan
    ).astype("float32")

    has_data = total_count > 0
    valid_any = valid.any(dim="time").where(has_data)

    stats_mask = has_data & (valid_count > 0)
    vf = valid_fraction_map.where(stats_mask)

    # Compute scalars (robust to dask)
    vf_np = vf.data.compute()
    median_valid_fraction = float(np.nanmedian(vf_np))

    valid_any_np = valid_any.data.compute()
    valid_fraction_any = float(np.nanmean(valid_any_np))

    if plot:
        plt.figure()
        valid_fraction_map.plot(robust=True)
        plt.title(f"Valid (clear) observation fraction — {year}-{month:02d}")
        plt.show()

    # 5) NDWI composite
    green = ds_clip["B03"].astype("float32")
    nir = ds_clip["B08"].astype("float32")
    
    # NDWI involves dividing by (green + nir).
    # In rare pixels, this denominator can be zero or extremely small
    # (e.g., due to missing data or very low signal), which would
    # produce infinite or unstable NDWI values.
    #
    # To avoid propagating numerical artifacts into the median composite
    # and water mask, we explicitly skip those pixels and set NDWI to NaN.
    den = green + nir
    ndwi = xr.where(den != 0, (green - nir) / den, np.nan)


    ndwi_clear = ndwi.where(valid)
    ndwi_med = ndwi_clear.median(dim="time", skipna=True)

    if plot:
        plt.figure()
        ndwi_med.plot(robust=True)
        plt.title(f"NDWI median composite — {year}-{month:02d}")
        plt.show()

    # 6) Water area
    water = ndwi_med > cfg.water_thresh

    res_x, res_y = ds_clip.rio.resolution()
    pixel_area_m2 = abs(res_x * res_y)

    water_area_m2 = float(water.sum().values) * pixel_area_m2
    water_area_km2 = water_area_m2 / 1e6

    if plot:
        plt.figure()
        water.plot()
        plt.title(f"Water mask — {year}-{month:02d} (NDWI>{cfg.water_thresh})")
        plt.show()

    return {
        "year": year,
        "month": month,
        "water_area_km2": float(water_area_km2),
        "median_valid_fraction": float(median_valid_fraction),
        "valid_fraction_any": float(valid_fraction_any),
        "n_items": int(n_items),
    }


# %%
# -----------------------------
# Run
# -----------------------------
cfg = read_ndwi_config(CFG_NDWI)

# If user didn't set PLOT explicitly, you could fallback to cfg.plot_defaults.
plot_effective = bool(PLOT)

print(f"Repo root: {REPO_ROOT}")
print(f"AOI: {AOI_GEOJSON}")
print(f"Config: {CFG_NDWI}  (exists={CFG_NDWI.exists()})")
print(f"Processing: {YEAR}-{MONTH:02d}")
print(
    f"Settings: cloud_cover_max={cfg.cloud_cover_max}, "
    f"water_thresh={cfg.water_thresh}, "
    f"resolution_m={cfg.resolution_m}, crs={cfg.load_crs}"
)

metrics = compute_month_metrics(
    YEAR, MONTH, AOI_GEOJSON, cfg=cfg, plot=plot_effective
)

print("\n--- Monthly metrics ---")
for k in ["year", "month", "n_items", "water_area_km2", "median_valid_fraction", "valid_fraction_any"]:
    print(f"{k:>22}: {metrics.get(k)}")
