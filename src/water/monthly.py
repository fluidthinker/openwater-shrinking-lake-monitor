from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import yaml

from src.stac.search_items import search_month_from_config
from src.stac.load_odc import (
    clip_to_aoi,
    load_s2_items_odc,
    read_aoi_geojson,
    scl_cloud_mask,
)

# Delegate “domain logic” to water modules (avoids duplication)
from src.water.indices import ndwi_index
from src.water.water_mask import (
    compute_valid_metrics,
    compute_water_area_km2,
    water_mask_from_ndwi,
)


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
        Projected CRS for loading/area calculations (e.g., "EPSG:3857").
    chunks : dict, optional
        Chunk sizes for x/y.
    plot_defaults : bool
        Default plotting preference from config file (used by scripts).
    """

    cloud_cover_max: int = 80
    water_thresh: float = 0.10
    resolution_m: float = 10.0
    load_crs: str = "EPSG:3857"
    chunks: Optional[Dict[str, int]] = None
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
) -> Dict[str, Any]:
    """Compute NDWI-based water area and QA metrics for a single month.

    This function performs computation only (no plotting, no saving). It returns both
    summary metrics and optional layers for downstream plotting.

    Parameters
    ----------
    year : int
        Year to process.
    month : int
        Month to process (1–12).
    aoi_geojson : Path
        AOI polygon GeoJSON path.
    cfg : NdwiConfig
        NDWI configuration parameters.

    Returns
    -------
    dict
        Dictionary containing:
        - year, month, n_items
        - water_area_km2
        - median_valid_fraction
        - valid_fraction_any
        And layers for optional QA/plots:
        - aoi (GeoDataFrame) or None
        - valid_fraction_map (DataArray) or None
        - ndwi_med (DataArray) or None
        - water (DataArray) or None
    """
    # 1) STAC search
    items = search_month_from_config(
        year, month, cloud_cover_max=cfg.cloud_cover_max, limit=500
    )
    n_items = len(items)

    if n_items == 0:
        return {
            "year": year,
            "month": month,
            "n_items": 0,
            "water_area_km2": np.nan,
            "median_valid_fraction": np.nan,
            "valid_fraction_any": np.nan,
            "aoi": None,
            "valid_fraction_map": None,
            "ndwi_med": None,
            "water": None,
        }

    # 2) Load
    ds = load_s2_items_odc(
        items,
        bands=["B03", "B08", "SCL"],
        crs=cfg.load_crs,
        resolution=cfg.resolution_m,
        chunks=cfg.chunks or {"x": 2048, "y": 2048},
    )

    # 3) Clip
    aoi = read_aoi_geojson(aoi_geojson)
    ds_clip = clip_to_aoi(ds, aoi)

    # 4) Valid/clear mask + metrics
    valid = scl_cloud_mask(ds_clip["SCL"]).fillna(False)

    median_valid_fraction, valid_fraction_any, valid_fraction_map = compute_valid_metrics(
        valid=valid,
        scl=ds_clip["SCL"],
    )

    # 5) NDWI median composite (cloud-masked)
    green = ds_clip["B03"].astype("float32")
    nir = ds_clip["B08"].astype("float32")

    ndwi = ndwi_index(green=green, nir=nir)
    ndwi_clear = ndwi.where(valid)
    ndwi_med = ndwi_clear.median(dim="time", skipna=True)

    # 6) Water mask + area
    water = water_mask_from_ndwi(ndwi_med, cfg.water_thresh)
    water_area_km2 = compute_water_area_km2(water, ds_clip)

    return {
        "year": year,
        "month": month,
        "n_items": int(n_items),
        "water_area_km2": float(water_area_km2),
        "median_valid_fraction": float(median_valid_fraction),
        "valid_fraction_any": float(valid_fraction_any),
        "aoi": aoi,
        "valid_fraction_map": valid_fraction_map,
        "ndwi_med": ndwi_med,
        "water": water,
    }
