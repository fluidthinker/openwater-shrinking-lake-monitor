from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
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

    Parameters:
        cloud_cover_max: STAC filter for eo:cloud_cover (lenient recommended).
        water_thresh: NDWI threshold; water = NDWI > water_thresh.
        resolution_m: Target pixel resolution in meters.
        load_crs: Projected CRS for loading/area calculations (e.g., "EPSG:3857").
        chunks: Chunk sizes for x/y.
        plot_defaults: Default plotting preference from config file.
    """

    cloud_cover_max: int = 80
    water_thresh: float = 0.10
    resolution_m: float = 10.0
    load_crs: str = "EPSG:3857"
    chunks: Optional[Dict[str, int]] = None
    plot_defaults: bool = True


def read_ndwi_config(path: Path) -> NdwiConfig:
    """Read NDWI configuration from YAML.

    Parameters:
        path: Path to the YAML configuration file.

    Returns:
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


def make_watermask_overlay_figure(
    water: xr.DataArray,
    aoi_gdf,
    crs: str,
    title: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure showing water mask with AOI boundary overlay.

    Parameters:
        water: Boolean water mask in the same CRS as the raster grid.
        aoi_gdf: AOI GeoDataFrame (any CRS); will be reprojected to `crs`.
        crs: Target CRS string (e.g., "EPSG:3857").
        title: Plot title.

    Returns:
        (fig, ax) matplotlib figure and axes.
    """
    aoi_proj = aoi_gdf.to_crs(crs)

    fig, ax = plt.subplots(figsize=(8, 8))
    water.plot(ax=ax, add_colorbar=False)
    aoi_proj.boundary.plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_axis_off()
    return fig, ax


def save_figure_png(fig: plt.Figure, out_png: Path, dpi: int = 200) -> None:
    """Save a matplotlib figure to a PNG path.

    Parameters:
        fig: Matplotlib Figure.
        out_png: Output PNG path.
        dpi: Output resolution.

    Returns:
        None
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")


def plot_month_diagnostics(
    metrics: Dict[str, Any],
    cfg: NdwiConfig,
    out_dir: Optional[Path] = None,
) -> None:
    """Plot and optionally save diagnostic figures for a single month.

    Parameters:
        metrics: Output from `compute_month_metrics()`.
        cfg: NDWI configuration (used for titles/filenames).
        out_dir: If provided, figures are saved to this directory; otherwise show only.

    Returns:
        None
    """
    if metrics.get("water") is None:
        print("No data for this month; skipping plots.")
        return

    year = int(metrics["year"])
    month = int(metrics["month"])

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Valid fraction map
    fig, ax = plt.subplots()
    metrics["valid_fraction_map"].plot(ax=ax, robust=True)
    ax.set_title(f"Valid (clear) observation fraction — {year}-{month:02d}")
    ax.set_axis_off()
    if out_dir is not None:
        save_figure_png(fig, out_dir / f"valid_fraction_{year}-{month:02d}.png")
    plt.show()

    # 2) NDWI median composite
    fig, ax = plt.subplots()
    metrics["ndwi_med"].plot(ax=ax, robust=True)
    ax.set_title(f"NDWI median composite — {year}-{month:02d}")
    ax.set_axis_off()
    if out_dir is not None:
        save_figure_png(fig, out_dir / f"ndwi_median_{year}-{month:02d}.png")
    plt.show()

    # 3) Water mask + AOI overlay
    title = f"Water mask vs AOI outline — {year}-{month:02d} (NDWI>{cfg.water_thresh})"
    fig, _ = make_watermask_overlay_figure(metrics["water"], metrics["aoi"], cfg.load_crs, title)
    if out_dir is not None:
        save_figure_png(
            fig,
            out_dir / f"overlay_watermask_vs_aoi_{year}-{month:02d}_ndwi{cfg.water_thresh:.2f}.png",
        )
    plt.show()


def compute_month_metrics(
    year: int,
    month: int,
    aoi_geojson: Path,
    cfg: NdwiConfig,
) -> Dict[str, Any]:
    """Compute NDWI-based water area and QA metrics for a single month.

    This function does computation only (no plotting, no saving). It returns both
    summary metrics and optional layers for downstream plotting.

    Parameters:
        year: Year to process.
        month: Month to process (1–12).
        aoi_geojson: AOI polygon GeoJSON path.
        cfg: NDWI configuration parameters.

    Returns:
        Dict with:
            year, month, n_items,
            water_area_km2,
            median_valid_fraction,
            valid_fraction_any,
            plus layers for optional QA/plots:
            aoi, valid_fraction_map, ndwi_med, water
    """
    # 1) STAC search
    items = search_month_from_config(year, month, cloud_cover_max=cfg.cloud_cover_max, limit=500)
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

    ndwi = ndwi_index(green=green, nir=nir)          # <- indices.py
    ndwi_clear = ndwi.where(valid)
    ndwi_med = ndwi_clear.median(dim="time", skipna=True)

    # 6) Water mask + area
    water = water_mask_from_ndwi(ndwi_med, cfg.water_thresh)  # <- water_mask.py
    water_area_km2 = compute_water_area_km2(water, ds_clip)   # <- water_mask.py

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
