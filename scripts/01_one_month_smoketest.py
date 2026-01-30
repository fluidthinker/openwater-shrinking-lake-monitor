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

# %% [markdown]
# # Imports & setup
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
MONTH = 7 # 1-12

# Turn plots on/off for batch readiness
PLOT = True

# Paths
AOI_GEOJSON = REPO_ROOT / "data" / "external" / "aoi.geojson"
CFG_NDWI = REPO_ROOT / "configs" / "ndwi.yaml"



# %% [markdown]
# # Load Functions
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

def make_watermask_overlay_figure(
    water: xr.DataArray,
    aoi_gdf,
    crs: str,
    title: str,
):
    """Create a matplotlib figure showing water mask with AOI boundary overlay."""
    aoi_proj = aoi_gdf.to_crs(crs)

    fig, ax = plt.subplots(figsize=(8, 8))
    water.plot(ax=ax, add_colorbar=False)
    aoi_proj.boundary.plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_axis_off()
    return fig, ax

def save_figure_png(fig: plt.Figure, out_png: Path, dpi: int = 200) -> None:
    """Save a matplotlib figure to a PNG path."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")


from pathlib import Path
import matplotlib.pyplot as plt


def plot_month_diagnostics(
    metrics: dict,
    cfg: NdwiConfig,
    out_dir: Path | None = None,
) -> None:
    """Plot and optionally save diagnostic figures for a single month.

    Parameters
    ----------
    metrics : dict
        Output from `compute_month_metrics()`.
    cfg : NdwiConfig
        NDWI configuration (used for titles and filenames).
    out_dir : Path, optional
        If provided, figures are saved to this directory.
        If None, figures are shown only.

    Returns
    -------
    None
    """
    if metrics["water"] is None:
        print("No data for this month; skipping plots.")
        return

    year = metrics["year"]
    month = metrics["month"]

    # Ensure output directory exists if saving
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Valid fraction map
    fig, ax = plt.subplots()
    metrics["valid_fraction_map"].plot(ax=ax, robust=True)
    ax.set_title(f"Valid (clear) observation fraction — {year}-{month:02d}")
    ax.set_axis_off()

    if out_dir is not None:
        fig.savefig(
            out_dir / f"valid_fraction_{year}-{month:02d}.png",
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()

    # 2) NDWI median composite
    fig, ax = plt.subplots()
    metrics["ndwi_med"].plot(ax=ax, robust=True)
    ax.set_title(f"NDWI median composite — {year}-{month:02d}")
    ax.set_axis_off()

    if out_dir is not None:
        fig.savefig(
            out_dir / f"ndwi_median_{year}-{month:02d}.png",
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()

    # 3) Water mask + AOI overlay
    title = f"Water mask vs AOI outline — {year}-{month:02d} (NDWI>{cfg.water_thresh})"
    fig, _ = make_watermask_overlay_figure(
        metrics["water"],
        metrics["aoi"],
        cfg.load_crs,
        title,
    )

    if out_dir is not None:
        fig.savefig(
            out_dir / f"overlay_watermask_vs_aoi_{year}-{month:02d}_ndwi{cfg.water_thresh:.2f}.png",
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()





def compute_month_metrics(
    year: int,
    month: int,
    aoi_geojson: Path,
    cfg: NdwiConfig,
    ) -> Dict[str, Any]:
    """Run the one-month Sentinel-2 NDWI workflow and return summary metrics + key layers.

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
    Returns
    -------
    dict
        Contains:
        - year, month, n_items
        - water_area_km2
        - median_valid_fraction
        - valid_fraction_any

        And (for optional plotting/QA):
        - aoi (GeoDataFrame)
        - valid_fraction_map (DataArray)
        - ndwi_med (DataArray)
        - water (DataArray; boolean mask)
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
            # Optional layers for downstream plotting (None signals "no data")
            "aoi": None,
            "valid_fraction_map": None,
            "ndwi_med": None,
            "water": None,
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

    # AOI summary stats should reflect pixels that had at least one clear look
    stats_mask = has_data & (valid_count > 0)
    vf = valid_fraction_map.where(stats_mask)

    vf_np = vf.data.compute()
    median_valid_fraction = float(np.nanmedian(vf_np))

    valid_any_np = valid_any.data.compute()
    valid_fraction_any = float(np.nanmean(valid_any_np))

    # 5) NDWI composite
    green = ds_clip["B03"].astype("float32")
    nir = ds_clip["B08"].astype("float32")

    den = green + nir
    ndwi = xr.where(den != 0, (green - nir) / den, np.nan)

    ndwi_clear = ndwi.where(valid)
    ndwi_med = ndwi_clear.median(dim="time", skipna=True)

    # 6) Water area
    water = ndwi_med > cfg.water_thresh

    res_x, res_y = ds_clip.rio.resolution()
    pixel_area_m2 = abs(res_x * res_y)

    water_area_m2 = float(water.sum().values) * pixel_area_m2
    water_area_km2 = water_area_m2 / 1e6

    return {
        "year": year,
        "month": month,
        "n_items": int(n_items),
        "water_area_km2": float(water_area_km2),
        "median_valid_fraction": float(median_valid_fraction),
        "valid_fraction_any": float(valid_fraction_any),
        # Optional layers for downstream plotting/QA
        "aoi": aoi,
        "valid_fraction_map": valid_fraction_map,
        "ndwi_med": ndwi_med,
        "water": water,
    }


# %% [markdown]
# # Compute Metrics and Plot results

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

metrics = compute_month_metrics(YEAR, MONTH, AOI_GEOJSON, cfg=cfg)

print("\n--- Monthly metrics ---")
for k in ["year", "month", "n_items", "water_area_km2", "median_valid_fraction", "valid_fraction_any"]:
    print(f"{k:>22}: {metrics.get(k)}")

if PLOT:
    plot_month_diagnostics(
        metrics,
        cfg,
        out_dir=REPO_ROOT / "outputs" / "figures",
    )



# %%
