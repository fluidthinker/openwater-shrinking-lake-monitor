# %% [markdown]
# # Export Sentinel-2 True Color (RGB) Context Images
#
# Goal:
# Export "human readable" true-color images (RGB composite) for a small set of
# (year, month) targets (e.g., 2019-09 and 2025-09) to support the README.
#
# Approach:
# 1) Search Sentinel-2 L2A items for the month using the repo's bbox config
# 2) Select the "clearest" item by lowest `eo:cloud_cover` (STAC metadata)
# 3) Load only that single item using ODC-STAC for RGB bands (B04, B03, B02)
# 4) Clip to AOI polygon
# 5) Apply a simple percentile stretch for display
# 6) Save PNG to outputs/images/
#
# Output:
# outputs/images/s2_rgb_YYYY-MM.png

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt




# %% 
# Ensure repo root is on sys.path 
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stac.search_items import search_month_from_config
from src.stac.load_odc import load_s2_items_odc


# %% [markdown]
# ## Paths + targets

# %%

AOI_GEOJSON = REPO_ROOT / "data" / "external" / "aoi.geojson"

OUTPUT_DIR = REPO_ROOT / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal "low energy" targets, but loop-ready
TARGET_PERIODS: List[Tuple[int, int]] = [
    (2019, 9),
    (2025, 9),
]

# Sentinel-2 true color (RGB) bands:
# Red=B04, Green=B03, Blue=B02
RGB_BANDS = ["B04", "B03", "B02"]


# %% [markdown]
# ## Helper: choose the clearest item by STAC metadata
#
# Planetary Computer / STAC items include `eo:cloud_cover` in item.properties.
# Lower is generally better. This is not AOI-specific, but it's usually good
# enough for selecting a clean "context image" for a README.

# %%
def get_cloud_cover(item: Any) -> float:
    """Safely extract `eo:cloud_cover` from a STAC item."""
    props = getattr(item, "properties", {}) or {}
    cc = props.get("eo:cloud_cover")
    return float(cc) if cc is not None else 9999.0


def pick_clearest_item(items: List[Any]) -> Optional[Any]:
    """Pick the STAC item with the lowest `eo:cloud_cover`."""
    if not items:
        return None
    return sorted(items, key=get_cloud_cover)[0]


# %% [markdown]
# ## Helper: percentile stretch for display
#
# Raw reflectance is not photo-ready. A percentile stretch makes the image
# easier to interpret (similar to what GIS tools do for visualization).

# %%
def percentile_stretch(rgb: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Percentile stretch a (H, W, 3) RGB array to [0, 1].

    Parameters
    ----------
    rgb : np.ndarray
        Raw reflectance-like values, shape (H, W, 3).
    p_low : float
        Lower percentile clip.
    p_high : float
        Upper percentile clip.

    Returns
    -------
    np.ndarray
        Stretched float array in [0, 1], shape (H, W, 3).
    """
    out = np.zeros_like(rgb, dtype=np.float32)

    for b in range(3):
        band = rgb[..., b]
        lo = np.nanpercentile(band, p_low)
        hi = np.nanpercentile(band, p_high)
        band = np.clip(band, lo, hi)
        out[..., b] = (band - lo) / (hi - lo + 1e-9)

    return np.clip(out, 0.0, 1.0)


# %% [markdown]
# ## Export function: (year, month) -> RGB PNG
#
# Steps:
# - Search month items
# - Pick clearest by cloud cover
# - Load RGB bands for that single item
# - Clip to AOI polygon
# - Stretch & save a PNG
def select_items_for_month(
    year: int,
    month: int,
    *,
    cloud_cover_max: float,
    top_n: int,
) -> List[Any]:
    """
    Search a month and select up to `top_n` items with the lowest eo:cloud_cover.
    """
    items = search_month_from_config(year, month, cloud_cover_max=cloud_cover_max)
    if not items:
        raise RuntimeError(f"No Sentinel-2 items found for {year}-{month:02d}")

    items_sorted = sorted(items, key=get_cloud_cover)
    chosen = items_sorted[:top_n]

    best_cc = get_cloud_cover(chosen[0])
    print(f"✅ {year}-{month:02d}: using {len(chosen)} items (best cloud={best_cc:.1f}%)")
    return chosen


def load_rgb_composite(
    items: List[Any],
    *,
    crs: str = "EPSG:3857",
    resolution: float = 10.0,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Load RGB bands for multiple items and return a single composite dataset (no time dim)
    by taking the median over time.
    """
    if chunks is None:
        chunks = {"x": 2048, "y": 2048}

    ds = load_s2_items_odc(
        items=items,
        bands=RGB_BANDS,
        crs=crs,
        resolution=resolution,
        chunks=chunks,
    )

    # Mosaic-ish composite: median across time fills gaps across overlapping swaths
    ds_comp = ds.median(dim="time", skipna=True)
    return ds_comp


def dataset_to_rgb_array(ds: "xr.Dataset") -> np.ndarray:
    """
    Convert a dataset with B04/B03/B02 variables into an (H, W, 3) numpy array.
    """
    r = ds["B04"]
    g = ds["B03"]
    b = ds["B02"]
    rgb = np.stack([r.values, g.values, b.values], axis=-1).astype(np.float32)
    return rgb


def export_rgb_context(
    year: int,
    month: int,
    *,
    cloud_cover_max: float = 80.0,
    top_n: int = 10,
    stretch_low: float = 2.0,
    stretch_high: float = 98.0,
) -> Path:
    """
    Export a true-color RGB composite image for a given month.

    Strategy:
    - choose top N items by lowest eo:cloud_cover
    - load RGB for those items
    - compute a median composite over time to improve AOI coverage
    - save a stretched PNG for README context
    """
    chosen = select_items_for_month(
        year,
        month,
        cloud_cover_max=cloud_cover_max,
        top_n=top_n,
    )

    ds_comp = load_rgb_composite(chosen)

    rgb = dataset_to_rgb_array(ds_comp)
    rgb_disp = percentile_stretch(rgb, p_low=stretch_low, p_high=stretch_high)

    out_path = OUTPUT_DIR / f"s2_rgb_{year}-{month:02d}.png"

    # Use the best (lowest) cloud-cover value only for labeling
    best_cc = get_cloud_cover(sorted(chosen, key=get_cloud_cover)[0])

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_disp)
    plt.axis("off")
    plt.title(
        f"Sentinel-2 True Color (RGB) — {year}-{month:02d}  |  best cloud={best_cc:.1f}%  |  N={len(chosen)}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")
    return out_path



# %%
def main() -> None:
    for year, month in TARGET_PERIODS:
        export_rgb_context(
            year,
            month,
            cloud_cover_max=80.0,
            top_n=12,
            stretch_low=2.0,
            stretch_high=98.0,
        )

if __name__ == "__main__":
    main()

