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
# %% DIAGNOSTIC
try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # VS Code / interactive cell: __file__ may not exist
    REPO_ROOT = Path.cwd()

print("REPO_ROOT:", REPO_ROOT)
print("Contains src?:", (REPO_ROOT / "src").exists())
print("sys.path[0]:", sys.path[0])


# %% 
# Ensure repo root is on sys.path (Option A)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stac.search_items import search_month_from_config
from src.stac.load_odc import read_aoi_geojson, clip_to_aoi, load_s2_items_odc


# %% [markdown]
# ## Paths + targets

# %%
REPO_ROOT = Path(__file__).resolve().parents[1]



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

# %%
def export_rgb_context(year: int, month: int, cloud_cover_max: float = 80.0) -> Path:
    """
    Export a single true-color RGB image for a given month.

    Parameters
    ----------
    year : int
        Year (e.g., 2019)
    month : int
        Month 1-12 (e.g., 9 for September)
    cloud_cover_max : float
        Broad filter for search; final selection uses lowest eo:cloud_cover.

    Returns
    -------
    Path
        Output PNG path.
    """
    # 1) Search items using repo bbox config
    items = search_month_from_config(year, month, cloud_cover_max=cloud_cover_max)

    best = pick_clearest_item(items)
    if best is None:
        raise RuntimeError(f"No Sentinel-2 items found for {year}-{month:02d}")

    cc = get_cloud_cover(best)
    print(f"✅ {year}-{month:02d}: selected item eo:cloud_cover={cc:.1f}%")

    # 2) Load RGB for the single chosen item
    ds = load_s2_items_odc(
        items=[best],
        bands=RGB_BANDS,
        crs="EPSG:3857",
        resolution=10.0,
        chunks={"x": 2048, "y": 2048},
    )

    # 3) Clip to AOI polygon
    aoi = read_aoi_geojson(AOI_GEOJSON)
    ds_clip = clip_to_aoi(ds, aoi)

    # 4) Build (H, W, 3) RGB array
    r = ds_clip["B04"]
    g = ds_clip["B03"]
    b = ds_clip["B02"]

    # ODC load returns a time dimension even for a single item; take time=0
    if "time" in r.dims:
        r = r.isel(time=0)
        g = g.isel(time=0)
        b = b.isel(time=0)

    rgb = np.stack([r.values, g.values, b.values], axis=-1).astype(np.float32)

    # 5) Stretch for display and save
    rgb_disp = percentile_stretch(rgb, p_low=2, p_high=98)

    out_path = OUTPUT_DIR / f"s2_rgb_{year}-{month:02d}.png"

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_disp)
    plt.axis("off")
    plt.title(f"Sentinel-2 True Color (RGB) — {year}-{month:02d}  |  cloud={cc:.1f}%")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")
    return out_path


# %% [markdown]
# ## Run exports for all target periods

# %%
for year, month in TARGET_PERIODS:
    export_rgb_context(year, month, cloud_cover_max=80.0)
