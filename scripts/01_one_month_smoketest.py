# %% [markdown]
# # 01 — One-month smoke test (Sentinel-2 L2A + Planetary Computer)
#
# Goal:
# - Search STAC for one month of Sentinel-2 L2A scenes over the AOI bbox
# - Load bands (B03, B08, SCL) via ODC-STAC
# - Clip to AOI polygon
# - Visualize cloud/clear coverage
# - Compute NDWI monthly median composite
# - Compute first-pass water area + valid_fraction
#
# Prereqs (in your geospatial env):
#   python -m pip install -U pystac-client planetary-computer odc-stac rioxarray
#
# Inputs:
# - configs/aoi_bbox.yaml
# - data/external/aoi.geojson
# %%
from pathlib import Path
import sys

# Canonical repo-root resolution
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.stac.search_items import search_month_from_config
from src.stac.load_odc import (
    clip_to_aoi,
    load_s2_items_odc,
    read_aoi_geojson,
    scl_cloud_mask,
)

# %%
# -----------------------------
# Canonical repo-root paths (Option A)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
AOI_GEOJSON = REPO_ROOT / "data" / "external" / "aoi.geojson"

print("Repo root:", REPO_ROOT)
print("AOI GeoJSON:", AOI_GEOJSON)

# %%
# -----------------------------
# Pick a month to test
# -----------------------------
YEAR = 2020
MONTH = 7  # 1-12

# Keep this lenient at first; we do cloud masking later anyway.
CLOUD_COVER_MAX = 80

# NDWI threshold (tune later)
WATER_THRESH = 0.1

print(f"Testing month: {YEAR}-{MONTH:02d}")
print(f"Cloud cover max (STAC query): {CLOUD_COVER_MAX}")
print(f"Water threshold (NDWI): {WATER_THRESH}")

# %% [markdown]
# ## 1) STAC search: how many scenes are available?

items = search_month_from_config(YEAR, MONTH, cloud_cover_max=CLOUD_COVER_MAX, limit=500)

print("Items found:", len(items))
if len(items) == 0:
    raise ValueError(
        "No items returned from STAC search. Try a different month/year or increase CLOUD_COVER_MAX."
    )

# Quick look at cloud cover distribution
cc = []
for it in items:
    props = getattr(it, "properties", {}) or {}
    if "eo:cloud_cover" in props:
        cc.append(props["eo:cloud_cover"])

if cc:
    print(f"eo:cloud_cover: min={min(cc):.1f}  median={float(np.median(cc)):.1f}  max={max(cc):.1f}")

# %% [markdown]
# ## 2) Load data with ODC-STAC (B03, B08, SCL)

# %%
ds = load_s2_items_odc(
    items,
    bands=["B03", "B08", "SCL"],
    crs="EPSG:3857",
    resolution=10.0,
    chunks={"x": 2048, "y": 2048},
)

print(ds)

# %% [markdown]
# ## 3) Clip to AOI polygon

# %%
aoi = read_aoi_geojson(AOI_GEOJSON)

ds_clip = clip_to_aoi(ds, aoi)
print(ds_clip)

# %% REMOVE DIAGNOSTICS
times = ds_clip["time"].values
print("n items:", len(times))
print("n unique timestamps:", len(set(times.astype("datetime64[s]"))))
print("unique dates:", len(set(times.astype("datetime64[D]"))))

import pandas as pd

t = pd.to_datetime(ds_clip["time"].values)
counts = pd.Series(1, index=t.normalize()).groupby(level=0).sum().sort_index()
print(counts)




# %% [markdown]
# ## 4) Cloud/clear coverage map (valid observation fraction)

# %%
valid = scl_cloud_mask(ds_clip["SCL"])  # True = clear/valid pixel

valid_count = valid.sum(dim="time")
total_count = valid.count(dim="time")  # counts non-NaN SCL values

valid_fraction_map = (valid_count / total_count).astype("float32")
print(f'Type of valid_fracton_map.data', type(valid_fraction_map.data))

# %% REMOVE DIAGNOSTICS
print("Backend array type:", type(valid_fraction_map.data))
print("Is Dask-backed?", hasattr(valid_fraction_map.data, "compute"))

# %% 
plt.figure()
valid_fraction_map.plot(robust=True)
plt.title(f"Valid (clear) observation fraction — {YEAR}-{MONTH:02d}")
plt.show()

median_valid_fraction = float(
    np.nanmedian(valid_fraction_map.values)
)

print(
    "Median valid fraction (per-pixel, across AOI):",
    round(median_valid_fraction, 3),
)


# A scalar "is this month usable?" metric:
valid_any = valid.any(dim="time")  # True where at least one clear observation exists in month
valid_fraction = float(valid_any.mean().values)
print("Valid fraction (pixels with ANY clear obs):", round(valid_fraction, 3))

# %% [markdown]
# ## 5) NDWI monthly median composite

# %%
green = ds_clip["B03"].astype("float32")
nir = ds_clip["B08"].astype("float32")

ndwi = (green - nir) / (green + nir)

# Mask clouds per time slice, then composite
ndwi_clear = ndwi.where(valid)
ndwi_med = ndwi_clear.median(dim="time", skipna=True)

plt.figure()
ndwi_med.plot(robust=True)
plt.title(f"NDWI median composite — {YEAR}-{MONTH:02d}")
plt.show()

# %% [markdown]
# ## 6) First-pass water mask + water area estimate (km²)

# %%
water = ndwi_med > WATER_THRESH

# Pixel area in m^2 (we loaded EPSG:3857 at 10m resolution)
# rioxarray stores resolution on the clipped dataset
res_x, res_y = ds_clip.rio.resolution()
pixel_area_m2 = abs(res_x * res_y)

water_area_m2 = float(water.sum().values) * pixel_area_m2
water_area_km2 = water_area_m2 / 1e6

print(f"Pixel size (m): {abs(res_x):.2f} x {abs(res_y):.2f}  => pixel_area_m2={pixel_area_m2:.2f}")
print(f"Water area (km^2) @ NDWI>{WATER_THRESH}: {water_area_km2:.2f}")
print(f"Valid fraction (any-clear pixels): {valid_fraction:.3f}")

plt.figure()
water.plot()
plt.title(f"Water mask — {YEAR}-{MONTH:02d} (NDWI>{WATER_THRESH})")
plt.show()

# %% [markdown]
# ## Done
#
# If this looks good, next step is to:
# - loop months from 2018-01 → today
# - compute monthly NDWI composites + water area
# - write outputs/reports/water_area_monthly.csv
# - plot monthly series + annual min/median/max
