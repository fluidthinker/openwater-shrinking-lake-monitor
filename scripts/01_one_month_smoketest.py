


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
from __future__ import annotations
from pathlib import Path
import sys

# Canonical repo-root resolution
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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

# %%
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
# ## REMOVE DIAGNOSTIC
# --- Tiny read test (pre-clip) to see if assets actually load ---
# pick a small 512x512 window near the middle and only the first timestamp
y0 = ds.sizes["y"] // 2
x0 = ds.sizes["x"] // 2

scl_small = ds["SCL"].isel(
    time=0,
    y=slice(y0 - 256, y0 + 256),
    x=slice(x0 - 256, x0 + 256),
)

# Force an actual read of that small window
scl_small_np = scl_small.compute().values

print("PRE-CLIP SCL small window:")
print("  shape:", scl_small_np.shape)
print("  NaN fraction:", float(np.isnan(scl_small_np).mean()))
print("  min/max:", np.nanmin(scl_small_np), np.nanmax(scl_small_np))






# %% [markdown]
# ## REMOVE DIAGNOSTIC PRINTS for NDWI stats

print("ds CRS:", ds.rio.crs)
print("ds dims:", ds.dims)
print("ds sizes:", ds.sizes)

scl0 = ds["SCL"].isel(time=0)
print("SCL(time=0) NaN fraction (pre-clip):", float(np.isnan(scl0).mean().compute().values))
print("SCL(time=0) min/max (pre-clip):",
      scl0.min(skipna=True).compute().values,
      scl0.max(skipna=True).compute().values)



# %% [markdown]
# ## 3) Clip to AOI polygon

# %%
aoi = read_aoi_geojson(AOI_GEOJSON)

ds_clip = clip_to_aoi(ds, aoi)
print(ds_clip)

# %% [markdown]
# ## REMOVE DIAGNOSTIC PRINTS
print("POST-CLIP ds_clip sizes:", ds_clip.sizes)

# Same tiny read test after clip
y0c = ds_clip.sizes["y"] // 2
x0c = ds_clip.sizes["x"] // 2

scl_small_c = ds_clip["SCL"].isel(
    time=0,
    y=slice(y0c - 256, y0c + 256),
    x=slice(x0c - 256, x0c + 256),
)

scl_small_c_np = scl_small_c.compute().values
print("POST-CLIP SCL small window:")
print("  NaN fraction:", float(np.isnan(scl_small_c_np).mean()))
print("  min/max:", np.nanmin(scl_small_c_np), np.nanmax(scl_small_c_np))






# %% [markdown]
# ## REMOVE DIAGNOSTIC PRINTS for NDWI stats
# --- Sanity check: did the bands actually load? ---
for b in ["B03", "B08", "SCL"]:
    da = ds_clip[b].astype("float32")

    nan_frac = float(np.isnan(da).mean().compute().values)

    # min/max with skipna; compute because it's dask-backed
    vmin = da.min(skipna=True).compute().values
    vmax = da.max(skipna=True).compute().values

    print(f"{b}: NaN fraction={nan_frac:.3f}  min={vmin}  max={vmax}")

# %% DIAGNOSTIC PRINTS for SCL stats
scl = ds_clip["SCL"]
print("SCL NaN fraction:", float(np.isnan(scl).mean().compute().values))
print("SCL min/max:", scl.min(skipna=True).compute().values, scl.max(skipna=True).compute().values)
print(ds["SCL"])

# %% [markdown]
# ## 4) Cloud/clear coverage map (valid observation fraction)

# %%
valid = scl_cloud_mask(ds_clip["SCL"])  # True = clear/valid pixel

valid_count = valid.sum(dim="time")
total_count = valid.count(dim="time")  # counts non-NaN SCL values

valid_fraction_map = (valid_count / total_count).astype("float32")
valid_fraction_map = valid_fraction_map.rename("valid_fraction")

print("valid_fraction_map min/max:",
      float(valid_fraction_map.min().compute()),
      float(valid_fraction_map.max().compute()))



plt.figure()
valid_fraction_map.plot(robust=True)
plt.title(f"Valid (clear) observation fraction — {YEAR}-{MONTH:02d}")

plt.show()

# Convert from Dask → NumPy, then compute scalar
arr = valid_fraction_map.data.compute()   # now it's a NumPy ndarray in memory
median_valid_fraction = float(np.nanmedian(arr))

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
# ## REMOVE DIAGNOSTIC PRINTS for NDWI stats
print("ndwi dtype:", ndwi_med.dtype)
print("ndwi backed by dask?:", hasattr(ndwi_med.data, "compute"))

ndwi_min = float(ndwi_med.min(skipna=True).compute().values)
ndwi_max = float(ndwi_med.max(skipna=True).compute().values)
ndwi_nan_frac = float(np.isnan(ndwi_med).mean().compute().values)

print("ndwi min/max:", ndwi_min, ndwi_max)
print("ndwi NaN fraction:", round(ndwi_nan_frac, 3))


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
