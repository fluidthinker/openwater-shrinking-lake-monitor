# %% [markdown]
# # 00 — Define AOI (from ArcGIS Pro export)
#
# This notebook-style script:
# 1) Reads the AOI polygon exported from ArcGIS Pro (`data/external/aoi.gpkg`)
# 2) Lists available GeoPackage layers (and selects one)
# 3) Ensures CRS is EPSG:4326 (WGS84)
# 4) Writes `data/external/aoi.geojson`
# 5) Derives and writes a STAC-ready bbox YAML: `configs/aoi_bbox.yaml`
# 6) Produces quick plots for sanity-checking the geometry

# %%
print("Hello from Python")
import sys
print(sys.executable)






# %%
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt


# %%
# -----------------------------
# User-editable inputs
# -----------------------------
GPKG_PATH = Path("data/external/aoi.gpkg")
OUT_GEOJSON = Path("data/external/aoi.geojson")
OUT_BBOX_YAML = Path("configs/aoi_bbox.yaml")

# If your GeoPackage has multiple layers, set this.
# Otherwise, leave as None and the script will auto-select if only one layer exists.
LAYER_NAME: Optional[str] = None

# Quick visual settings
SHOW_BASE_AXES = False  # set True if you want axis ticks/labels


# %%
def choose_layer(gpkg_path: Path, layer_name: Optional[str] = None) -> str:
    """Choose which layer to read from a GeoPackage.

    Parameters:
        gpkg_path (Path): Path to the GeoPackage.
        layer_name (str, optional): Layer name to force (if known).

    Returns:
        str: The selected layer name.

    Raises:
        FileNotFoundError: If the GeoPackage does not exist.
        ValueError: If no layers exist, or multiple layers exist and none is specified.
    """
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    layers_df = gpd.list_layers(gpkg_path)
    layer_names = list(layers_df["name"])

    if len(layer_names) == 0:
        raise ValueError(f"No layers found in GeoPackage: {gpkg_path}")

    if layer_name is not None:
        if layer_name not in layer_names:
            raise ValueError(
                f"Requested layer '{layer_name}' not found.\n"
                f"Available layers: {layer_names}"
            )
        return layer_name

    if len(layer_names) == 1:
        return layer_names[0]

    raise ValueError(
        "Multiple layers found. Set LAYER_NAME at the top of this script.\n"
        f"Available layers: {layer_names}"
    )


# %%
def write_bbox_yaml(bbox: Tuple[float, float, float, float], out_path: Path) -> None:
    """Write a STAC-ready bbox YAML file.

    Parameters:
        bbox (tuple[float, float, float, float]): Bounding box as
            (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        out_path (Path): Output path for the YAML file.

    Returns:
        None
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        f"""bbox:
  min_lon: {min_lon}
  min_lat: {min_lat}
  max_lon: {max_lon}
  max_lat: {max_lat}
crs: EPSG:4326
""",
        encoding="utf-8",
    )


# %%
def print_bbox(bbox: Tuple[float, float, float, float]) -> None:
    """Print a bbox in a human-friendly form.

    Parameters:
        bbox (tuple[float, float, float, float]): Bounding box as
            (min_lon, min_lat, max_lon, max_lat).

    Returns:
        None
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    print("BBox (EPSG:4326):")
    print(f"  min_lon: {min_lon:.6f}")
    print(f"  min_lat: {min_lat:.6f}")
    print(f"  max_lon: {max_lon:.6f}")
    print(f"  max_lat: {max_lat:.6f}")


# %% [markdown]
# ## 1) List layers in the GeoPackage (and select one)

# %%
print(f"Reading AOI GeoPackage: {GPKG_PATH}")
layers_df = gpd.list_layers(GPKG_PATH)
print(layers_df)
# %%
layer = choose_layer(GPKG_PATH, LAYER_NAME)
print(f"\n✅ Selected layer: {layer}")


# %% [markdown]
# ## 2) Read AOI layer and verify CRS

# %%
gdf = gpd.read_file(GPKG_PATH, layer=layer)
print(f"Loaded {len(gdf)} feature(s).")
print("CRS:", gdf.crs)

# Quick peek at fields
print("\nColumns:", list(gdf.columns))
display_cols = [c for c in ["gnis_name", "gnisid", "id3dhp", "areasqkm"] if c in gdf.columns]
if display_cols:
    print("\nAttribute preview:")
    print(gdf[display_cols].head())
else:
    print("\n(No expected GNIS/3DHP fields found — that's fine.)")


# %%
if gdf.empty:
    raise ValueError(f"Layer '{layer}' is empty. Check your export from ArcGIS Pro.")

if gdf.crs is None:
    raise ValueError(
        "AOI CRS is missing. In ArcGIS Pro, define the layer CRS before exporting."
    )


# %% [markdown]
# ## 3) Reproject to EPSG:4326 (WGS84) for STAC compatibility

# %%
gdf_wgs84 = gdf.to_crs(epsg=4326)
print("Reprojected CRS:", gdf_wgs84.crs)


# %% [markdown]
# ## 4) Derive bounding box (for STAC searches) and write `configs/aoi_bbox.yaml`

# %%
minx, miny, maxx, maxy = gdf_wgs84.total_bounds
bbox = (float(minx), float(miny), float(maxx), float(maxy))

print_bbox(bbox)

write_bbox_yaml(bbox, OUT_BBOX_YAML)
print(f"\n✅ Wrote bbox YAML: {OUT_BBOX_YAML}")


# %% [markdown]
# ## 5) Write GeoJSON AOI (nice for inspection + reuse)

# %%
OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
gdf_wgs84.to_file(OUT_GEOJSON, driver="GeoJSON")
print(f"✅ Wrote AOI GeoJSON: {OUT_GEOJSON}")


# %% [markdown]
# ## 6) Quick plots (sanity check)
#
# If your AOI looks wrong here, it's usually:
# - wrong layer selected
# - CRS not what you think
# - definition query didn't apply before copying/exporting

# %%
fig, ax = plt.subplots()
gdf_wgs84.plot(ax=ax)
ax.set_title("AOI polygon (EPSG:4326)")
if not SHOW_BASE_AXES:
    ax.set_axis_off()
plt.show()

# %%
# BBox outline plot
from shapely.geometry import box  # imported here so it doesn't error if shapely isn't installed earlier

bbox_geom = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326")
fig, ax = plt.subplots()
bbox_geom.plot(ax=ax, facecolor="none")
gdf_wgs84.plot(ax=ax)
ax.set_title("AOI + derived bbox (EPSG:4326)")
if not SHOW_BASE_AXES:
    ax.set_axis_off()
plt.show()


# %% [markdown]
# ## Done
# Next step is to use `configs/aoi_bbox.yaml` in your STAC search routines to fetch imagery.
