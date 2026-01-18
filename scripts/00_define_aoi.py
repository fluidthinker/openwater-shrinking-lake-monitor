# %% [markdown]
# # 00 — Define AOI (ArcGIS Pro → GeoPackage → GeoJSON + STAC bbox)
#
# This notebook-style script is designed to run cell-by-cell in VS Code.
#
# Outputs:
# - data/external/aoi.geojson
# - configs/aoi_bbox.yaml
#
# Inputs:
# - data/external/aoi.gpkg (exported from ArcGIS Pro)

# %%
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt


# %%
# -----------------------------
# Canonical path setup (repo-root)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_EXTERNAL = REPO_ROOT / "data" / "external"
CONFIGS_DIR = REPO_ROOT / "configs"

GPKG_PATH = DATA_EXTERNAL / "aoi.gpkg"
OUT_GEOJSON = DATA_EXTERNAL / "aoi.geojson"
OUT_BBOX_YAML = CONFIGS_DIR / "aoi_bbox.yaml"

# If your GeoPackage has multiple layers, set this to the exact layer name.
# If None and exactly 1 layer exists, it will auto-select that layer.
LAYER_NAME: Optional[str] = None

# Plot preferences
SHOW_AXES = False


# %%
def print_runtime_context() -> None:
    """Print context useful for debugging path issues.

    Parameters:
        None

    Returns:
        None
    """
    print("Runtime context")
    print("---------------")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Input GPKG: {GPKG_PATH}")
    print(f"Output GeoJSON: {OUT_GEOJSON}")
    print(f"Output BBox YAML: {OUT_BBOX_YAML}")
    print("")


# %%
def choose_layer(gpkg_path: Path, layer_name: Optional[str] = None) -> str:
    """Choose which layer to read from a GeoPackage.

    If `layer_name` is provided, validates it exists and returns it.
    If not provided:
      - If there is exactly one layer, returns it.
      - If there are multiple layers, raises an error listing them.

    Parameters:
        gpkg_path (Path): Path to the GeoPackage.
        layer_name (str, optional): Explicit layer name to use.

    Returns:
        str: Selected layer name.

    Raises:
        FileNotFoundError: If the GeoPackage file does not exist.
        ValueError: If no layers exist, or multiple layers exist and no layer_name is provided.
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
        "Multiple layers found. Set LAYER_NAME near the top of this script.\n"
        f"Available layers: {layer_names}"
    )


# %%
def get_bbox_wgs84(gdf_wgs84: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Compute a bounding box from a GeoDataFrame in EPSG:4326.

    Parameters:
        gdf_wgs84 (geopandas.GeoDataFrame): AOI GeoDataFrame in EPSG:4326.

    Returns:
        tuple[float, float, float, float]: Bounding box as (min_lon, min_lat, max_lon, max_lat).
    """
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


# %%
def write_bbox_yaml(bbox: Tuple[float, float, float, float], out_path: Path) -> None:
    """Write a STAC-ready bounding box YAML file.

    Parameters:
        bbox (tuple[float, float, float, float]): Bounding box in EPSG:4326 as
            (min_lon, min_lat, max_lon, max_lat).
        out_path (Path): Output YAML path.

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
    print("BBox (EPSG:4326)")
    print("--------------")
    print(f"min_lon: {min_lon:.6f}")
    print(f"min_lat: {min_lat:.6f}")
    print(f"max_lon: {max_lon:.6f}")
    print(f"max_lat: {max_lon:.6f}")  # note: printed twice? keep below fixed
    print("")


# %%
def plot_aoi(gdf_wgs84: gpd.GeoDataFrame, title: str) -> None:
    """Plot AOI geometry quickly using GeoPandas/Matplotlib.

    Parameters:
        gdf_wgs84 (geopandas.GeoDataFrame): AOI in EPSG:4326.
        title (str): Plot title.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    gdf_wgs84.plot(ax=ax)
    ax.set_title(title)
    if not SHOW_AXES:
        ax.set_axis_off()
    plt.show()


# %%
def plot_aoi_with_bbox(gdf_wgs84: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]) -> None:
    """Plot AOI and its derived bbox outline.

    Parameters:
        gdf_wgs84 (geopandas.GeoDataFrame): AOI in EPSG:4326.
        bbox (tuple[float, float, float, float]): Bounding box as (min_lon, min_lat, max_lon, max_lat).

    Returns:
        None
    """
    from shapely.geometry import box

    bbox_geom = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326")

    fig, ax = plt.subplots()
    bbox_geom.plot(ax=ax, facecolor="none")
    gdf_wgs84.plot(ax=ax)
    ax.set_title("AOI + derived bbox (EPSG:4326)")
    if not SHOW_AXES:
        ax.set_axis_off()
    plt.show()


# %% [markdown]
# ## 1) Context + layer listing

# %%
print_runtime_context()


## Warning will occur running cell below because 
# 'Esri stores richer geometry/attribute types (Z/M, Esri date formats) and open-source readers sometimes warn when they simplify them.'


# %%
layers_df = gpd.list_layers(GPKG_PATH)
print("GeoPackage layers")
print("-----------------")
print(layers_df)
print("")

layer = choose_layer(GPKG_PATH, LAYER_NAME)
print(f"✅ Selected layer: {layer}")


# %% [markdown]
# ## 2) Read AOI + CRS sanity checks

# %%
gdf = gpd.read_file(GPKG_PATH, layer=layer)
print(f"Loaded {len(gdf)} feature(s).")
print("CRS:", gdf.crs)
print("")

if gdf.empty:
    raise ValueError(f"Layer '{layer}' contains 0 features. Check your ArcGIS export.")

if gdf.crs is None:
    raise ValueError(
        "AOI CRS is missing. In ArcGIS Pro, ensure the layer has a defined CRS before exporting."
    )


# %% [markdown]
# ## 3) Reproject to WGS84 (EPSG:4326)

# %%
gdf_wgs84 = gdf.to_crs(epsg=4326)
print("Reprojected CRS:", gdf_wgs84.crs)

# %%
print(gdf_wgs84.total_bounds)
gdf_wgs84.plot()
plt.show()

# %% [markdown]
# ## 4) Write GeoJSON + STAC bbox YAML

# %%
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Write GeoJSON
gdf_wgs84.to_file(OUT_GEOJSON, driver="GeoJSON")
print(f"✅ Wrote AOI GeoJSON: {OUT_GEOJSON}")

# Compute + write bbox YAML
bbox = get_bbox_wgs84(gdf_wgs84)
write_bbox_yaml(bbox, OUT_BBOX_YAML)
print(f"✅ Wrote bbox YAML: {OUT_BBOX_YAML}")

# Print bbox nicely
min_lon, min_lat, max_lon, max_lat = bbox
print("BBox (EPSG:4326)")
print("--------------")
print(f"min_lon: {min_lon:.6f}")
print(f"min_lat: {min_lat:.6f}")
print(f"max_lon: {max_lon:.6f}")
print(f"max_lat: {max_lat:.6f}")
print("")


# %% [markdown]
# ## 5) Quick plots (sanity check)

# %%
plot_aoi(gdf_wgs84, "AOI polygon (EPSG:4326)")

# %%
plot_aoi_with_bbox(gdf_wgs84, bbox)

# %% [markdown]
# ## Done
# Next: feed `configs/aoi_bbox.yaml` into your STAC search routines.
