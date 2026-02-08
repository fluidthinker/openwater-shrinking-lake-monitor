from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import xarray as xr
from odc.stac import load as odc_load





def repo_root_from_file(py_file: str) -> Path:
    """Resolve repository root from a module file path.

    Parameters:
        py_file (str): Typically pass __file__ from the calling module.

    Returns:
        pathlib.Path: Repository root directory.
    """
    return Path(py_file).resolve().parents[2]


def read_aoi_geojson(aoi_geojson_path: Path) -> gpd.GeoDataFrame:
    """Read AOI GeoJSON as a GeoDataFrame.

    Parameters:
        aoi_geojson_path (Path): Path to data/external/aoi.geojson.

    Returns:
        geopandas.GeoDataFrame: AOI features.
    """
    gdf = gpd.read_file(aoi_geojson_path)
    if gdf.empty:
        raise ValueError(f"AOI GeoJSON is empty: {aoi_geojson_path}")
    if gdf.crs is None:
        raise ValueError(f"AOI GeoJSON has no CRS: {aoi_geojson_path}")
    return gdf


def scl_cloud_mask(scl: xr.DataArray) -> xr.DataArray:
    """Create a boolean valid-pixel mask from Sentinel-2 Scene Classification Layer (SCL).

    We mark pixels invalid if they are:
      - NaN
      - No data (0)
      - Saturated/defective (1)
      - Cloud shadows (3)
      - Cloud medium probability (8)
      - Cloud high probability (9)
      - Cirrus (10)
      - Snow/ice (11)

    Parameters:
        scl (xarray.DataArray): SCL band.

    Returns:
        xarray.DataArray: Boolean mask where True means "valid / clear-sky".
    """
   

    invalid = (
    scl.isnull()
    | (scl == 0)
    | (scl == 1)
    | (scl == 3)
    | (scl == 8)
    | (scl == 9)
    | (scl == 10)
    | (scl == 11)
)

    
    return ~invalid

def load_s2_items_odc(
    items: List[Any],
    *,
    bands: List[str],
    crs: str = "EPSG:3857",
    resolution: float = 10.0,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Load Sentinel-2 items into an xarray Dataset using ODC-STAC.

    Parameters:
        items (list[Any]): Signed STAC items.
        bands (list[str]): Bands to load (must be explicitly specified).
        crs (str): Output CRS for analysis grid.
        resolution (float): Output pixel resolution in meters.
        chunks (dict, optional): Dask chunk sizes.

    Returns:
        xarray.Dataset: Dataset with dims (time, y, x) and requested bands.
    """
    if not items:
        raise ValueError("No items provided to load_s2_items_odc().")

    if not bands:
        raise ValueError("bands must be a non-empty list of band names.")

    if chunks is None:
        chunks = {"x": 2048, "y": 2048}

    ds = odc_load(
        items,
        bands=bands,
        crs=crs,
        resolution=resolution,
        chunks=chunks,
        groupby="solar_day",
        fail_on_error=True,
    )
    return ds




def clip_to_aoi(ds: xr.Dataset, aoi_gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """Clip a Dataset to AOI bounds using rioxarray.

    Parameters:
        ds (xarray.Dataset): Dataset in a projected CRS (e.g., EPSG:3857).
        aoi_gdf (geopandas.GeoDataFrame): AOI polygon(s). Will be reprojected to ds CRS.

    Returns:
        xarray.Dataset: Clipped dataset.
    """
    # rioxarray works best when CRS is attached properly
    import rioxarray  # noqa: F401

    if not hasattr(ds, "rio"):
        raise ValueError("rioxarray accessor not available on dataset.")

    if ds.rio.crs is None:
        raise ValueError("Dataset CRS is missing; cannot clip.")

    aoi_proj = aoi_gdf.to_crs(ds.rio.crs)

    ds = ds.rio.clip(aoi_proj.geometry, aoi_proj.crs, drop=True, all_touched=True)
    return ds

