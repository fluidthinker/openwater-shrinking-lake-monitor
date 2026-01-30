import numpy as np
import xarray as xr

def water_mask_from_ndwi(ndwi_med: xr.DataArray, thresh: float) -> xr.DataArray:
    """Threshold NDWI median composite to a boolean water mask."""
    return ndwi_med > thresh


def compute_water_area_km2(water: xr.DataArray, ds_clip: xr.Dataset) -> float:
    """Compute water area in km^2 from a boolean mask and raster resolution."""
    res_x, res_y = ds_clip.rio.resolution()
    pixel_area_m2 = abs(res_x * res_y)
    water_area_m2 = float(water.sum().values) * pixel_area_m2
    return water_area_m2 / 1e6


def compute_valid_metrics(valid: xr.DataArray, scl: xr.DataArray):
    """Compute monthly QA metrics and valid fraction map.

    Parameters:
        valid: Boolean array (time,y,x) where True means clear/usable.
        scl: SCL DataArray used to compute total_count (not-null count).

    Returns:
        (median_valid_fraction, valid_fraction_any, valid_fraction_map)
    """
    valid_count = valid.sum(dim="time")
    total_count = scl.notnull().sum(dim="time")

    valid_fraction_map = xr.where(total_count > 0, valid_count / total_count, np.nan).astype("float32")

    has_data = total_count > 0
    valid_any = valid.any(dim="time").where(has_data)

    stats_mask = has_data & (valid_count > 0)
    vf = valid_fraction_map.where(stats_mask)

    median_valid_fraction = float(np.nanmedian(vf.data.compute()))
    valid_fraction_any = float(np.nanmean(valid_any.data.compute()))

    return median_valid_fraction, valid_fraction_any, valid_fraction_map
