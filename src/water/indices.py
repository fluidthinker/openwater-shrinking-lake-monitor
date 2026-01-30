import numpy as np
import xarray as xr

def ndwi_index(green: xr.DataArray, nir: xr.DataArray, eps: float = 0.0) -> xr.DataArray:
    """Compute NDWI = (green - nir) / (green + nir).

    Parameters:
        green: Green band reflectance.
        nir: NIR band reflectance.
        eps: Optional epsilon added to denominator to avoid divide-by-zero.
             Use eps=0.0 to keep your current "set to NaN when denom==0" behavior.

    Returns:
        NDWI DataArray.
    """
    den = green + nir
    if eps > 0:
        return (green - nir) / (den + eps)
    return xr.where(den != 0, (green - nir) / den, np.nan)
