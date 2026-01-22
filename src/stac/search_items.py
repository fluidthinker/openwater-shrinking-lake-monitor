from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from pystac_client import Client
import planetary_computer as pc


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class BBox:
    """A geographic bounding box in WGS84.

    Parameters:
        min_lon (float): Minimum longitude.
        min_lat (float): Minimum latitude.
        max_lon (float): Maximum longitude.
        max_lat (float): Maximum latitude.

    Returns:
        None
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_list(self) -> List[float]:
        """Return bbox as [min_lon, min_lat, max_lon, max_lat].

        Parameters:
            None

        Returns:
            list[float]: Bounding box list.
        """
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


def repo_root_from_file(py_file: str) -> Path:
    """Resolve repository root from a module file path.

    Parameters:
        py_file (str): Typically pass __file__ from the calling module.

    Returns:
        pathlib.Path: Repository root directory.
    """
    return Path(py_file).resolve().parents[2]


def read_bbox_yaml(bbox_yaml_path: Path) -> BBox:
    """Read bbox config from YAML.

    Expected YAML format:
        bbox:
          min_lon: ...
          min_lat: ...
          max_lon: ...
          max_lat: ...
        crs: EPSG:4326

    Parameters:
        bbox_yaml_path (Path): Path to configs/aoi_bbox.yaml.

    Returns:
        BBox: Bounding box object.
    """
    data = yaml.safe_load(bbox_yaml_path.read_text(encoding="utf-8"))
    b = data["bbox"]
    return BBox(
        min_lon=float(b["min_lon"]),
        min_lat=float(b["min_lat"]),
        max_lon=float(b["max_lon"]),
        max_lat=float(b["max_lat"]),
    )


def month_datetime_range(year: int, month: int) -> str:
    """Return an ISO-8601 datetime range string for a calendar month.

    Parameters:
        year (int): Year (e.g., 2020).
        month (int): Month 1-12.

    Returns:
        str: Datetime range like "2020-01-01/2020-01-31".
    """
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    # STAC uses end-exclusive nicely; ODC can handle either. We'll use end-exclusive.
    return f"{start.isoformat()}/{end.isoformat()}"


def search_sentinel2_l2a_items(
    bbox: BBox,
    datetime_range: str,
    cloud_cover_max: Optional[float] = 80.0,
    limit: int = 500,
) -> List[Any]:
    """Search Planetary Computer STAC for Sentinel-2 L2A items.

    Parameters:
        bbox (BBox): Search bbox in EPSG:4326.
        datetime_range (str): ISO datetime range, e.g., "2019-01-01/2019-02-01".
        cloud_cover_max (float, optional): Max eo:cloud_cover percent. If None, no filter.
        limit (int): Max items to request from STAC.

    Returns:
        list[Any]: List of STAC Items (pystac.Item-like).
    """
    catalog = Client.open(PC_STAC_URL)

    query: Dict[str, Dict[str, float]] = {}
    if cloud_cover_max is not None:
        query["eo:cloud_cover"] = {"lt": float(cloud_cover_max)}

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox.to_list(),
        datetime=datetime_range,
        query=query if query else None,
        limit=limit,
    )

    items = list(search.get_items())

    # Sign the items so asset URLs work (Planetary Computer requires signing).
    signed_items = [pc.sign(item) for item in items]
    return signed_items


def search_month_from_config(
    year: int,
    month: int,
    cloud_cover_max: Optional[float] = 80.0,
    limit: int = 500,
) -> List[Any]:
    """Convenience wrapper: read bbox from configs/aoi_bbox.yaml and search one month.

    Parameters:
        year (int): Year.
        month (int): Month 1-12.
        cloud_cover_max (float, optional): Max eo:cloud_cover percent.
        limit (int): Max items.

    Returns:
        list[Any]: Signed STAC items.
    """
    root = repo_root_from_file(__file__)
    bbox_yaml = root / "configs" / "aoi_bbox.yaml"
    bbox = read_bbox_yaml(bbox_yaml)
    dt = month_datetime_range(year, month)
    return search_sentinel2_l2a_items(
        bbox=bbox,
        datetime_range=dt,
        cloud_cover_max=cloud_cover_max,
        limit=limit,
    )
