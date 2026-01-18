from __future__ import annotations

import argparse
from pathlib import Path
import geopandas as gpd


def pick_layer(gpkg_path: Path, requested_layer: str | None) -> str | None:
    """
    Returns a layer name to read, or None if geopandas should use default.
    - If requested_layer is provided, use it.
    - If not, and there's exactly 1 layer in the GPKG, use it.
    - If multiple layers exist, raise with a helpful message.
    """
    if requested_layer:
        return requested_layer

    layers = gpd.list_layers(gpkg_path)
    names = list(layers["name"])

    if len(names) == 0:
        raise ValueError(f"No layers found in GeoPackage: {gpkg_path}")

    if len(names) == 1:
        return names[0]

    raise ValueError(
        "Multiple layers found in GeoPackage. Re-run with --layer.\n"
        f"Layers: {names}"
    )


def write_bbox_yaml(bbox, out_path: Path) -> None:
    minx, miny, maxx, maxy = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""bbox:
  min_lon: {minx}
  min_lat: {miny}
  max_lon: {maxx}
  max_lat: {maxy}
crs: EPSG:4326
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AOI from GeoPackage to GeoJSON and write STAC-ready bbox YAML."
    )
    parser.add_argument(
        "--gpkg",
        default="data/external/aoi.gpkg",
        help="Path to AOI GeoPackage",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional layer name inside the GeoPackage (ArcGIS Output Name).",
    )
    parser.add_argument(
        "--out-geojson",
        default="data/external/aoi.geojson",
        help="Output GeoJSON path",
    )
    parser.add_argument(
        "--out-bbox-yaml",
        default="configs/aoi_bbox.yaml",
        help="Output bbox YAML path",
    )
    args = parser.parse_args()

    gpkg_path = Path(args.gpkg)
    out_geojson = Path(args.out_geojson)
    out_bbox_yaml = Path(args.out_bbox_yaml)

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    layer = pick_layer(gpkg_path, args.layer)

    # Read AOI
    gdf = gpd.read_file(gpkg_path, layer=layer)

    if gdf.empty:
        raise ValueError(f"Layer '{layer}' read successfully but contains 0 features.")

    if gdf.crs is None:
        raise ValueError("AOI has no CRS defined. Assign CRS in ArcGIS Pro before exporting.")

    # Enforce WGS84 for STAC
    gdf = gdf.to_crs(epsg=4326)

    # Write GeoJSON
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")

    # BBox for STAC: [min_lon, min_lat, max_lon, max_lat]
    bbox = gdf.total_bounds
    write_bbox_yaml(bbox, out_bbox_yaml)

    print("✅ AOI conversion complete")
    print(f"  Input GPKG: {gpkg_path}")
    print(f"  Layer: {layer}")
    print(f"  GeoJSON: {out_geojson}")
    print(f"  BBox YAML: {out_bbox_yaml}")
    print(f"  BBox: {bbox}")


if __name__ == "__main__":
    main()
