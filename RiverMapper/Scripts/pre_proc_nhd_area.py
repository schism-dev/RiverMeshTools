"""
Build a clean polygon mask from a shapefile by removing overlaps, gaps, slivers,

In QGIS, this can be done by "Delete duplicate Geometries" + "Buffer" (positive then negative),
which seems faster than python script below.
"""

from pathlib import Path
import geopandas as gpd

TARGET_CRS = "esri:102008"  # North America Albers Equal Area Conic
PRECISION = 3  # number of decimal places to keep when snapping coordinates (in target CRS)
GAP = 0.1  # meters; gap size for morphological closing to clean up slivers and gaps


def build_clean_polygon_mask(poly_shp):
    polys = gpd.read_file(poly_shp)

    # 0) Reproject to a metric CRS (so GAP is in meters)
    polys = polys.to_crs(TARGET_CRS)

    # 1) Optional: snap coordinates to a small grid to collapse near-coincident edges
    try:
        from shapely import set_precision  # shapely>=2.0
        polys["geometry"] = polys.geometry.apply(lambda g: set_precision(g, PRECISION))
    except Exception:
        pass

    # 2) Repair invalid geometries
    polys["geometry"] = polys.buffer(0)

    # 3) Dissolve to a single geometry (cascaded union) — removes overlaps
    mask = polys.dissolve().geometry.iloc[0]

    # 4) Morphological closing to remove slivers/gaps:
    #    - buffer(+GAP): dilate, bridges gaps narrower than ~2*GAP
    #    - buffer(-GAP): erode back, keeps boundary movement ~<= GAP
    mask_closed = gpd.GeoSeries(mask, crs=polys.crs).buffer(+GAP).buffer(-GAP).iloc[0]

    # 5) Final repair (just in case)
    mask_closed = gpd.GeoSeries(mask_closed, crs=polys.crs).buffer(0).iloc[0]

    # Return as a one-row GeoDataFrame (handy for clip/difference)
    return gpd.GeoDataFrame(geometry=[mask_closed], crs=polys.crs)


if __name__ == "__main__":
    poly_shapefile = Path(
        "/sciclone/schism10/Hgrid_projects/STOFS3D-v8/a51_RiverMapper/Shapefiles/"
        "nhdarea_la_ms.shp"
    )
    mask_gdf = build_clean_polygon_mask(poly_shapefile)
    mask_gdf.to_file(
        poly_shapefile.parent / f"{poly_shapefile.stem}_cleaned.shp",
        driver="ESRI Shapefile"
    )
