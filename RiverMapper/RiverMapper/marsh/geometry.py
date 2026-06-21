"""General-purpose geometry and input helpers for marsh decomposition."""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid


def clean_geom(geom):
    """Make a geometry valid and remove empties."""
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    return geom


def explode_to_polygons(geom):
    """Extract polygons from polygonal or collection geometry."""
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            polys.extend(explode_to_polygons(g))
        return polys

    return []


def explode_to_lines(geom):
    """Extract lines from linear or collection geometry."""
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "LineString":
        return [geom]

    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        lines = []
        for g in geom.geoms:
            lines.extend(explode_to_lines(g))
        return lines

    return []


def lines_to_polygons(gdf_line):
    """Polygonize closed or connected LineString/MultiLineString features."""
    line_union = unary_union(gdf_line.geometry)
    polys = list(polygonize(line_union))

    if len(polys) == 0:
        raise ValueError(
            "No polygons could be generated from the input line layer. "
            "Check whether the marsh boundary lines are closed."
        )

    return gpd.GeoDataFrame(
        {"source": ["polygonized_line"] * len(polys)},
        geometry=polys,
        crs=gdf_line.crs,
    )


def polygon_boundaries_to_lines(geom):
    """Convert Polygon/MultiPolygon boundaries to individual LineStrings.

    For each Polygon:
        - exterior ring becomes one LineString
        - each interior hole ring becomes one LineString
    """
    if geom is None or geom.is_empty:
        return []

    lines = []

    if geom.geom_type == "Polygon":
        lines.append(LineString(geom.exterior.coords))

        for ring in geom.interiors:
            lines.append(LineString(ring.coords))

    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            lines.extend(polygon_boundaries_to_lines(poly))

    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            lines.extend(polygon_boundaries_to_lines(g))

    return lines


def get_input_polygons(input_file):
    """Read input file and return a polygon GeoDataFrame."""
    gdf = gpd.read_file(input_file)

    if gdf.empty:
        raise ValueError(f"Input file is empty: {input_file}")

    if gdf.crs is None:
        raise ValueError(
            "Input CRS is missing. Please assign a projected CRS first."
        )
    if not gdf.crs.is_projected:
        raise ValueError(
            "Input CRS must be projected because parameters are in metres; "
            f"got {gdf.crs}"
        )

    geom_types = set(gdf.geometry.geom_type.unique())

    line_types = {"LineString", "MultiLineString"}
    polygon_types = {"Polygon", "MultiPolygon"}

    if geom_types <= polygon_types:
        poly_gdf = gdf.copy()
        if "source" not in poly_gdf.columns:
            poly_gdf["source"] = "input_polygon"

    elif geom_types <= line_types:
        poly_gdf = lines_to_polygons(gdf)

    else:
        raise ValueError(f"Unsupported geometry types: {geom_types}")

    poly_gdf["geometry"] = poly_gdf.geometry.apply(clean_geom)
    poly_gdf = poly_gdf[
        poly_gdf.geometry.notnull() & ~poly_gdf.geometry.is_empty
    ].copy()

    poly_gdf = poly_gdf.explode(index_parts=False).reset_index(drop=True)

    rows = []
    for i, row in poly_gdf.iterrows():
        for j, poly in enumerate(explode_to_polygons(row.geometry)):
            rows.append(
                {
                    "parent_input_id": i,
                    "part_id": j,
                    "source": row.get("source", "unknown"),
                    "area_m2": poly.area,
                    "geometry": poly,
                }
            )

    if len(rows) == 0:
        raise ValueError("No valid polygon geometries found after processing.")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=poly_gdf.crs)


def filter_small_polygons(polys, min_area):
    """Remove tiny slivers during optional diagnostic filtering."""
    if min_area <= 0:
        return polys

    return [p for p in polys if p.area >= min_area]


def minimum_rectangle_dimension(poly):
    """Approximate minimum dimension using minimum rotated rectangle.

    Returns the smaller side length of the minimum rotated rectangle.
    """
    poly = clean_geom(poly)
    if poly is None:
        return np.nan

    rect = poly.minimum_rotated_rectangle

    if rect is None or rect.is_empty or rect.geom_type != "Polygon":
        return np.nan

    coords = list(rect.exterior.coords)

    if len(coords) < 5:
        return np.nan

    side_lengths = []
    for i in range(4):
        p0 = np.array(coords[i])
        p1 = np.array(coords[i + 1])
        side_lengths.append(float(np.linalg.norm(p1 - p0)))

    side_lengths = [s for s in side_lengths if s > 1.0e-8]

    if len(side_lengths) == 0:
        return np.nan

    return min(side_lengths)


def resample_linestring(line, spacing):
    """Resample a LineString to approximately uniform spacing.

    Always rebuilds the LineString.
    If line.length <= spacing, keeps only start and end points.
    """
    line = clean_geom(line)

    if line is None or line.is_empty:
        return None

    if line.geom_type != "LineString":
        return line

    length = line.length

    if length <= 0.0:
        return None

    if length <= spacing:
        distances = [0.0, length]
    else:
        distances = list(np.arange(0.0, length, spacing))

        if distances[-1] < length:
            distances.append(length)

    points = [line.interpolate(float(d)) for d in distances]

    # Remove duplicated consecutive points, just in case.
    coords = []
    for p in points:
        xy = (p.x, p.y)
        if len(coords) == 0 or xy != coords[-1]:
            coords.append(xy)

    if len(coords) < 2:
        return None

    return LineString(coords)
