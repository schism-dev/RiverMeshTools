#!/usr/bin/env python3
"""
Decompose marsh polygons into fleshy and skinny regions.

Input:
    /sciclone/schism10/feiye/Marsh/Shapefiles/marsh_test1.shp

Output:
    /sciclone/schism10/feiye/Marsh/Shapefiles/marsh_test1_decomposed.gpkg

Layers:
    original                 - original marsh polygons, after polygonization if needed
    filtered_for_distance    - lightly filtered geometry used only for distance/mask calculation
    fleshy_core              - inward-buffered core computed from filtered_for_distance
    fleshy_mask              - diagnostic fleshy mask
    candidate_skinny_mask    - skinny mask in filtered/reference geometry
    candidate_skinny_raw     - candidate skinny mapped back to original geometry, before merging
    fleshy                   - final fleshy portion of original geometry
    fleshy_boundary_lines    - boundary LineStrings converted from final fleshy polygons
    skinny                   - final skinny portion of original geometry
    skinny_boundary_lines    - boundary LineStrings converted from final skinny polygons
    skinny_skeleton_lines    - detailed 1 m skeleton centerlines from skinny polygons
    skinny_skeleton_vertices - optional 10 m points along skeletons, with D attribute

Important:
    The filtered geometry is used only to calculate the fleshy/skinny mask.
    The final fleshy and skinny outputs are clipped from the original polygon,
    so the original marsh boundary is retained.

    Small skinny pieces whose area is less than a given fraction of the parent
    original polygon are merged back into fleshy.

    Final outputs are reconstructed as:
        final_skinny = retained candidate skinny pieces
        final_fleshy = original - final_skinny

    Skeleton lines are kept at raster resolution, controlled by skeleton_dx.
    RiverMapper can later smooth/bend/generate inner arcs.
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union
from shapely.prepared import prep
from shapely.validation import make_valid


# =============================================================================
# User parameters
# =============================================================================

input_file = Path("/sciclone/schism10/feiye/Marsh/Shapefiles/marsh_bound_line.shp")
output_file = Path("/sciclone/schism10/feiye/Marsh/Shapefiles/marsh_bound_line_decomposed.gpkg")

# Filtering used only for distance/mask calculation:
#     filtered_poly = poly.buffer(+filter_dist).buffer(-filter_dist)
use_filter_for_distance = True
filter_dist = 5.0       # m; light filtering only
filter_min_area = 0.0   # keep diagnostic pieces

# Marsh meshing design parameters
X2 = 10.0
Y = 10.0
Z = 30.0

# Decomposition threshold
skinny_full_width_threshold = 20.0  # m
fleshy_core_dist = 0.5 * skinny_full_width_threshold

small_skinny_area_ratio = 0.001
small_fleshy_area_ratio = 0.01
small_polygon_cleanup_mode = "direct_discard"  # "iterative" or "direct_discard"
# Iterative cleanup:
# Merge very small skinny pieces back into fleshy.
# Example:
#   0.001 = merge skinny pieces smaller than 0.1% of parent polygon area
#   0.0   = keep all candidate skinny pieces
# Merge very small fleshy pieces back into skinny.
max_cleanup_iter = 3
discard_remaining_small_parts = True  # only for small_polygon_cleanup_mode = 'iterative'

# Fleshy paving rule
fleshy_resolution_threshold = 30.0
small_fleshy_resolution_factor = 0.4

# Skeleton extraction
skeleton_dx = 0.2                   # m, raster spacing for medial-axis extraction
skeleton_vertex_spacing = 1.0      # m, optional point spacing along skeleton lines
min_skeleton_line_length = 0.0      # m, discard very short skeleton branches

# Diagnostics
diagnostic_min_area = 0.0

# Buffer style
buffer_join_style = 1  # 1=round, 2=mitre, 3=bevel


# =============================================================================
# Helper functions: geometry
# =============================================================================

def clean_geom(geom):
    """Make a geometry valid and remove empties."""
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    return geom


def explode_to_polygons(geom):
    """Extract Polygon geometries from Polygon/MultiPolygon/GeometryCollection."""
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
    """Extract LineString geometries from LineString/MultiLineString/GeometryCollection."""
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
        raise ValueError("Input CRS is missing. Please assign a projected CRS first.")

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
    """Remove tiny polygon slivers. Use only for optional diagnostic filtering."""
    if min_area <= 0:
        return polys

    return [p for p in polys if p.area >= min_area]


def build_filtered_polygon_for_distance(poly):
    """Build the geometry used only for calculating the fleshy/skinny mask."""
    poly = clean_geom(poly)
    if poly is None:
        return None

    if not use_filter_for_distance or filter_dist <= 0.0:
        return poly

    filtered = poly.buffer(
        filter_dist,
        join_style=buffer_join_style,
    ).buffer(
        -filter_dist,
        join_style=buffer_join_style,
    )

    filtered = clean_geom(filtered)

    if filtered is None or filtered.is_empty:
        return poly

    return filtered


def cleanup_direct_discard(original_poly, candidate_skinny_raw_parts):
    """Directly discard small fleshy/skinny polygons.

    This does not preserve exact original coverage if small pieces are discarded.
    It is useful when tiny slivers are clearly numerical artifacts.
    """
    original_poly = clean_geom(original_poly)
    if original_poly is None:
        return None, None

    original_area = original_poly.area

    small_skinny_area_threshold = small_skinny_area_ratio * original_area
    small_fleshy_area_threshold = small_fleshy_area_ratio * original_area

    large_skinny_parts = [
        p for p in candidate_skinny_raw_parts
        if p.area >= small_skinny_area_threshold
    ]

    if len(large_skinny_parts) == 0:
        skinny = None
        fleshy = original_poly
    else:
        skinny = clean_geom(unary_union(large_skinny_parts))
        fleshy = clean_geom(original_poly.difference(skinny))

    fleshy_parts = explode_to_polygons(fleshy)
    large_fleshy_parts = [
        p for p in fleshy_parts
        if p.area >= small_fleshy_area_threshold
    ]

    small_fleshy_parts = [
        p for p in fleshy_parts
        if p.area < small_fleshy_area_threshold
    ]

    if len(small_fleshy_parts) > 0:
        discarded_area = sum(p.area for p in small_fleshy_parts)
        print(
            f"Warning: direct_discard removed {len(small_fleshy_parts)} "
            f"small fleshy polygons; discarded_area={discarded_area:.6f} m^2 "
            f"({discarded_area / original_area:.6%} of parent polygon)"
        )

    fleshy = clean_geom(unary_union(large_fleshy_parts)) if large_fleshy_parts else None

    skinny_parts = explode_to_polygons(skinny)
    large_skinny_parts = [
        p for p in skinny_parts
        if p.area >= small_skinny_area_threshold
    ]

    small_skinny_parts = [
        p for p in skinny_parts
        if p.area < small_skinny_area_threshold
    ]

    if len(small_skinny_parts) > 0:
        discarded_area = sum(p.area for p in small_skinny_parts)
        print(
            f"Warning: direct_discard removed {len(small_skinny_parts)} "
            f"small skinny polygons; discarded_area={discarded_area:.6f} m^2 "
            f"({discarded_area / original_area:.6%} of parent polygon)"
        )

    skinny = clean_geom(unary_union(large_skinny_parts)) if large_skinny_parts else None

    return fleshy, skinny


def cleanup_iterative(original_poly, candidate_skinny_raw_parts):
    """Iteratively move small polygons to the opposite class.

    Small skinny pieces are merged into fleshy.
    Small fleshy pieces are merged into skinny.

    This preserves original coverage during iteration.
    After max_cleanup_iter, remaining small pieces can optionally be discarded.
    """
    original_poly = clean_geom(original_poly)
    if original_poly is None:
        return None, None

    original_area = original_poly.area

    small_skinny_area_threshold = small_skinny_area_ratio * original_area
    small_fleshy_area_threshold = small_fleshy_area_ratio * original_area

    if len(candidate_skinny_raw_parts) == 0:
        skinny = None
        fleshy = original_poly
    else:
        skinny = clean_geom(unary_union(candidate_skinny_raw_parts))
        fleshy = clean_geom(original_poly.difference(skinny))

    for cleanup_iter in range(max_cleanup_iter):
        changed = False

        skinny_parts = explode_to_polygons(skinny)
        large_skinny_parts = [
            p for p in skinny_parts
            if p.area >= small_skinny_area_threshold
        ]

        if len(large_skinny_parts) != len(skinny_parts):
            changed = True

        if len(large_skinny_parts) == 0:
            skinny = None
            fleshy = original_poly
        else:
            skinny = clean_geom(unary_union(large_skinny_parts))
            fleshy = clean_geom(original_poly.difference(skinny))

        fleshy_parts = explode_to_polygons(fleshy)
        large_fleshy_parts = [
            p for p in fleshy_parts
            if p.area >= small_fleshy_area_threshold
        ]

        if len(large_fleshy_parts) != len(fleshy_parts):
            changed = True

        if len(large_fleshy_parts) == 0:
            fleshy = None
            skinny = original_poly
        else:
            fleshy = clean_geom(unary_union(large_fleshy_parts))
            skinny = clean_geom(original_poly.difference(fleshy))

        if not changed:
            break

    final_fleshy_parts = explode_to_polygons(fleshy)
    final_skinny_parts = explode_to_polygons(skinny)

    remaining_small_fleshy = [
        p for p in final_fleshy_parts
        if p.area < small_fleshy_area_threshold
    ]
    remaining_small_skinny = [
        p for p in final_skinny_parts
        if p.area < small_skinny_area_threshold
    ]

    if len(remaining_small_fleshy) > 0 or len(remaining_small_skinny) > 0:
        print(
            "Warning: small polygons remain after iterative cleanup: "
            f"{len(remaining_small_fleshy)} fleshy, "
            f"{len(remaining_small_skinny)} skinny"
        )

        if discard_remaining_small_parts:
            large_fleshy_parts = [
                p for p in final_fleshy_parts
                if p.area >= small_fleshy_area_threshold
            ]
            large_skinny_parts = [
                p for p in final_skinny_parts
                if p.area >= small_skinny_area_threshold
            ]

            discarded_area = (
                sum(p.area for p in remaining_small_fleshy)
                + sum(p.area for p in remaining_small_skinny)
            )

            print(
                f"Warning: discarded remaining small polygons after "
                f"{max_cleanup_iter} cleanup iterations; "
                f"discarded_area={discarded_area:.6f} m^2 "
                f"({discarded_area / original_area:.6%} of parent polygon)"
            )

            fleshy = clean_geom(unary_union(large_fleshy_parts)) if large_fleshy_parts else None
            skinny = clean_geom(unary_union(large_skinny_parts)) if large_skinny_parts else None

    return fleshy, skinny


# =============================================================================
# Helper functions: fleshy dimension / paving resolution
# =============================================================================

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


def fleshy_paving_resolution(min_dim):
    """Return paving resolution for a fleshy polygon."""
    if not np.isfinite(min_dim):
        return Z

    if min_dim < fleshy_resolution_threshold:
        return small_fleshy_resolution_factor * min_dim

    return Z


# =============================================================================
# Helper functions: skeleton extraction
# =============================================================================

def rasterize_polygon_by_point_test(poly, dx):
    """Rasterize polygon by testing cell centers.

    Returns:
        mask: 2D bool array, True inside polygon
        xs: x coordinate of cell centers
        ys: y coordinate of cell centers
    """
    minx, miny, maxx, maxy = poly.bounds

    # Add one-cell padding to avoid skeleton touching array edges.
    minx -= dx
    miny -= dx
    maxx += dx
    maxy += dx

    xs = np.arange(minx + 0.5 * dx, maxx, dx)
    ys = np.arange(miny + 0.5 * dx, maxy, dx)

    mask = np.zeros((len(ys), len(xs)), dtype=bool)

    prepared = prep(poly)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = Point(float(x), float(y))
            if prepared.contains(p):
                mask[iy, ix] = True

    return mask, xs, ys


def pixel_to_xy(pixel, xs, ys):
    """Convert skeleton pixel index (iy, ix) to x, y."""
    iy, ix = pixel
    return float(xs[ix]), float(ys[iy])


def build_skeleton_graph(skel):
    """Build 8-neighbor graph for skeleton pixels.

    Returns:
        nodes: set of (iy, ix)
        neighbors: dict node -> list of neighboring nodes
    """
    coords = np.argwhere(skel)
    nodes = {tuple(c) for c in coords}

    neighbors = {}
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for node in nodes:
        iy, ix = node
        nbrs = []
        for dy, dx in offsets:
            nb = (iy + dy, ix + dx)
            if nb in nodes:
                nbrs.append(nb)
        neighbors[node] = nbrs

    return nodes, neighbors


def trace_skeleton_branches(skel):
    """Trace skeleton pixels into branch polylines.

    This creates one polyline per branch between graph nodes.
    A graph node is either an endpoint or a junction.
    Remaining all-degree-2 loops are also traced.
    """
    nodes, neighbors = build_skeleton_graph(skel)

    if len(nodes) == 0:
        return []

    degrees = {n: len(neighbors[n]) for n in nodes}
    graph_nodes = {n for n, deg in degrees.items() if deg != 2}

    visited_edges = set()
    branches = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def walk_from(start, nxt):
        path = [start, nxt]
        visited_edges.add(edge_key(start, nxt))

        prev = start
        curr = nxt

        while True:
            if curr in graph_nodes and curr != start:
                break

            candidates = [n for n in neighbors[curr] if n != prev]

            if len(candidates) == 0:
                break

            if len(candidates) > 1 and curr not in graph_nodes:
                break

            nxt2 = candidates[0]
            ek = edge_key(curr, nxt2)

            if ek in visited_edges:
                break

            path.append(nxt2)
            visited_edges.add(ek)

            prev, curr = curr, nxt2

        return path

    # Trace branches from endpoints/junctions.
    for start in graph_nodes:
        for nb in neighbors[start]:
            if edge_key(start, nb) not in visited_edges:
                branch = walk_from(start, nb)
                if len(branch) >= 2:
                    branches.append(branch)

    # Handle loops where every pixel has degree 2.
    for node in nodes:
        for nb in neighbors[node]:
            if edge_key(node, nb) in visited_edges:
                continue

            path = [node, nb]
            visited_edges.add(edge_key(node, nb))

            prev = node
            curr = nb

            while True:
                candidates = [n for n in neighbors[curr] if n != prev]

                if len(candidates) == 0:
                    break

                nxt = candidates[0]
                ek = edge_key(curr, nxt)

                if ek in visited_edges:
                    break

                path.append(nxt)
                visited_edges.add(ek)

                prev, curr = curr, nxt

            if len(path) >= 2:
                branches.append(path)

    return branches


def skeletonize_skinny_polygon(poly, parent_id, skinny_id):
    """Extract detailed skeleton lines and optional 10 m vertices with D.

    The skeleton line geometry is kept at raster resolution, controlled by:

        skeleton_dx = 1.0 m

    This detailed line is intended for RiverMapper, which can later handle
    smoothing, bending, and generation of inner arcs.

    The optional skeleton vertices are sampled every skeleton_vertex_spacing
    meters and store:

        D = distance(vertex, skinny_polygon.boundary)
    """
    try:
        from skimage.morphology import medial_axis
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for skeleton extraction. "
            "Install it or load an environment with skimage available."
        ) from exc

    poly = clean_geom(poly)

    if poly is None or poly.is_empty:
        return [], []

    if poly.area <= 0:
        return [], []

    mask, xs, ys = rasterize_polygon_by_point_test(poly, skeleton_dx)

    if mask.sum() == 0:
        return [], []

    skel = medial_axis(mask)
    branches = trace_skeleton_branches(skel)

    line_records = []
    vertex_records = []

    branch_id = 0

    for branch in branches:
        coords = [pixel_to_xy(pix, xs, ys) for pix in branch]

        if len(coords) < 2:
            continue

        # Keep the detailed raster skeleton as the line geometry.
        # Do not resample/simplify this line to 10 m spacing.
        line_detail = LineString(coords)
        line_detail = clean_geom(line_detail)

        if line_detail is None or line_detail.is_empty:
            continue

        if line_detail.length < min_skeleton_line_length:
            continue

        # Safety clip. Usually this should do little; it prevents occasional
        # corner-cutting outside the skinny polygon.
        line_inside = line_detail.intersection(poly.buffer(1.0e-6))
        line_inside = clean_geom(line_inside)

        line_pieces = explode_to_lines(line_inside)

        for line_piece in line_pieces:
            line_piece = clean_geom(line_piece)

            if line_piece is None or line_piece.is_empty:
                continue

            if line_piece.length < min_skeleton_line_length:
                continue

            # Optional 10 m diagnostic/control vertices along the detailed line.
            length = line_piece.length

            if length <= skeleton_vertex_spacing:
                distances = [0.0, length]
            else:
                nseg = max(1, int(np.ceil(length / skeleton_vertex_spacing)))
                distances = np.linspace(0.0, length, nseg + 1)

            vertex_points = [line_piece.interpolate(float(d)) for d in distances]
            D_values = [float(p.distance(poly.boundary)) for p in vertex_points]

            line_records.append(
                {
                    "parent_id": parent_id,
                    "skinny_id": skinny_id,
                    "branch_id": branch_id,
                    "length_m": float(line_piece.length),
                    "skeleton_dx": float(skeleton_dx),
                    "n_vertices": len(line_piece.coords),
                    "n_vertices_10m": len(vertex_points),
                    "D_min": float(np.min(D_values)) if len(D_values) else np.nan,
                    "D_mean": float(np.mean(D_values)) if len(D_values) else np.nan,
                    "D_max": float(np.max(D_values)) if len(D_values) else np.nan,
                    "geometry": line_piece,
                }
            )

            for vertex_id, p in enumerate(vertex_points):
                vertex_records.append(
                    {
                        "parent_id": parent_id,
                        "skinny_id": skinny_id,
                        "branch_id": branch_id,
                        "vertex_id": vertex_id,
                        "D": float(D_values[vertex_id]),
                        "geometry": p,
                    }
                )

            branch_id += 1

    return line_records, vertex_records


# =============================================================================
# Marsh decomposition
# =============================================================================

def decompose_marsh_polygon(original_poly, fleshy_core_dist):
    """Decompose one marsh polygon into fleshy and skinny parts."""

    original_poly = clean_geom(original_poly)
    if original_poly is None:
        return [], [], [], [], [], [], []

    filtered_poly = build_filtered_polygon_for_distance(original_poly)
    if filtered_poly is None:
        return [], [], [], [], [], [], []

    filtered_parts = explode_to_polygons(filtered_poly)
    filtered_parts = filter_small_polygons(filtered_parts, filter_min_area)

    # 1. Compute fleshy core on the filtered geometry.
    core = filtered_poly.buffer(
        -fleshy_core_dist,
        join_style=buffer_join_style,
    )
    core = clean_geom(core)

    if core is None or core.is_empty:
        fleshy_mask = None
        candidate_skinny_mask = filtered_poly
    else:
        # 2. Expand core back outward and clip to filtered_poly.
        fleshy_mask = core.buffer(
            fleshy_core_dist,
            join_style=buffer_join_style,
        )

        fleshy_mask = clean_geom(fleshy_mask)

        if fleshy_mask is None or fleshy_mask.is_empty:
            candidate_skinny_mask = filtered_poly
        else:
            fleshy_mask = fleshy_mask.intersection(filtered_poly)
            fleshy_mask = clean_geom(fleshy_mask)

            candidate_skinny_mask = filtered_poly.difference(fleshy_mask)
            candidate_skinny_mask = clean_geom(candidate_skinny_mask)

    # 3. Map candidate skinny mask back to original geometry.
    if candidate_skinny_mask is None or candidate_skinny_mask.is_empty:
        candidate_skinny_raw = None
    else:
        candidate_skinny_raw = original_poly.intersection(candidate_skinny_mask)
        candidate_skinny_raw = clean_geom(candidate_skinny_raw)

    candidate_skinny_raw_parts = explode_to_polygons(candidate_skinny_raw)

    # 4. Clean tiny fleshy/skinny pieces.
    if small_polygon_cleanup_mode == "iterative":
        final_fleshy, final_skinny = cleanup_iterative(
            original_poly,
            candidate_skinny_raw_parts,
        )
    elif small_polygon_cleanup_mode == "direct_discard":
        final_fleshy, final_skinny = cleanup_direct_discard(
            original_poly,
            candidate_skinny_raw_parts,
        )
    else:
        raise ValueError(
            f"Unknown small_polygon_cleanup_mode: {small_polygon_cleanup_mode}. "
            "Use 'iterative' or 'direct_discard'."
        )

    fleshy_parts = explode_to_polygons(final_fleshy)
    skinny_parts = explode_to_polygons(final_skinny)
    core_parts = explode_to_polygons(core)
    fleshy_mask_parts = explode_to_polygons(fleshy_mask)
    candidate_skinny_mask_parts = explode_to_polygons(candidate_skinny_mask)

    core_parts = filter_small_polygons(core_parts, diagnostic_min_area)
    fleshy_mask_parts = filter_small_polygons(fleshy_mask_parts, diagnostic_min_area)
    candidate_skinny_mask_parts = filter_small_polygons(
        candidate_skinny_mask_parts,
        diagnostic_min_area,
    )
    candidate_skinny_raw_parts = filter_small_polygons(
        candidate_skinny_raw_parts,
        diagnostic_min_area,
    )

    return (
        fleshy_parts,
        skinny_parts,
        core_parts,
        filtered_parts,
        fleshy_mask_parts,
        candidate_skinny_mask_parts,
        candidate_skinny_raw_parts,
    )


# =============================================================================
# Main workflow
# =============================================================================
# =============================================================================
# MPI / output helper functions
# =============================================================================

LAYER_KEYS = [
    "fleshy",
    "fleshy_boundary_lines",
    "skinny",
    "skinny_boundary_lines",
    "core",
    "filtered",
    "fleshy_mask",
    "candidate_skinny_mask",
    "candidate_skinny_raw",
    "skeleton_lines",
    "skeleton_vertices",
    "log_lines",
]


def make_empty_records():
    """Create a dictionary of output record lists."""
    return {key: [] for key in LAYER_KEYS}


def sort_key(record):
    """Stable sorting key for reproducible output."""
    return (
        record.get("parent_id", -1),
        record.get("skinny_id", -1),
        record.get("fleshy_id", -1),
        record.get("part_id", -1),
        record.get("branch_id", -1),
        record.get("vertex_id", -1),
    )


def sort_record_dict(records):
    """Sort all geometry record lists in place."""
    for key, value in records.items():
        if key == "log_lines":
            continue
        value.sort(key=sort_key)


def print_run_header(original_gdf, rank, size):
    """Print run configuration from rank 0."""
    if rank != 0:
        return

    print(f"MPI size = {size}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"CRS: {original_gdf.crs}")
    print(f"Original polygon count: {len(original_gdf)}")
    print()
    print("Parameters:")
    print(f"  use_filter_for_distance = {use_filter_for_distance}")
    print(f"  filter_dist = {filter_dist:.2f} m")
    print(f"  buffer_join_style = {buffer_join_style}")
    print(f"  X2 = {X2:.2f} m")
    print(f"  Y  = {Y:.2f} m")
    print(f"  Z  = {Z:.2f} m")
    print(f"  skinny_full_width_threshold = {skinny_full_width_threshold:.2f} m")
    print(f"  fleshy_core_dist = {fleshy_core_dist:.2f} m")
    print(f"  implied skinny/full-width cutoff ≈ {2 * fleshy_core_dist:.2f} m")
    print(f"  small_skinny_area_ratio = {small_skinny_area_ratio:.4f}")
    print(f"  fleshy_resolution_threshold = {fleshy_resolution_threshold:.2f} m")
    print(f"  small_fleshy_resolution_factor = {small_fleshy_resolution_factor:.2f}")
    print(f"  skeleton_dx = {skeleton_dx:.2f} m")
    print(f"  skeleton_vertex_spacing = {skeleton_vertex_spacing:.2f} m")
    print()
    print(f"Total polygons: {len(original_gdf)}")
    print(f"Processing with {size} MPI ranks")
    print()


def add_fleshy_records(records, parent_id, original_area, fleshy_parts):
    """Append fleshy polygon records and corresponding boundary LineStrings."""
    for part_id, geom in enumerate(fleshy_parts):
        min_dim = minimum_rectangle_dimension(geom)
        paving_res = fleshy_paving_resolution(min_dim)

        records["fleshy"].append(
            {
                "parent_id": parent_id,
                "part_id": part_id,
                "type": "fleshy",
                "area_m2": geom.area,
                "area_ratio_parent": geom.area / original_area,
                "min_dimension_m": min_dim,
                "paving_res_m": paving_res,
                "geometry": geom,
            }
        )

        for line_id, line in enumerate(polygon_boundaries_to_lines(geom)):
            records["fleshy_boundary_lines"].append(
                {
                    "parent_id": parent_id,
                    "fleshy_id": part_id,
                    "line_id": line_id,
                    "type": "fleshy_boundary",
                    "area_m2": geom.area,
                    "area_ratio_parent": geom.area / original_area,
                    "min_dimension_m": min_dim,
                    "paving_res_m": paving_res,
                    "geometry": line,
                }
            )


def add_simple_polygon_records(records, key, parent_id, geoms, type_name, original_area=None):
    """Append simple polygon records for diagnostic layers."""
    for part_id, geom in enumerate(geoms):
        rec = {
            "parent_id": parent_id,
            "part_id": part_id,
            "type": type_name,
            "area_m2": geom.area,
            "geometry": geom,
        }

        if original_area is not None:
            rec["area_ratio_parent"] = geom.area / original_area

        records[key].append(rec)


def add_skinny_records_and_skeleton(
    records,
    parent_id,
    original_area,
    skinny_parts,
):
    """Append skinny polygon records and optional skeleton outputs."""
    for skinny_id, geom in enumerate(skinny_parts):
        records["skinny"].append(
            {
                "parent_id": parent_id,
                "part_id": skinny_id,
                "type": "skinny",
                "area_m2": geom.area,
                "area_ratio_parent": geom.area / original_area,
                "geometry": geom,
            }
        )

        for line_id, line in enumerate(polygon_boundaries_to_lines(geom)):
            records["skinny_boundary_lines"].append(
                {
                    "parent_id": parent_id,
                    "skinny_id": skinny_id,
                    "line_id": line_id,
                    "type": "skinny_boundary",
                    "area_m2": geom.area,
                    "area_ratio_parent": geom.area / original_area,
                    "geometry": line,
                }
            )

        line_records, vertex_records = skeletonize_skinny_polygon(
            geom,
            parent_id=parent_id,
            skinny_id=skinny_id,
        )
        records["skeleton_lines"].extend(line_records)
        records["skeleton_vertices"].extend(vertex_records)


def process_one_polygon(parent_id, original_poly, rank):
    """Process one marsh polygon and return record lists."""
    records = make_empty_records()

    original_area = original_poly.area

    (
        fleshy_parts,
        skinny_parts,
        core_parts,
        filtered_parts,
        fleshy_mask_parts,
        candidate_skinny_mask_parts,
        candidate_skinny_raw_parts,
    ) = decompose_marsh_polygon(
        original_poly,
        fleshy_core_dist=fleshy_core_dist,
    )

    raw_skinny_area = sum(g.area for g in candidate_skinny_raw_parts)
    mask_skinny_area = sum(g.area for g in candidate_skinny_mask_parts)
    retained_skinny_area = sum(g.area for g in skinny_parts)

    records["log_lines"].append(
        f"Rank {rank}, Parent {parent_id}: original_area={original_area:.3f}, "
        f"candidate_skinny_mask_area={mask_skinny_area:.3f}, "
        f"candidate_skinny_raw_area={raw_skinny_area:.3f} "
        f"({raw_skinny_area / original_area:.4%}), "
        f"retained_skinny_area={retained_skinny_area:.3f} "
        f"({retained_skinny_area / original_area:.4%})"
    )

    add_fleshy_records(
        records=records,
        parent_id=parent_id,
        original_area=original_area,
        fleshy_parts=fleshy_parts,
    )

    add_skinny_records_and_skeleton(
        records=records,
        parent_id=parent_id,
        original_area=original_area,
        skinny_parts=skinny_parts,
    )

    add_simple_polygon_records(
        records,
        key="core",
        parent_id=parent_id,
        geoms=core_parts,
        type_name="fleshy_core",
    )

    add_simple_polygon_records(
        records,
        key="filtered",
        parent_id=parent_id,
        geoms=filtered_parts,
        type_name="filtered_for_distance",
    )

    add_simple_polygon_records(
        records,
        key="fleshy_mask",
        parent_id=parent_id,
        geoms=fleshy_mask_parts,
        type_name="fleshy_mask",
    )

    add_simple_polygon_records(
        records,
        key="candidate_skinny_mask",
        parent_id=parent_id,
        geoms=candidate_skinny_mask_parts,
        type_name="candidate_skinny_mask",
    )

    add_simple_polygon_records(
        records,
        key="candidate_skinny_raw",
        parent_id=parent_id,
        geoms=candidate_skinny_raw_parts,
        type_name="candidate_skinny_raw",
        original_area=original_area,
    )

    return records


def merge_record_dicts(record_dicts):
    """Merge a list of record dictionaries into one dictionary."""
    merged = make_empty_records()

    for recs in record_dicts:
        for key in LAYER_KEYS:
            merged[key].extend(recs[key])

    return merged


def process_local_polygons(original_gdf, local_indices, rank):
    """Process the subset of polygons assigned to one MPI rank."""
    local_records_per_polygon = []

    for parent_id in local_indices:
        row = original_gdf.iloc[parent_id]
        original_poly = row.geometry

        polygon_records = process_one_polygon(
            parent_id=parent_id,
            original_poly=original_poly,
            rank=rank,
        )

        local_records_per_polygon.append(polygon_records)

    return merge_record_dicts(local_records_per_polygon)


def gather_records(comm, local_records, rank):
    """Gather local records to rank 0 and flatten."""
    gathered = comm.gather(local_records, root=0)

    if rank != 0:
        return None

    records = merge_record_dicts(gathered)
    sort_record_dict(records)

    for rank_records in gathered:
        for line in rank_records["log_lines"]:
            print(line)

    return records


def make_output_gdfs(original_gdf, records):
    """Convert record lists into GeoDataFrames."""
    original_out_gdf = original_gdf.copy()
    original_out_gdf["type"] = "original"
    original_out_gdf["area_m2"] = original_out_gdf.geometry.area

    gdfs = {
        "original": original_out_gdf,
        "filtered_for_distance": gpd.GeoDataFrame(
            records["filtered"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "fleshy_core": gpd.GeoDataFrame(
            records["core"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "fleshy_mask": gpd.GeoDataFrame(
            records["fleshy_mask"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "candidate_skinny_mask": gpd.GeoDataFrame(
            records["candidate_skinny_mask"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "candidate_skinny_raw": gpd.GeoDataFrame(
            records["candidate_skinny_raw"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "fleshy": gpd.GeoDataFrame(
            records["fleshy"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "fleshy_boundary_lines": gpd.GeoDataFrame(
            records["fleshy_boundary_lines"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "skinny": gpd.GeoDataFrame(
            records["skinny"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "skinny_boundary_lines": gpd.GeoDataFrame(
            records["skinny_boundary_lines"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "skinny_skeleton_lines": gpd.GeoDataFrame(
            records["skeleton_lines"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
        "skinny_skeleton_vertices": gpd.GeoDataFrame(
            records["skeleton_vertices"],
            geometry="geometry",
            crs=original_gdf.crs,
        ),
    }

    return gdfs


def write_output_gpkg(gdfs):
    """Write nonempty GeoDataFrames to output GeoPackage."""
    if output_file.exists():
        output_file.unlink()

    # Always write original.
    gdfs["original"].to_file(output_file, layer="original", driver="GPKG")

    for layer_name, gdf in gdfs.items():
        if layer_name == "original":
            continue

        if len(gdf) > 0:
            gdf.to_file(output_file, layer=layer_name, driver="GPKG")


def layer_area(gdf):
    """Return total area for polygon layers; zero for empty layers."""
    if len(gdf) == 0:
        return 0.0

    return gdf.geometry.area.sum()


def print_summary(gdfs):
    """Print area/count summary and QA diagnostics."""
    original_gdf = gdfs["original"]
    filtered_gdf = gdfs["filtered_for_distance"]
    core_gdf = gdfs["fleshy_core"]
    fleshy_mask_gdf = gdfs["fleshy_mask"]
    candidate_skinny_mask_gdf = gdfs["candidate_skinny_mask"]
    candidate_skinny_raw_gdf = gdfs["candidate_skinny_raw"]
    fleshy_gdf = gdfs["fleshy"]
    fleshy_boundary_lines_gdf = gdfs["fleshy_boundary_lines"]
    skinny_gdf = gdfs["skinny"]
    skinny_boundary_lines_gdf = gdfs["skinny_boundary_lines"]
    skeleton_lines_gdf = gdfs["skinny_skeleton_lines"]
    skeleton_vertices_gdf = gdfs["skinny_skeleton_vertices"]

    original_area = layer_area(original_gdf)
    fleshy_area = layer_area(fleshy_gdf)
    skinny_area = layer_area(skinny_gdf)
    reconstructed_area = fleshy_area + skinny_area

    summary = pd.DataFrame(
        {
            "layer": [
                "original",
                "filtered_for_distance",
                "fleshy_core",
                "fleshy_mask",
                "candidate_skinny_mask",
                "candidate_skinny_raw",
                "fleshy",
                "fleshy_boundary_lines",
                "skinny",
                "skinny_boundary_lines",
                "skinny_skeleton_lines",
                "skinny_skeleton_vertices",
                "fleshy+skinny",
            ],
            "count": [
                len(original_gdf),
                len(filtered_gdf),
                len(core_gdf),
                len(fleshy_mask_gdf),
                len(candidate_skinny_mask_gdf),
                len(candidate_skinny_raw_gdf),
                len(fleshy_gdf),
                len(fleshy_boundary_lines_gdf),
                len(skinny_gdf),
                len(skinny_boundary_lines_gdf),
                len(skeleton_lines_gdf),
                len(skeleton_vertices_gdf),
                len(fleshy_gdf) + len(skinny_gdf),
            ],
            "area_m2": [
                original_area,
                layer_area(filtered_gdf),
                layer_area(core_gdf),
                layer_area(fleshy_mask_gdf),
                layer_area(candidate_skinny_mask_gdf),
                layer_area(candidate_skinny_raw_gdf),
                fleshy_area,
                fleshy_area,
                skinny_area,
                skinny_area,
                np.nan,
                np.nan,
                reconstructed_area,
            ],
        }
    )

    print()
    print(f"Saved: {output_file}")
    print()
    print(summary.to_string(index=False))
    print()

    if len(fleshy_gdf) > 0:
        print("Fleshy polygon dimension / paving summary:")
        cols = ["parent_id", "part_id", "area_m2", "min_dimension_m", "paving_res_m"]
        print(fleshy_gdf[cols].to_string(index=False))
        print()

    area_error = reconstructed_area - original_area
    rel_error = area_error / original_area if original_area > 0 else 0.0

    print("Area reconstruction check, relative to the original geometry:")
    print(f"  fleshy + skinny - original = {area_error:.6f} m^2")
    print(f"  relative error = {rel_error:.6e}")
    print()
    print("Note:")
    print("  candidate_skinny_mask is computed in filtered/reference geometry.")
    print("  candidate_skinny_raw is candidate skinny mapped back to original geometry.")
    print("  final fleshy and skinny are exact complements clipped from the original geometry.")
    print("  fleshy_boundary_lines is converted from final fleshy polygon boundaries.")
    print("  skinny_boundary_lines is converted from final skinny polygon boundaries.")
    print("  skeleton line geometry is detailed medial axis at skeleton_dx.")
    print("  skeleton vertex D = distance to nearest skinny polygon boundary.")


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


def extract_arc_lines_from_decomposed_gpkg(
    decomposed_gpkg,
    output_file,
    output_crs="EPSG:4326",
):
    """
    Extract arc lines from decomposed marsh GPKG.

    Input layers are assumed to already be LineString/MultiLineString:
        fleshy_boundary_lines      -> arc_pos = "regular",   dummy = 0
        skinny_boundary_lines      -> arc_pos = "left half", dummy = 0
        skinny_skeleton_lines      -> arc_pos = "dummy",     dummy = 1

    skinny_skeleton_lines are resampled with spacing Y in the source projected CRS.

    All available attributes from source layers are preserved.
    Output is reprojected to output_crs before writing.
    """

    decomposed_gpkg = Path(decomposed_gpkg)
    output_file = Path(output_file)

    layer_specs = [
        {
            "layer": "fleshy_boundary_lines",
            "arc_pos": "regular",
            "dummy": 0,
            "resample": 0.2 * Y,
        },
        {
            "layer": "skinny_boundary_lines",
            "arc_pos": "left half",
            "dummy": 0,
            "resample": 0.2 * Y,
        },
        {
            "layer": "skinny_skeleton_lines",
            "arc_pos": "dummy",
            "dummy": 1,
            "resample": Y,
        },
    ]

    def read_layer(layer):
        try:
            return gpd.read_file(decomposed_gpkg, layer=layer)
        except Exception:
            return None

    records = []
    crs = None

    for spec in layer_specs:
        layer_name = spec["layer"]
        arc_pos = spec["arc_pos"]
        dummy = spec["dummy"]
        resample = spec["resample"]
        resample = resample if resample is not None and resample > 0 else None

        gdf = read_layer(layer_name)

        if gdf is None or gdf.empty:
            print(f"Warning: missing or empty layer: {layer_name}")
            continue

        if crs is None:
            crs = gdf.crs

        for _, row in gdf.iterrows():
            attrs = row.drop(labels="geometry").to_dict()

            for line in explode_to_lines(row.geometry):
                line = clean_geom(line)

                if line is None or line.is_empty or line.length <= 0:
                    continue

                if resample is not None:
                    line = resample_linestring(line, resample)
                    line = clean_geom(line)

                    if line is None or line.is_empty or line.length <= 0:
                        continue

                rec = attrs.copy()
                rec["src_layer"] = layer_name
                rec["arc_pos"] = arc_pos
                rec["dummy"] = dummy
                rec["length_m"] = line.length
                rec["resampled"] = bool(resample) if resample is not None else False
                rec["resamp_m"] = float(resample) if resample is not None else np.nan
                rec["geometry"] = line

                records.append(rec)

    if crs is None:
        raise ValueError("No valid input LineString layers found.")

    if len(records) == 0:
        raise ValueError("No line features generated.")

    out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    if output_crs is not None:
        out_gdf = out_gdf.to_crs(output_crs)

    if output_file.exists():
        if output_file.suffix.lower() == ".shp":
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                f = output_file.with_suffix(suffix)
                if f.exists():
                    f.unlink()
        else:
            output_file.unlink()

    if output_file.suffix.lower() == ".shp":
        out_gdf.to_file(output_file, driver="ESRI Shapefile")
    else:
        out_gdf.to_file(output_file, layer="arc_lines", driver="GPKG")

    print(f"Saved: {output_file}")
    print(f"Output CRS: {out_gdf.crs}")
    print(out_gdf[["src_layer", "arc_pos", "dummy", "resampled"]].value_counts())

    return out_gdf


def main():
    # Import MPI only for the executable workflow.  Keeping it out of module
    # initialization allows the geometry functions to be imported by serial
    # tools and regression tests without starting an MPI runtime.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    original_gdf = get_input_polygons(input_file)

    print_run_header(original_gdf, rank, size)

    local_indices = list(range(rank, len(original_gdf), size))
    print(f"Rank {rank}: processing {len(local_indices)} polygons")

    local_records = process_local_polygons(
        original_gdf=original_gdf, local_indices=local_indices, rank=rank)
    records = gather_records(comm=comm, local_records=local_records, rank=rank)

    if rank != 0:
        return

    gdfs = make_output_gdfs(original_gdf, records)

    write_output_gpkg(gdfs)

    print_summary(gdfs)

    # extract arc lines for RiverMapper from the decomposed GPKG output, if it exists
    extract_arc_lines_from_decomposed_gpkg(
        decomposed_gpkg=output_file,
        output_file=output_file.with_name(output_file.stem + "_arc_lines.shp"),
        output_crs="EPSG:4326",
    )


if __name__ == "__main__":
    main()
