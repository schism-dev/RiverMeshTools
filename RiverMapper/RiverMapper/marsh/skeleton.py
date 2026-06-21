"""Raster medial-axis extraction for skinny marsh polygons."""

import numpy as np
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from .geometry import clean_geom, explode_to_lines


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
    for start in sorted(graph_nodes):
        for nb in sorted(neighbors[start]):
            if edge_key(start, nb) not in visited_edges:
                branch = walk_from(start, nb)
                if len(branch) >= 2:
                    branches.append(branch)

    # Handle loops where every pixel has degree 2.
    for node in sorted(nodes):
        for nb in sorted(neighbors[node]):
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


def skeletonize_skinny_polygon(poly, parent_id, skinny_id, config):
    """Extract detailed skeleton lines and optional 10 m vertices with D.

    The skeleton line geometry is kept at raster resolution, controlled by:

        config.skeleton_dx = 1.0 m

    This detailed line is intended for RiverMapper, which can later handle
    smoothing, bending, and generation of inner arcs.

    Optional skeleton vertices use ``config.skeleton_vertex_spacing`` and
    store:

        D = distance(vertex, skinny_polygon.boundary)
    """
    try:
        from skimage.morphology import medial_axis
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for skeleton extraction. "
            "Install RiverMapper with 'pip install RiverMapper[marsh]' "
            "or load an environment with skimage available."
        ) from exc

    poly = clean_geom(poly)

    if poly is None or poly.is_empty:
        return [], []

    if poly.area <= 0:
        return [], []

    mask, xs, ys = rasterize_polygon_by_point_test(poly, config.skeleton_dx)

    if mask.sum() == 0:
        return [], []

    # medial_axis uses randomness to resolve equal-distance pixel ties.
    # A fixed seed makes centerline geometry and branch IDs reproducible.
    skel = medial_axis(mask, rng=config.skeleton_random_seed)
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

        if line_detail.length < config.min_skeleton_line_length:
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

            if line_piece.length < config.min_skeleton_line_length:
                continue

            # Optional diagnostic/control vertices along the detailed line.
            length = line_piece.length

            if length <= config.skeleton_vertex_spacing:
                distances = [0.0, length]
            else:
                nseg = max(
                    1,
                    int(np.ceil(length / config.skeleton_vertex_spacing)),
                )
                distances = np.linspace(0.0, length, nseg + 1)

            vertex_points = [
                line_piece.interpolate(float(distance))
                for distance in distances
            ]
            d_values = [
                float(point.distance(poly.boundary))
                for point in vertex_points
            ]

            line_records.append(
                {
                    "parent_id": parent_id,
                    "skinny_id": skinny_id,
                    "branch_id": branch_id,
                    "length_m": float(line_piece.length),
                    "skeleton_dx": float(config.skeleton_dx),
                    "n_vertices": len(line_piece.coords),
                    "n_vertices_10m": len(vertex_points),
                    "D_min": float(np.min(d_values)) if d_values else np.nan,
                    "D_mean": float(np.mean(d_values)) if d_values else np.nan,
                    "D_max": float(np.max(d_values)) if d_values else np.nan,
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
                        "D": float(d_values[vertex_id]),
                        "geometry": p,
                    }
                )

            branch_id += 1

    return line_records, vertex_records
