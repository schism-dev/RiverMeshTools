#!/usr/bin/env python3
"""Command-line entry point for marsh polygon decomposition.

The implementation lives in :mod:`RiverMapper.marsh`. Keeping this small
wrapper preserves the historical command and gives a quick copy/paste place
for common runs.

Serial examples::

    python Scripts/marsh_decomposition.py --recipe standard

    python Scripts/marsh_decomposition.py \
        --input /path/to/marsh.shp \
        --output /path/to/marsh_decomposed.gpkg

    python Scripts/marsh_decomposition.py \
        --set min_skinny_width_m=5.0 \
        --set min_fleshy_area_m2=25.0

MPI example::

    mpirun -np 8 python Scripts/marsh_decomposition.py \
        --input /path/to/marsh.shp \
        --output /path/to/marsh_decomposed.gpkg \
        --set min_skinny_width_m=5.0 \
        --set min_fleshy_area_m2=25.0
"""

from RiverMapper.marsh.config import (
    MarshConfig,
    MarshRunConfig,
    make_config,
    parse_run_config,
)
from RiverMapper.marsh.decomposition import (
    DecompositionResult,
    cleanup_direct_discard,
    cleanup_iterative,
    decompose_marsh_polygon,
)
from RiverMapper.marsh.geometry import (
    clean_geom,
    explode_to_lines,
    explode_to_polygons,
    get_input_polygons,
    lines_to_polygons,
    minimum_rectangle_dimension,
    polygon_boundaries_to_lines,
    resample_linestring,
)
from RiverMapper.marsh.output import extract_arc_lines_from_decomposed_gpkg
from RiverMapper.marsh.skeleton import (
    build_skeleton_graph,
    rasterize_polygon_by_point_test,
    skeletonize_skinny_polygon,
    trace_skeleton_branches,
)
from RiverMapper.marsh.workflow import main

__all__ = [
    "DecompositionResult",
    "MarshConfig",
    "MarshRunConfig",
    "build_skeleton_graph",
    "clean_geom",
    "cleanup_direct_discard",
    "cleanup_iterative",
    "decompose_marsh_polygon",
    "explode_to_lines",
    "explode_to_polygons",
    "extract_arc_lines_from_decomposed_gpkg",
    "get_input_polygons",
    "lines_to_polygons",
    "make_config",
    "minimum_rectangle_dimension",
    "parse_run_config",
    "polygon_boundaries_to_lines",
    "rasterize_polygon_by_point_test",
    "resample_linestring",
    "skeletonize_skinny_polygon",
    "trace_skeleton_branches",
]


if __name__ == "__main__":
    main()
