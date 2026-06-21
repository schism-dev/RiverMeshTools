"""MPI orchestration and record assembly for marsh decomposition."""

from .config import parse_run_config
from .decomposition import decompose_marsh_polygon, fleshy_paving_resolution
from .geometry import (
    get_input_polygons,
    minimum_rectangle_dimension,
    polygon_boundaries_to_lines,
)
from .output import (
    extract_arc_lines_from_decomposed_gpkg,
    make_output_gdfs,
    print_summary,
    write_output_gpkg,
)
from .skeleton import skeletonize_skinny_polygon


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


def print_run_header(original_gdf, rank, size, recipe_name, run_config):
    """Print run configuration from rank 0."""
    if rank != 0:
        return

    print(f"MPI size = {size}")
    config = run_config.parameters
    print(f"Recipe: {recipe_name}")
    print(f"Input file: {run_config.input_file}")
    print(f"Output file: {run_config.output_file}")
    print(f"CRS: {original_gdf.crs}")
    print(f"Original polygon count: {len(original_gdf)}")
    print()
    print("Parameters:")
    print(f"  use_filter_for_distance = {config.use_filter_for_distance}")
    print(f"  filter_dist = {config.filter_dist:.2f} m")
    print(f"  buffer_join_style = {config.buffer_join_style}")
    print(f"  X2 = {config.X2:.2f} m")
    print(f"  Y  = {config.Y:.2f} m")
    print(f"  Z  = {config.Z:.2f} m")
    print(
        "  skinny_full_width_threshold = "
        f"{config.skinny_full_width_threshold:.2f} m"
    )
    core_dist = config.effective_fleshy_core_dist
    print(f"  fleshy_core_dist = {core_dist:.2f} m")
    print(f"  implied skinny/full-width cutoff ≈ {2 * core_dist:.2f} m")
    print(f"  small_skinny_area_ratio = {config.small_skinny_area_ratio:.4f}")
    print(f"  small_fleshy_area_ratio = {config.small_fleshy_area_ratio:.4f}")
    print(
        "  small_polygon_cleanup_mode = "
        f"{config.small_polygon_cleanup_mode}"
    )
    print(
        "  fleshy_resolution_threshold = "
        f"{config.fleshy_resolution_threshold:.2f} m"
    )
    print(
        "  small_fleshy_resolution_factor = "
        f"{config.small_fleshy_resolution_factor:.2f}"
    )
    print(f"  skeleton_dx = {config.skeleton_dx:.2f} m")
    print(f"  skeleton_random_seed = {config.skeleton_random_seed}")
    print(
        f"  skeleton_vertex_spacing = {config.skeleton_vertex_spacing:.2f} m"
    )
    print()
    print(f"Total polygons: {len(original_gdf)}")
    print(f"Processing with {size} MPI ranks")
    print()


def add_fleshy_records(
    records, parent_id, original_area, fleshy_parts, config
):
    """Append fleshy polygon records and corresponding boundary LineStrings."""
    for part_id, geom in enumerate(fleshy_parts):
        min_dim = minimum_rectangle_dimension(geom)
        paving_res = fleshy_paving_resolution(min_dim, config)

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


def add_simple_polygon_records(
    records,
    key,
    parent_id,
    geoms,
    type_name,
    original_area=None,
):
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
    config,
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
            config=config,
        )
        records["skeleton_lines"].extend(line_records)
        records["skeleton_vertices"].extend(vertex_records)


def process_one_polygon(parent_id, original_poly, rank, config):
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
    ) = decompose_marsh_polygon(original_poly, config)

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
        config=config,
    )

    add_skinny_records_and_skeleton(
        records=records,
        parent_id=parent_id,
        original_area=original_area,
        skinny_parts=skinny_parts,
        config=config,
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


def process_local_polygons(original_gdf, local_indices, rank, config):
    """Process the subset of polygons assigned to one MPI rank."""
    local_records_per_polygon = []

    for parent_id in local_indices:
        row = original_gdf.iloc[parent_id]
        original_poly = row.geometry

        polygon_records = process_one_polygon(
            parent_id=parent_id,
            original_poly=original_poly,
            rank=rank,
            config=config,
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


def main(argv=None):
    resolved = parse_run_config(argv)
    if resolved is None:
        return
    recipe_name, run_config = resolved
    config = run_config.parameters

    # Import MPI only for the executable workflow.  Keeping it out of module
    # initialization allows the geometry functions to be imported by serial
    # tools and regression tests without starting an MPI runtime.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    original_gdf = get_input_polygons(run_config.input_file)

    print_run_header(original_gdf, rank, size, recipe_name, run_config)

    local_indices = list(range(rank, len(original_gdf), size))
    print(f"Rank {rank}: processing {len(local_indices)} polygons")

    local_records = process_local_polygons(
        original_gdf=original_gdf,
        local_indices=local_indices,
        rank=rank,
        config=config,
    )
    records = gather_records(comm=comm, local_records=local_records, rank=rank)

    if rank != 0:
        return

    gdfs = make_output_gdfs(original_gdf, records)

    write_output_gpkg(gdfs, run_config.output_file)

    print_summary(gdfs, run_config.output_file)

    # RiverMapper consumes longitude/latitude arcs.
    extract_arc_lines_from_decomposed_gpkg(
        decomposed_gpkg=run_config.output_file,
        output_file=run_config.output_file.with_name(
            run_config.output_file.stem + "_arc_lines.shp"
        ),
        config=config,
        output_crs="EPSG:4326",
    )

    # Also retain a projected copy for exact overlays and QA in GIS software.
    extract_arc_lines_from_decomposed_gpkg(
        decomposed_gpkg=run_config.output_file,
        output_file=run_config.output_file.with_name(
            run_config.output_file.stem + "_arc_lines_projected.shp"
        ),
        config=config,
        output_crs=None,
    )
