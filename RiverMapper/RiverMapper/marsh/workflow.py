"""MPI orchestration and record assembly for marsh decomposition."""

import math
from pathlib import Path
import pickle
import time

from .config import parse_run_config
from .decomposition import decompose_marsh_polygon, fleshy_paving_resolution
from .geometry import (
    get_input_polygons,
    minimum_rectangle_dimension,
    polygon_boundaries_to_lines,
)
from .output import (
    make_arc_line_records,
    make_output_gdfs,
    print_summary,
    write_arc_line_records,
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
    print(
        "  boundary_buffer_distance = "
        f"{config.boundary_buffer_distance:.2f} m"
    )
    print(
        "  along_boundary_resolution = "
        f"{config.along_boundary_resolution:.2f} m"
    )
    print(
        "  skinny_centerline_spacing = "
        f"{config.skinny_centerline_spacing:.2f} m"
    )
    print(
        "  default_fleshy_paving_resolution = "
        f"{config.default_fleshy_paving_resolution:.2f} m"
    )
    print(
        "  boundary_vertex_spacing = "
        f"{config.boundary_vertex_spacing:.2f} m"
    )
    print(
        "  skinny_full_width_threshold = "
        f"{config.skinny_full_width_threshold:.2f} m"
    )
    core_dist = config.effective_fleshy_core_dist
    print(f"  fleshy_core_dist = {core_dist:.2f} m")
    print(f"  implied skinny/full-width cutoff ≈ {2 * core_dist:.2f} m")
    print(f"  small_skinny_area_ratio = {config.small_skinny_area_ratio:.4f}")
    print(f"  small_fleshy_area_ratio = {config.small_fleshy_area_ratio:.4f}")
    print(f"  min_skinny_width_m = {config.min_skinny_width_m}")
    print(f"  min_fleshy_area_m2 = {config.min_fleshy_area_m2}")
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
                    "buffer_m": config.boundary_buffer_distance,
                    "along_m": config.along_boundary_resolution,
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
    """Append skinny polygon records and optional skeleton outputs.

    Returns records for skinny polygons dropped by the optional average-width
    final-product filter.
    """
    dropped_skinny_parts = []

    for skinny_id, geom in enumerate(skinny_parts):
        min_width = minimum_rectangle_dimension(geom)
        line_records, vertex_records = skeletonize_skinny_polygon(
            geom,
            parent_id=parent_id,
            skinny_id=skinny_id,
            config=config,
        )
        avg_width = average_skeleton_width(line_records, min_width)

        if (
            config.min_skinny_width_m is not None
            and avg_width < config.min_skinny_width_m
        ):
            dropped_skinny_parts.append((geom, avg_width))
            continue

        records["skinny"].append(
            {
                "parent_id": parent_id,
                "part_id": skinny_id,
                "type": "skinny",
                "area_m2": geom.area,
                "area_ratio_parent": geom.area / original_area,
                "min_width_m": min_width,
                "avg_width_m": avg_width,
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
                    "min_width_m": min_width,
                    "avg_width_m": avg_width,
                    "buffer_m": config.boundary_buffer_distance,
                    "along_m": config.along_boundary_resolution,
                    "geometry": line,
                }
            )

        records["skeleton_lines"].extend(line_records)
        records["skeleton_vertices"].extend(vertex_records)

    return dropped_skinny_parts


def average_skeleton_width(line_records, fallback_width):
    """Return length-weighted mean full width from skeleton D values."""
    weighted_sum = 0.0
    total_length = 0.0

    for record in line_records:
        width = record.get("width_mean_m")
        if width is None:
            d_mean = record.get("D_mean")
            width = None if d_mean is None else 2.0 * d_mean

        length = record.get("length_m", 0.0)

        if width is None or not math.isfinite(width) or length <= 0:
            continue

        weighted_sum += width * length
        total_length += length

    if total_length <= 0:
        return fallback_width

    return weighted_sum / total_length


def filter_final_fleshy_parts(fleshy_parts, config):
    """Drop optional small fleshy final-product pieces before output."""
    kept_fleshy = []
    dropped_fleshy = []
    min_fleshy_area = config.min_fleshy_area_m2

    for geom in fleshy_parts:
        if min_fleshy_area is not None and geom.area < min_fleshy_area:
            dropped_fleshy.append(geom)
        else:
            kept_fleshy.append(geom)

    return kept_fleshy, dropped_fleshy


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

    fleshy_parts, dropped_fleshy_parts = filter_final_fleshy_parts(
        fleshy_parts,
        config,
    )

    raw_skinny_area = sum(g.area for g in candidate_skinny_raw_parts)
    mask_skinny_area = sum(g.area for g in candidate_skinny_mask_parts)
    retained_skinny_area = sum(g.area for g in skinny_parts)
    dropped_fleshy_area = sum(g.area for g in dropped_fleshy_parts)

    add_fleshy_records(
        records=records,
        parent_id=parent_id,
        original_area=original_area,
        fleshy_parts=fleshy_parts,
        config=config,
    )

    dropped_skinny_parts = add_skinny_records_and_skeleton(
        records=records,
        parent_id=parent_id,
        original_area=original_area,
        skinny_parts=skinny_parts,
        config=config,
    )

    retained_skinny_area -= sum(g.area for g, _ in dropped_skinny_parts)
    dropped_skinny_area = sum(g.area for g, _ in dropped_skinny_parts)

    records["log_lines"].append(
        f"Rank {rank}, Parent {parent_id}: original_area={original_area:.3f}, "
        f"candidate_skinny_mask_area={mask_skinny_area:.3f}, "
        f"candidate_skinny_raw_area={raw_skinny_area:.3f} "
        f"({raw_skinny_area / original_area:.4%}), "
        f"retained_skinny_area={retained_skinny_area:.3f} "
        f"({retained_skinny_area / original_area:.4%}), "
        f"final_filter_dropped_fleshy={len(dropped_fleshy_parts)} "
        f"({dropped_fleshy_area:.3f} m^2), "
        f"final_filter_dropped_skinny={len(dropped_skinny_parts)} "
        f"({dropped_skinny_area:.3f} m^2)"
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


def gather_records(comm, local_records, rank, scratch_dir=None):
    """Gather local records to rank 0 and flatten.

    Geometry records can become too large for MPI's pickled object gather.
    For MPI runs, use the shared filesystem as a staging area: each rank writes
    one pickle file, then rank 0 reads and merges them in rank order.
    """
    size = comm.Get_size()

    if size == 1:
        sort_record_dict(local_records)
        for line in local_records["log_lines"]:
            print(line)
        return local_records

    if scratch_dir is None:
        raise ValueError("scratch_dir is required for MPI record gathering")

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    rank_file = scratch_dir / f"records_rank_{rank:06d}.pkl"
    with rank_file.open("wb") as stream:
        pickle.dump(local_records, stream, protocol=pickle.HIGHEST_PROTOCOL)

    comm.Barrier()

    if rank != 0:
        comm.Barrier()
        return None

    gathered = []
    for source_rank in range(size):
        filename = scratch_dir / f"records_rank_{source_rank:06d}.pkl"
        with filename.open("rb") as stream:
            gathered.append(pickle.load(stream))

    records = merge_record_dicts(gathered)
    sort_record_dict(records)

    for rank_records in gathered:
        for line in rank_records["log_lines"]:
            print(line)

    for source_rank in range(size):
        filename = scratch_dir / f"records_rank_{source_rank:06d}.pkl"
        filename.unlink(missing_ok=True)
    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    comm.Barrier()

    return records


def gather_record_list(comm, local_records, rank, scratch_dir, stem):
    """Gather a list of records to rank 0 using shared-file staging."""
    size = comm.Get_size()

    if size == 1:
        return local_records

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    rank_file = scratch_dir / f"{stem}_rank_{rank:06d}.pkl"
    with rank_file.open("wb") as stream:
        pickle.dump(local_records, stream, protocol=pickle.HIGHEST_PROTOCOL)

    comm.Barrier()

    if rank != 0:
        comm.Barrier()
        return None

    gathered = []
    for source_rank in range(size):
        filename = scratch_dir / f"{stem}_rank_{source_rank:06d}.pkl"
        with filename.open("rb") as stream:
            gathered.extend(pickle.load(stream))

    for source_rank in range(size):
        filename = scratch_dir / f"{stem}_rank_{source_rank:06d}.pkl"
        filename.unlink(missing_ok=True)
    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    comm.Barrier()

    return gathered


def reduce_timing_max(timings, comm, rank, mpi_module):
    """Return max elapsed time per stage on rank 0."""
    reduced = []

    for label, elapsed in timings:
        max_elapsed = comm.reduce(float(elapsed), op=mpi_module.MAX, root=0)
        if rank == 0:
            reduced.append((label, max_elapsed))

    return reduced if rank == 0 else None


def print_timing_summary(timings):
    """Print a compact rank-0 timing summary."""
    print()
    print("Timing summary:")
    for label, elapsed in timings:
        print(f"  {label:<38} {elapsed:10.3f} s")
    print()


def main(argv=None):
    total_start = time.perf_counter()
    step_start = total_start

    # 1. Resolve recipe/configuration, including input/output filenames.
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
    timings = [
        ("1 config + MPI initialization", time.perf_counter() - step_start)
    ]

    # 2. Read the source marsh polygons on every rank.  This keeps indexing
    # simple; each rank then selects only its assigned polygon IDs.
    step_start = time.perf_counter()
    original_gdf = get_input_polygons(run_config.input_file)
    timings.append(("2 read input polygons", time.perf_counter() - step_start))

    print_run_header(original_gdf, rank, size, recipe_name, run_config)

    # 3. Build all local products on each rank.  Arc records are derived
    # while line geometries are still in memory, avoiding a slow GPKG re-read
    # and parallelizing the expensive explode/resample step.
    step_start = time.perf_counter()
    local_indices = list(range(rank, len(original_gdf), size))
    print(f"Rank {rank}: processing {len(local_indices)} polygons")

    local_records = process_local_polygons(
        original_gdf=original_gdf,
        local_indices=local_indices,
        rank=rank,
        config=config,
    )

    local_arc_records = make_arc_line_records(local_records, config)
    timings.append(
        ("3 build local decomposition + arcs", time.perf_counter() - step_start)
    )

    # 4. Gather all local products to rank 0 for final file I/O.
    # Large Shapely records are staged through shared files instead of one
    # MPI object gather, which is more robust for big marsh datasets.
    step_start = time.perf_counter()
    record_scratch_dir = (
        run_config.output_file.parent
        / f".{run_config.output_file.stem}_mpi_records"
    )
    arc_scratch_dir = (
        run_config.output_file.parent
        / f".{run_config.output_file.stem}_mpi_arc_records"
    )

    records = gather_records(
        comm=comm,
        local_records=local_records,
        rank=rank,
        scratch_dir=record_scratch_dir,
    )
    arc_records = gather_record_list(
        comm=comm,
        local_records=local_arc_records,
        rank=rank,
        scratch_dir=arc_scratch_dir,
        stem="arc_records",
    )
    timings.append(("4 gather products to rank 0", time.perf_counter() - step_start))

    # Only rank 0 prints timings.  For parallel stages, report the slowest
    # rank's elapsed time so load imbalance is visible.
    timings = reduce_timing_max(timings, comm, rank, MPI)

    if rank != 0:
        return

    # 5. Rank 0 writes the full diagnostic GeoPackage and prints QA summary.
    step_start = time.perf_counter()
    gdfs = make_output_gdfs(original_gdf, records)

    write_output_gpkg(gdfs, run_config.output_file)

    print_summary(gdfs, run_config.output_file)
    timings.append(("5 write GeoPackage + summary", time.perf_counter() - step_start))

    # 6. Rank 0 writes final RiverMapper arc lines in lon/lat plus the
    # source-CRS QA copy.  Arc explosion/resampling already happened above.
    step_start = time.perf_counter()
    write_arc_line_records(
        records=arc_records,
        crs=original_gdf.crs,
        output_file=run_config.output_file.with_name(
            run_config.output_file.stem + "_arc_lines.shp"
        ),
        output_crs="EPSG:4326",
    )
    timings.append(("6 write arc-line outputs", time.perf_counter() - step_start))
    timings.append(("total rank-0 wall time", time.perf_counter() - total_start))
    print_timing_summary(timings)
