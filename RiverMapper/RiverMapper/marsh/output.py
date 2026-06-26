"""GeoDataFrame construction, QA summaries, and marsh file export."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.ops import transform as transform_geometry

from .geometry import clean_geom, explode_to_lines, resample_linestring


ARC_LAYER_SPECS = (
    {
        "layer": "fleshy_boundary_lines",
        "record_key": "fleshy_boundary_lines",
        "arc_pos": "regular",
        "dummy": 0,
        "spacing_parameter": "boundary_vertex_spacing",
    },
    {
        "layer": "skinny_boundary_lines",
        "record_key": "skinny_boundary_lines",
        "arc_pos": "left half",
        "dummy": 0,
        "spacing_parameter": "boundary_vertex_spacing",
    },
    {
        "layer": "skinny_skeleton_lines",
        "record_key": "skeleton_lines",
        "arc_pos": "dummy",
        "dummy": 1,
        "spacing_parameter": "skinny_centerline_spacing",
    },
)


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


def write_output_gpkg(gdfs, output_file):
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


def print_summary(gdfs, output_file):
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
                np.nan,
                skinny_area,
                np.nan,
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
        cols = [
            "parent_id",
            "part_id",
            "area_m2",
            "min_dimension_m",
            "paving_res_m",
        ]
        print(fleshy_gdf[cols].to_string(index=False))
        print()

    area_error = reconstructed_area - original_area
    rel_error = area_error / original_area if original_area > 0 else 0.0

    print("Area reconstruction check, relative to the original geometry:")
    print(f"  fleshy + skinny - original = {area_error:.6f} m^2")
    print(f"  relative error = {rel_error:.6e}")
    print()
    print("Note:")
    print("  candidate_skinny_mask uses filtered/reference geometry.")
    print("  candidate_skinny_raw maps the candidate to original geometry.")
    print("  Cleanup may discard tiny pieces, depending on the recipe.")
    print("  Boundary-line layers come from the corresponding final polygons.")
    print("  skeleton line geometry is detailed medial axis at skeleton_dx.")
    print("  skeleton vertex D = distance to nearest skinny polygon boundary.")


def make_arc_line_records(records, config):
    """Build RiverMapper arc-line records from in-memory decomposition records.

    This is the fast path used by the MPI workflow.  Each rank can resample
    its own boundary/skeleton records before rank 0 writes the final datasets.
    """
    arc_records = []

    for spec in ARC_LAYER_SPECS:
        layer_name = spec["layer"]
        record_key = spec["record_key"]
        arc_pos = spec["arc_pos"]
        dummy = spec["dummy"]
        resample = getattr(config, spec["spacing_parameter"])
        resample = resample if resample is not None and resample > 0 else None

        for source_record in records.get(record_key, []):
            attrs = {
                name: value
                for name, value in source_record.items()
                if name != "geometry"
            }

            for line in explode_to_lines(source_record["geometry"]):
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
                rec["resampled"] = "T" if resample is not None else "F"
                rec["resamp_m"] = (
                    float(resample) if resample is not None else np.nan
                )
                rec["geometry"] = line

                arc_records.append(rec)

    return arc_records


def arc_sort_key(record):
    """Stable sorting key for reproducible arc-line output."""
    return (
        record.get("parent_id", -1),
        record.get("src_layer", ""),
        record.get("skinny_id", -1),
        record.get("fleshy_id", -1),
        record.get("part_id", -1),
        record.get("line_id", -1),
        record.get("branch_id", -1),
    )


def write_arc_lines(gdf, filename):
    """Replace one arc-line dataset and report its CRS."""
    filename = Path(filename)

    if filename.exists():
        if filename.suffix.lower() == ".shp":
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                f = filename.with_suffix(suffix)
                if f.exists():
                    f.unlink()
        else:
            filename.unlink()

    if filename.suffix.lower() == ".shp":
        gdf.to_file(filename, driver="ESRI Shapefile")
    else:
        gdf.to_file(filename, layer="arc_lines", driver="GPKG")

    print(f"Saved: {filename}")
    print(f"Output CRS: {gdf.crs}")


def write_arc_line_gdf(source_gdf, output_file, output_crs="EPSG:4326"):
    """Write source-CRS and optional reprojected RiverMapper arc lines."""
    output_file = Path(output_file)

    if source_gdf.empty:
        raise ValueError("No line features generated.")

    if output_file.stem.endswith("_arc_lines"):
        original_crs_stem = (
            output_file.stem.removesuffix("_arc_lines")
            + "_arc_line_original_crs"
        )
    else:
        original_crs_stem = output_file.stem + "_arc_line_original_crs"
    original_crs_output_file = output_file.with_name(
        original_crs_stem + output_file.suffix
    )

    # Write coordinates exactly as constructed in the input projected CRS.
    write_arc_lines(source_gdf, original_crs_output_file)

    value_columns = ["src_layer", "arc_pos", "dummy", "resampled"]
    print(source_gdf[value_columns].value_counts())

    if output_crs is None:
        output_gdf = source_gdf
    else:
        # Keep reprojection isolated at the end so source-coordinate output
        # and any datum displacement can be inspected independently.
        transformer = Transformer.from_crs(
            source_gdf.crs,
            output_crs,
            always_xy=True,
        )
        geometries = [
            transform_geometry(transformer.transform, geometry)
            for geometry in source_gdf.geometry
        ]
        output_gdf = source_gdf.copy()
        output_gdf = output_gdf.set_geometry(
            gpd.GeoSeries(
                geometries,
                index=source_gdf.index,
                crs=output_crs,
            )
        )

        operation = transformer.get_last_used_operation()
        print(f"Selected operation: {operation.description}")
        print(f"Selected pipeline: {operation.to_proj4()}")

        roundtrip_gdf = output_gdf.to_crs(source_gdf.crs)
        errors = source_gdf.geometry.hausdorff_distance(
            roundtrip_gdf.geometry
        )
        print("Round-trip displacement statistics (source CRS units):")
        print(errors.describe())
        print(
            "Maximum round-trip displacement: "
            f"{errors.max():.12g} m"
        )

    write_arc_lines(output_gdf, output_file)

    return output_gdf


def write_arc_line_records(
    records,
    crs,
    output_file,
    output_crs="EPSG:4326",
):
    """Write arc-line records collected from one or more MPI ranks."""
    records = sorted(records, key=arc_sort_key)
    source_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    return write_arc_line_gdf(
        source_gdf=source_gdf,
        output_file=output_file,
        output_crs=output_crs,
    )


def extract_arc_lines_from_decomposed_gpkg(
    decomposed_gpkg,
    output_file,
    config,
    output_crs="EPSG:4326",
):
    """
    Extract arc lines from decomposed marsh GPKG.

    Input layers are assumed to already be LineString/MultiLineString:
        fleshy_boundary_lines      -> arc_pos = "regular",   dummy = 0
        skinny_boundary_lines      -> arc_pos = "left half", dummy = 0
        skinny_skeleton_lines      -> arc_pos = "dummy",     dummy = 1

    Boundary lines use ``boundary_vertex_spacing``. Skeleton lines use
    ``skinny_centerline_spacing``.

    All available attributes from source layers are preserved.  Two copies
    are always written:

    * ``*_arc_line_original_crs`` always retains the source CRS.
    * The requested ``*_arc_lines`` output uses ``output_crs`` when provided,
      or retains the source CRS when ``output_crs`` is ``None``.
    """

    decomposed_gpkg = Path(decomposed_gpkg)

    def read_layer(layer):
        try:
            return gpd.read_file(decomposed_gpkg, layer=layer)
        except Exception as exc:
            print(f"Warning: could not read layer {layer!r}: {exc}")
            return None

    source_records = {
        spec["record_key"]: []
        for spec in ARC_LAYER_SPECS
    }
    crs = None

    for spec in ARC_LAYER_SPECS:
        layer_name = spec["layer"]
        gdf = read_layer(layer_name)

        if gdf is None or gdf.empty:
            print(f"Warning: missing or empty layer: {layer_name}")
            continue

        if crs is None:
            crs = gdf.crs

        for _, row in gdf.iterrows():
            rec = row.drop(labels="geometry").to_dict()
            rec["geometry"] = row.geometry
            source_records[spec["record_key"]].append(rec)

    if crs is None:
        raise ValueError("No valid input LineString layers found.")

    return write_arc_line_records(
        records=make_arc_line_records(source_records, config),
        crs=crs,
        output_file=output_file,
        output_crs=output_crs,
    )
