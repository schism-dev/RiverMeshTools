"""GeoDataFrame construction, QA summaries, and marsh file export."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .geometry import clean_geom, explode_to_lines, resample_linestring


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

    Skeleton lines are resampled with spacing Y in the projected source CRS.

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
            "resample": 0.2 * config.Y,
        },
        {
            "layer": "skinny_boundary_lines",
            "arc_pos": "left half",
            "dummy": 0,
            "resample": 0.2 * config.Y,
        },
        {
            "layer": "skinny_skeleton_lines",
            "arc_pos": "dummy",
            "dummy": 1,
            "resample": config.Y,
        },
    ]

    def read_layer(layer):
        try:
            return gpd.read_file(decomposed_gpkg, layer=layer)
        except Exception as exc:
            print(f"Warning: could not read layer {layer!r}: {exc}")
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
                rec["resampled"] = (
                    bool(resample) if resample is not None else False
                )
                rec["resamp_m"] = (
                    float(resample) if resample is not None else np.nan
                )
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
    value_columns = ["src_layer", "arc_pos", "dummy", "resampled"]
    print(out_gdf[value_columns].value_counts())

    return out_gdf
