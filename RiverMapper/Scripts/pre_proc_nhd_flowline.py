"""
Preprocess NHD flowline shapefile for use in RiverMapper
"""

import os
from pathlib import Path
import math
# import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, split, substring
# from multiprocessing import Pool, cpu_count


def split_line(line: LineString, threshold: float, strategy: str = "ceil"):
    """
    Split a (densified) LineString into ~equal pieces based on a target max length 'threshold'.
    Cuts occur ONLY at existing vertices (nodes).

    Parameters
    ----------
    line : LineString (2D)
    threshold : float
        Target maximum segment length (same units as line.length). Must be > 0.
    strategy : {"ceil","round"}
        - "ceil": n = ceil(L / threshold)  → tends to keep segments <= threshold (on average).
        - "round": n ≈ round(L / threshold) → more balanced, may slightly exceed threshold.

    Returns
    -------
    list[LineString]
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    # Get 2D vertices in case input has Z
    pts = [(xy[0], xy[1]) for xy in line.coords]
    N = len(pts)
    if N < 2:
        return [line]

    L = line.length
    if L <= threshold:
        return [line]

    M = N - 1  # number of edges
    # choose number of segments from length + threshold
    n = int(math.ceil(L / threshold)) if strategy == "ceil" else max(1, int(round(L / threshold)))
    # clamp so each piece has at least one edge
    n = max(1, min(n, M))

    # Evenly distribute edges; first r segments get (base+1) edges, rest get base
    base, r = divmod(M, n)

    pieces = []
    start_edge = 0
    for i in range(n):
        seg_edges = base + (1 if i < r else 0)  # >= 1
        end_edge = start_edge + seg_edges

        # convert edge indices to vertex indices; include end vertex
        start_idx = start_edge
        end_idx = end_edge
        coords = pts[start_idx:end_idx + 1]  # guaranteed len >= 2
        pieces.append(LineString(coords))

        start_edge = end_edge

    return pieces


def split_line2(line: LineString, max_len: float):
    """
    Split into ~equal pieces with length <= max_len using along-track substring.
    """
    L = line.length
    if L == 0 or L <= max_len:
        return [line]

    n = int(math.ceil(L / max_len))   # number of segments
    segL = L / n

    pieces = []
    start = 0.0
    for i in range(n):
        end = L if i == n - 1 else (i + 1) * segL
        seg = substring(line, start, end, normalized=False)
        if isinstance(seg, LineString) and seg.length > 0:
            pieces.append(seg)
        start = end
    return pieces


def merge_lines(gdf):
    '''
    For a gdf with LineString and MultiLineString geometries,
    merge parts of a MultiLineString into a single LineString,
    and for those that cannot be merged because they are disconnected,
    convert each part into a LineString.
    Original lineStrings are kept as they are.
    '''
    new_geometries = []
    for index, row in gdf.iterrows():
        geom = row.geometry

        if isinstance(geom, MultiLineString):  # merge multi-linestrings into one
            geom = linemerge(geom)
            if not isinstance(geom, LineString):
                print(
                    f"Warning: failed to merge MultiLineString at index {index}; "
                    "This can happen if the MultiLineString has disconnected segments."
                    "The disconnected segments will be treated as individual LineStrings."
                )

        if isinstance(geom, LineString):
            new_geometries.append(geom)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                new_geometries.append(line)
        else:
            print(f"Skipping non-LineString geometry at index {index}")

    new_gdf = gpd.GeoDataFrame(geometry=new_geometries, crs=gdf.crs)
    return new_gdf


def split_nhdflowline(gdf, max_segment_length=15000):
    '''
    Function to split NHD flowlines into shorter segments based on a maximum length.
    This is useful for ensuring that the flowlines are not too long for processing
    in RiverMapper.

    Inputs:
    - gdf: GeoDataFrame containing NHD flowlines
    - max_segment_length: Maximum length of each segment in meters

    Outputs:
    - new_gdf: GeoDataFrame with flowlines split into shorter segments
    '''
    new_geometries = []
    for index, row in gdf.iterrows():
        geom = row.geometry

        if isinstance(geom, LineString):
            if geom.length > max_segment_length:
                new_geometries.extend(split_line2(geom, max_segment_length))
            else:
                new_geometries.append(geom)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                if line.length > max_segment_length:
                    new_geometries.extend(split_line2(line, max_segment_length))
                else:
                    new_geometries.append(line)
        else:
            print(f"Skipping non-LineString geometry at index {index}")

    new_gdf = gpd.GeoDataFrame(geometry=new_geometries, crs=gdf.crs)

    return new_gdf


def group_line_by_polygons(line_row, polygons):
    '''
    Group a line as inside or outside of polygons.
    Split the line into inside/outside lines if it intersects with a polygon.
    '''

    line = line_row.geometry
    lines_inside = []
    lines_outside = []
    split_occurred = False

    for _, polygon_row in polygons.iterrows():
        polygon = polygon_row.geometry

        if line.intersects(polygon):
            split_result = split(line, polygon.boundary)

            for segment in split_result.geoms:
                if segment.intersects(polygon):
                    lines_inside.append(segment)
                else:
                    lines_outside.append(segment)

            split_occurred = True

    if not split_occurred:
        if any(line.within(polygon) for _, polygon_row in polygons.iterrows()):
            lines_inside.append(line)
        else:
            lines_outside.append(line)

    return lines_inside, lines_outside


def densify_linestring(line, resolution):
    '''
    Function to densify a LineString by adding points along its length.
    The original points are retained, and new points are added at regular intervals.
    '''
    if not isinstance(line, LineString):
        return line

    coords = list(line.coords)
    is_3d = len(coords[0]) == 3

    new_coords = [coords[0]]

    for i in range(1, len(coords)):
        start = coords[i - 1]
        end = coords[i]
        segment = LineString([start, end])
        length = segment.length

        if length > resolution:
            num_points = int(length // resolution)
            for j in range(1, num_points + 1):
                # Distance along the segment
                fraction = j / (num_points + 1)
                point_2d = segment.interpolate(fraction, normalized=True)
                x, y = point_2d.x, point_2d.y

                if is_3d:
                    z = start[2] + (end[2] - start[2]) * fraction
                    new_coords.append((x, y, z))
                else:
                    new_coords.append((x, y))

        new_coords.append(end)

    return LineString(new_coords)


def _dedup_line_consecutive(ls: LineString, tol: float = 0.0) -> LineString:
    """Remove consecutive duplicate vertices from a LineString.
    tol is a distance threshold in coordinate units; 0 means exact duplicates only.
    """
    if ls.is_empty:
        return ls
    coords = list(ls.coords)
    if len(coords) <= 1:
        return LineString(coords)

    kept = [coords[0]]
    tol2 = tol * tol
    for c in coords[1:]:
        dx = c[0] - kept[-1][0]
        dy = c[1] - kept[-1][1]
        if dx*dx + dy*dy > tol2:
            kept.append(c)

    # If everything collapsed, return empty LineString
    return LineString(kept) if len(kept) >= 2 else LineString()


def clean_duplicate_vertices(geom, tol: float = 0.0):
    """Apply consecutive-duplicate removal to LineString / MultiLineString.
    Other geometry types are returned unchanged.
    """
    if geom is None:
        return None
    gt = geom.geom_type
    if gt == "LineString":
        return _dedup_line_consecutive(geom, tol)
    elif gt == "MultiLineString":
        parts = [_dedup_line_consecutive(ls, tol) for ls in geom.geoms]
        parts = [p for p in parts if not p.is_empty and len(p.coords) >= 2]
        if not parts:
            return LineString()
        return parts[0] if len(parts) == 1 else MultiLineString(parts)
    else:
        return geom  # leave Points/Polygons/etc. as-is


def pre_process_nhdflowlines(
    input_flowline=None, input_nhdarea=None, intermediate_crs="esri:102008",
    line_identifier="gnis_id",
    max_segment_length=15e3, along_segment_resolution=20,
    output_dir=None, diag_output=False
):
    '''
    Main function to pre-process NHD flowlines.
    The precedures are labeled as (1) to (6) below:

    Inputs:
    - input_flowline: Path to the NHD flowline shapefile
    - input_nhdarea: Path to the NHD area shapefile; if None, bypass the
      inside/outside NHD Area splitting
    - intermediate_crs: CRS to project the shapefiles to (default is ESRI:102008)
    - line_identifier: The identifier for selecting lines, e.g., "gnis_id"
    - max_segment_length: Maximum length of each segment in meters
    - along_segment_resolution: Resolution for densifying the lines in meters
    - diag_output: If True, outputs diagnostic shapefiles for each step
    Outputs:
    - A new shapefile with pre-processed NHD flowlines saved in the output directory.
    - Other diagnostic shapefiles if diag_output is True under output directory.
    '''
    # Load the shapefiles
    lines = gpd.read_file(input_flowline)
    polygons = gpd.read_file(input_nhdarea) if input_nhdarea is not None else None

    if output_dir is None:
        output_dir = f"{input_flowline.parent}/{input_flowline.stem}_processed/"
        os.makedirs(output_dir, exist_ok=True)

    # *) project to meters
    if lines.crs is None or (polygons is not None and polygons.crs is None):
        raise ValueError("Input shapefiles must have a defined CRS.")
    original_crs = lines.crs
    print(f'projecting to intermediate CRS: {intermediate_crs}')
    lines = lines.to_crs(intermediate_crs)
    if polygons is not None:
        polygons = polygons.to_crs(intermediate_crs)
    # slightly buffer polygons to avoid gaps (e.g., at river intersections)
    # polygons = gpd.GeoDataFrame( geometry=polygons.geometry.buffer(0.1), crs=intermediate_crs)

    # *) Subset lines based on certain criteria (in this example valid line_identifer),
    #    because NHD flowlines can be too dense for the purpose of compound flood modeling
    if line_identifier is not None:
        print(f'subsetting lines based on {line_identifier}')
        lines = lines[lines[line_identifier].notnull() & (lines[line_identifier] != "")]

    # *) Dissolve lines with the same name (gnis_id), otherwise one river can be broken
    #    into too many segments due to intersection with tributaries. Most tributaries
    #    are negligible and discarded in Step 1)
    if line_identifier is not None:
        print(f'dissolving lines with the same line_identifier: {line_identifier}')
        lines = lines.dissolve(by=line_identifier, as_index=False)
        lines = merge_lines(lines)
        if diag_output:
            lines.to_file(output_dir + input_flowline.stem + "_merged.shp")

    # *) Group lines by inside and outside of NHDArea polygons. If a line intersects
    #    with a polygon, it is split into segments. This is important for RiverMapper
    #    to correctly identify the river arcs. The lines outside of the polygons
    #    are expanded into pseudo river arcs in RiverMapper.
    if polygons is not None:
        inside_lines = gpd.clip(lines, polygons, keep_geom_type=False)
        inside_lines = inside_lines.explode(index_parts=False).reset_index(drop=True)

        outside_lines = gpd.overlay(lines, polygons, how='difference')
        outside_lines = outside_lines.explode(index_parts=False).reset_index(drop=True)

        # only retain linestrings (sometimes points may be generated by clipping and exploding)
        inside_lines = inside_lines[inside_lines.geometry.type == 'LineString']
        outside_lines = outside_lines[outside_lines.geometry.type == 'LineString']

        if diag_output:
            if not inside_lines.empty:
                inside_lines.to_file(output_dir + input_flowline.stem + "_inside.shp")
            else:
                print("Warning: No inside lines found.")
            if not outside_lines.empty:
                outside_lines.to_file(output_dir + input_flowline.stem + "_outside.shp")
            else:
                print("Warning: No outside lines found.")

        lines = pd.concat([inside_lines, outside_lines], ignore_index=True)
        if diag_output:
            lines.to_file(output_dir + input_flowline.stem + "_inside_outside.shp")
    else:
        print('input_nhdarea is None; bypassing inside/outside NHD Area splitting')

    # *) Split long lines into shorter segments. A long river can change morphology
    #    (e.g., width, sinuosity) along its length, and RiverMapper will perform better
    #    if the river is split into shorter segments. The threshold for splitting
    #    is set to 15 km, which can be adjusted based on your needs.
    print('splitting long lines into shorter segments')
    lines = split_nhdflowline(lines, max_segment_length)
    if diag_output:
        lines.to_file(output_dir + input_flowline.stem + "_split.shp")

    # *) Densify the vertices on each line. This is important for RiverMapper to
    #    accurately represent the river geometry. The resolution is set to 20 m,
    #    which can be adjusted based on your needs.
    if along_segment_resolution is not None and along_segment_resolution > 0:
        print('densifying lines')
        for index, row in lines.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                lines.at[index, 'geometry'] = densify_linestring(geom, along_segment_resolution)
            else:
                raise ValueError(f"Geometry at index {index} is not a LineString")

    # *) Add an attribute "keep = 1" to the new GeoDataFrame. This forces the lines
    #    to be expanded into river arcs in RiverMapper regardless of other criteria
    #    set in the RiverMapper configuration.
    print('adding keep = 1 attribute to the lines')
    lines['keep'] = 1

    # *) Clean up duplicate vertices
    lines["geometry"] = lines.geometry.apply(lambda g: clean_duplicate_vertices(g, tol=1.0))
    lines = lines[~lines.geometry.is_empty]

    # *) Drop short lines, which can be generated if slivers exist in the NHD area polygons
    lines = lines[lines.geometry.length > 20]  # keep lines longer than 20 m

    # Save the result to a new shapefile
    lines.to_crs(original_crs, inplace=True)  # reproject back to original CRS
    lines.to_file(output_dir + input_flowline.stem + "_processed.shp")


def sample_densify(shpfname, max_segment_length=20):
    '''
    Sample usage of the densify_linestring function
    '''
    gdf = gpd.read_file(shpfname)
    new_geometries = []
    for index, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            new_geometries.append(densify_linestring(geom, max_segment_length * 1e-5))
        else:
            print(f"Skipping non-LineString geometry at index {index}")

    new_gdf = gpd.GeoDataFrame(geometry=new_geometries, crs=gdf.crs)
    output_file = Path(shpfname).with_name(f"{Path(shpfname).stem}_processed_{max_segment_length}.shp")
    new_gdf.to_file(output_file)
    return new_gdf


def sample_detect_duplicate_vertices(shpfname):
    '''
    Sample usage of the clean_duplicate_vertices function
    '''
    gdf = gpd.read_file(shpfname)
    for index, row in gdf.iterrows():
        geom = row.geometry
        gdf.at[index, 'geometry'] = clean_duplicate_vertices(geom, tol=1.0)
    gdf = gdf[~gdf.geometry.is_empty]
    output_file = Path(shpfname).with_name(f"{Path(shpfname).stem}_dedup.shp")
    gdf.to_file(output_file)
    return gdf


def sample():
    '''
    Example usage of the pre_process_nhdflowlines function.
    This will preprocess the NHD flowline shapefile and save the result to a new shapefile.

    It is recommended to clip the NHD flowline and NHD area to the area of interest
    before running this function to avoid processing too many lines.
    '''
    pre_process_nhdflowlines(
        input_flowline=Path(
            "/sciclone/schism10/Hgrid_projects/STOFS3D-v8/a51_RiverMapper/Shapefiles/"
            "a51filler_IWW_filler_cleaned_20m.shp"),
        input_nhdarea=Path(
            "/sciclone/schism10/Hgrid_projects/STOFS3D-v8/a51_RiverMapper/Shapefiles/"
            "nhdarea_la_ms_cleaned.shp"),
        line_identifier=None,  # 'gnis_id',  # use gnis_id to select lines
        max_segment_length=15000,  # split segments with a maximum segment length in kilometers
        along_segment_resolution=None,  # 20,  # meters, densify while retaining original points
        diag_output=True  # set to True to output diagnostic shapefiles
    )
    print('Done!')


if __name__ == "__main__":
    sample()
