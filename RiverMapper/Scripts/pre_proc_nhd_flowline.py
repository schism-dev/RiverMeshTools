"""
Preprocess NHD flowline shapefile for use in RiverMapper
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, split
from multiprocessing import Pool, cpu_count


def split_line(line, threshold):
    '''Function to split a LineString into shorter segments'''
    points = list(line.coords)  # Get the coordinates of the LineString
    new_lines = []

    nsub = max(1, int(line.length / threshold))  # Number of subsegments
    for i in range(nsub):
        start_idx = min(i * len(points) // nsub, len(points))
        end_idx = min((i + 1) * len(points) // nsub, len(points))
        new_lines.append(LineString(points[start_idx:end_idx]))

    return new_lines


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


def split_nhdflowline(gdf, max_segment_length=15):
    '''
    Function to split NHD flowlines into shorter segments based on a maximum length.
    This is useful for ensuring that the flowlines are not too long for processing
    in RiverMapper.

    Inputs:
    - gdf: GeoDataFrame containing NHD flowlines
    - max_segment_length: Maximum length of each segment in kilometers

    Outputs:
    - new_gdf: GeoDataFrame with flowlines split into shorter segments
    '''
    new_geometries = []
    for index, row in gdf.iterrows():
        geom = row.geometry

        if isinstance(geom, LineString):
            if geom.length > max_segment_length * 1e-2:  # 1e-5 * 1000, i.e., convert km to degree approximately
                new_geometries.extend(split_line(geom, max_segment_length * 1e-2))
            else:
                new_geometries.append(geom)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                if line.length > max_segment_length * 1e-2:
                    new_geometries.extend(split_line(line, max_segment_length * 1e-2))
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


def densify_linestring(line, max_segment_length):
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

        if length > max_segment_length:
            num_points = int(length // max_segment_length)
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


def pre_process_nhdflowlines(
    input_flowline=None, input_nhdarea=None,
    line_identifier="gnis_id",
    max_segment_length=15, along_segment_resolution=20,
    output_dir=None, diag_output=False
):
    '''
    Main function to pre-process NHD flowlines.
    The precedures are labeled as (1) to (6) below:

    Inputs:
    - input_flowline: Path to the NHD flowline shapefile
    - input_nhdarea: Path to the NHD area shapefile
    - line_identifier: The identifier for selecting lines, e.g., "gnis_id"
    - max_segment_length: Maximum length of each segment in kilometers
    - along_segment_resolution: Resolution for densifying the lines in meters
    - diag_output: If True, outputs diagnostic shapefiles for each step
    
    Outputs:
    - A new shapefile with pre-processed NHD flowlines saved in the output directory.
    - Other diagnostic shapefiles if diag_output is True under output directory.
    '''
    # Load the shapefiles
    lines = gpd.read_file(input_flowline)
    polygons = gpd.read_file(input_nhdarea)

    if output_dir is None:
        output_dir = f"{input_flowline.parent}/{input_flowline.stem}_processed/"
        os.makedirs(output_dir, exist_ok=True)

    # 1) Subset lines based on certain criteria (in this example valid line_identifer),
    #    because NHD flowlines can be too dense for the purpose of compound flood modeling
    if line_identifier is not None:
        print(f'subsetting lines based on {line_identifier}')
        lines = lines[lines[line_identifier].notnull() & (lines[line_identifier] != "")]

    # 2) Dissolve lines with the same name (gnis_id), otherwise one river can be broken
    #    into too many segments due to intersection with tributaries. Most tributaries
    #    are negligible and discarded in Step 1)
    print('dissolving lines with the same gnis_id')
    lines = lines.dissolve(by='gnis_id', as_index=False)
    lines = merge_lines(lines)
    if diag_output:
        lines.to_file(input_flowline.with_name(input_flowline.stem + "_merged.shp"))

    # 3) Group lines by inside and outside of NHDArea polygons. If a line intersects
    #    with a polygon, it is split into segments. This is important for RiverMapper
    #    to correctly identify the river arcs. The lines outside of the polygons
    #    are expanded into pseudo river arcs in RiverMapper.
    inside_lines = gpd.clip(lines, polygons, keep_geom_type=False)
    inside_lines = inside_lines.explode(index_parts=False).reset_index(drop=True)

    outside_lines = gpd.overlay(lines, polygons, how='difference')
    outside_lines = outside_lines.explode(index_parts=False).reset_index(drop=True)

    # only retain linestrings (sometimes points may occur after clipping and exploding)
    inside_lines = inside_lines[inside_lines.geometry.type == 'LineString']
    outside_lines = outside_lines[outside_lines.geometry.type == 'LineString']

    if diag_output:
        inside_lines.to_file(input_flowline.with_name(input_flowline.stem + "_inside.shp"))
        outside_lines.to_file(input_flowline.with_name(input_flowline.stem + "_outside.shp"))

    lines = pd.concat([inside_lines, outside_lines], ignore_index=True)
    if diag_output:
        lines.to_file(input_flowline.with_name(input_flowline.stem + "_inside_outside.shp"))

    # 4) Split long lines into shorter segments. A long river can change morphology
    #    (e.g., width, sinuosity) along its length, and RiverMapper will perform better
    #    if the river is split into shorter segments. The threshold for splitting
    #    is set to 15 km, which can be adjusted based on your needs.
    print('splitting long lines into shorter segments')
    lines = split_nhdflowline(lines, max_segment_length)

    # 5) Densify the vertices on each line. This is important for RiverMapper to
    #    accurately represent the river geometry. The resolution is set to 20 m,
    #    which can be adjusted based on your needs.
    print('densifying lines')
    for index, row in lines.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            lines.at[index, 'geometry'] = densify_linestring(geom, along_segment_resolution * 1e-5)
        else:
            raise ValueError(f"Geometry at index {index} is not a LineString")

    # 6) Add an attribute "keep = 1" to the new GeoDataFrame. This forces the lines
    #    to be expanded into river arcs in RiverMapper regardless of other criteria
    #    set in the RiverMapper configuration.
    print('adding keep = 1 attribute to the lines')
    lines['keep'] = 1

    # Save the result to a new shapefile
    lines.to_file(input_flowline.with_name(input_flowline.stem + f"_processed.shp"))


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


if __name__ == "__main__":
    '''
    Example usage of the pre_process_nhdflowlines function.
    This will preprocess the NHD flowline shapefile and save the result to a new shapefile.

    It is recommended to clip the NHD flowline and NHD area to the area of interest
    before running this function to avoid processing too many lines.
    '''
    pre_process_nhdflowlines(
        input_flowline=Path("/sciclone/schism10/Hgrid_projects/Wilmington/Shapefiles/nhdflowline_wilmington.shp"),
        input_nhdarea=Path("/sciclone/schism10/Hgrid_projects/Wilmington/Shapefiles/nhdarea_wilmington.shp"),
        line_identifier='gnis_id',  # use gnis_id to select lines
        max_segment_length=15,  # split segments with a maximum segment length in kilometers
        along_segment_resolution=20,  # densify segments with a resolution in meters, original points are retained
        diag_output=True  # set to True to output diagnostic shapefiles
    )
    print('Done!')
