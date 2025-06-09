"""
This script provides classes and methods for processing tif files.
"""


import os
from glob import glob
from pathlib import Path
import errno
import copy
import pickle
import json
import math
from dataclasses import dataclass

import numpy as np
from osgeo import gdal
import geopandas as gpd
from shapely import Polygon
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from RiverMapper.SMS import lonlat2cpp, cpp2lonlat, get_all_points_from_shp
from RiverMapper.util import silentremove


gdal.UseExceptions()


@dataclass
class DemData():
    '''
    Simple class to store DEM data
    '''
    x: np.ndarray
    y: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    elev: np.ndarray
    dx: float
    dy: float


def gen_splitter(dem_box, dl, overlap_ratio=0.01, crs='EPSG:4326'):
    '''
    Generate a splitter for splitting a large DEM file into smaller tiles

    Input:
    - dem_box: bounding box of the DEM file, [xmin, ymin, xmax, ymax]
    - dl: side length of each square tile
    - overlap_ratio: overlap ratio between tiles

    Output:
    - splitter: list of bounding boxes of tiles
    '''
    splitter = []
    for x in np.arange(dem_box[0], dem_box[2], dl):
        for y in np.arange(dem_box[1], dem_box[3], dl):
            splitter.append([
                x - dl * overlap_ratio, y - dl * overlap_ratio,
                x + dl + dl * overlap_ratio, y + dl + dl * overlap_ratio
            ])

    # make a gpd dataframe for the splitter
    gdf = gpd.GeoDataFrame(
        geometry=[Polygon([(x[0], x[1]), (x[2], x[1]), (x[2], x[3]), (x[0], x[3])]) for x in splitter], crs=crs)

    return splitter, gdf


def split_vector_shp(shp_fname, splitter_gdf, outdir='./split/'):
    '''
    Split a vector shapefile into smaller tiles

    Inputs:
    - shp_fname: input shapefile name, consisting vectors such as linestrings or polygons
    - splitter_gdf: a GeoDataFrame of bounding boxes of tiles
    - outdir: output directory
    '''

    shp_fname = Path(shp_fname)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_shp_gdf = gpd.read_file(shp_fname)

    split_shp_fnames = []
    for i, splitter in enumerate(splitter_gdf.geometry):
        clipped_gdf = gpd.clip(input_shp_gdf, splitter)
        if clipped_gdf.empty:
            print(f'skip empty tile {i}')
            continue
        split_shp_fnames.append(f'{outdir}/{shp_fname.stem}_{i}.shp')
        clipped_gdf.to_file(split_shp_fnames[-1])

    return split_shp_fnames


def rasterize_shp(shp_fname, burn_value=-1, pixel_size=2e-5):
    '''
    Rasterize a shapefile into a tif file

    Inputs:
    -burn_value: value to burn into the rasterized tif file
    -dl: resolution of the tif file in degrees
    '''
    shp_fname = Path(shp_fname)

    gdf = gpd.read_file(shp_fname)

    # Define raster properties
    minx, miny, maxx, maxy = gdf.total_bounds  # Get the bounding box of the shapefile
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)  # Transform for rasterization

    # Prepare a list of geometries and values
    shapes = [(geom, burn_value) for geom in gdf.geometry]  # Inside value is -1

    # Rasterize the polygons
    raster = rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int16')

    # Save the raster to a file
    with rasterio.open(
        shp_fname.with_suffix(".tif"), 'w', driver='GTiff', height=height, width=width,
        count=1, dtype='int16', crs=gdf.crs, transform=transform,
    ) as dst:
        dst.write(raster, 1)


def parse_dem_tiles(dem_code, dem_tile_digits):
    '''
    Parse a dem_code into the original DEM id.
    A unique code is assigned to all parent tiles (from the same DEM source) of a thalweg, e.g.:
    327328329 is actually Tile No. 327, 328, 329
    Normally, a thalweg has <= 4 parent tiles (1 mostly; > 2 at the boundary of DEM tiles);
    n_tiles > 4 will generate an exception.
    '''
    if dem_code == 0:
        return [-1]  # no DEM found

    dem_tile_ids = []
    n_tiles = int(math.log10(dem_code)/dem_tile_digits) + 1
    if n_tiles > 4:
        raise ValueError(
            "Some thalweg points belong to more than 4 tiles from one DEM source, "
            "you may need to clean up the DEM tiles first.")
    for digit in reversed(range(n_tiles)):
        x, dem_code = divmod(dem_code, 10**(digit*dem_tile_digits))
        dem_tile_ids.append(int(x-1))
    return dem_tile_ids


def get_tif_box(tif_fname=None):
    '''
    Get the bounding box of a tif file
    '''
    src = gdal.Open(tif_fname)
    ulx, xres, _, uly, _, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    return [ulx, lry, lrx, uly]


def Tif2XYZ(tif_fname=None, cache=True):
    '''
    Read a tif file and return a DemData object,
    which includes x, y, lon, lat, z, dx, dy
    '''
    is_new_cache = False

    cache_name = tif_fname + '.pkl'

    if cache:
        try:
            with open(cache_name, 'rb') as f:
                dem_data = pickle.load(f)
                return [dem_data, is_new_cache]  # cache successfully read
        except (ModuleNotFoundError, AttributeError, pickle.UnpicklingError) as e:
            print(f'Warning: failed to read cache: {e}')
            print('removing existing cache and regenerating it ...')
            silentremove(cache_name)
        except OSError as e:
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise e

    # read from raw tif and generate cache
    ds = gdal.Open(tif_fname, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)

    width = ds.RasterXSize
    height = ds.RasterYSize

    gt = ds.GetGeoTransform()
    TL_x, TL_y = gt[0], gt[3]

    # showing a 2D image of the topo
    # plt.imshow(elevation, cmap='gist_earth',extent=[minX, maxX, minY, maxY])
    # plt.show()

    z = band.ReadAsArray()

    dx = gt[1]
    dy = gt[5]
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError()

    x_idx = np.array(range(width))
    y_idx = np.array(range(height))
    xp = dx * x_idx + TL_x + dx/2
    yp = dy * y_idx + TL_y + dy/2

    ds = None  # close dataset

    dem_data = DemData(xp, yp, xp, yp, z, dx, dy)

    if cache:
        with open(cache_name, 'wb') as f:
            pickle.dump(dem_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        is_new_cache = True

    return [dem_data, is_new_cache]  # already_cached = False


def reproject_tifs(tif_files: list, src_crs='EPSG:4326', dst_crs='EPSG:26917', outdir='./'):
    '''
    The function failed on CRM tiles, for which use the cmd line tool gdalwarp, e.g.:
    gdalwarp -s_srs EPSG:4326 -t_srs EPSG:26917 -of GTiff ../Lonlat/crm_vol5.tif crm_vol5.26917.tif
    '''
    for i, tif_file in enumerate(tif_files):
        print(f'reprojecting tifs: {i+1} of {len(tif_files)}, {tif_file}')
        epsg = dst_crs.split(':')[1]
        tif_outfile = outdir + os.path.basename(tif_file).split('.')[0] + '.' + epsg + '.tif'
        if not os.path.exists(tif_outfile):
            gdal.Warp(tif_outfile, tif_file, srcSRS=src_crs, dstSRS=dst_crs)


def pts_in_box(pts, box):
    '''
    Simple function to check if a point is in a box
    '''
    in_box = (pts[:, 0] > box[0]) * (pts[:, 0] <= box[2]) * \
             (pts[:, 1] > box[1]) * (pts[:, 1] <= box[3])
    return in_box


def get_pt_idx_from_tile(dem_data, lon, lat):
    '''
    return nearest index (i, j) in DEM mesh for point (x, y),
    assuming lon/lat, not projected coordinates
    '''
    dx = dem_data.lon[1] - dem_data.lon[0]
    dy = dem_data.lat[1] - dem_data.lat[0]
    i = (np.round((lon - dem_data.lon[0]) / dx)).astype(int)
    j = (np.round((lat - dem_data.lat[0]) / dy)).astype(int)

    valid = (i < dem_data.lon.shape) * (j < dem_data.lat.shape) * (i >= 0) * (j >= 0)
    return [i, j], valid


def get_elev_from_tiles(x_cpp, y_cpp, tile_list, scale=1.0, valid_range=None):
    '''
    x: vector of x coordinates, assuming cpp;
    y: vector of x coordinates, assuming cpp;
    tile_list: list of DEM tiles (in DemData type, defined in river_map_tif_preproc)
    scale: scale factor for elevation; use -1 to invert the elevation, e.g., for barrier islands
    valid_range: values outside valid elevation range will be set to nan
    '''
    if valid_range is None:
        valid_range = [-1e3, 1e3]

    lon, lat = cpp2lonlat(x_cpp, y_cpp)

    elevs = np.full_like(lon, np.nan, dtype=float)
    for dem_data in tile_list:
        [j, i], in_box = get_pt_idx_from_tile(dem_data, lon, lat)

        # only update valid entries that are not already set (i.e. nan at this step) and in DEM box
        idx = (np.isnan(elevs) * in_box).astype(bool)
        elevs[idx] = dem_data.elev[i[idx], j[idx]]

        # set invalid values to nan
        elevs[(elevs < valid_range[0]) | (elevs > valid_range[1])] = np.nan

    if np.isnan(elevs).any():
        # raise ValueError('failed to find elevation')
        return None
    else:
        return elevs * scale


def find_parent_box(pts, boxes):
    '''
    Find parent boxes for each point in pts,
    and translate the boxes into a tile code

    Input:
    - pts: n x 2 array of points
    - boxes: list of bounding boxes of tiles

    Output:
    - parent: n x 1 array of tile codes
    '''

    # number of digits needed for representing tile id, e.g., CuDEM (819 tiles) needs 3 digits
    ndigits = int(math.log10(len(boxes))) + 1
    parent = np.zeros((len(pts), 1), dtype='int')
    digits = np.zeros((len(pts), 1), dtype='int')
    for j, box in enumerate(boxes):
        in_box = pts_in_box(pts[:, :2], box)
        # save multiple tiles in an integer, e.g.,
        # 100101 of CuDEM (819 tiles) means tile 100 and tile 101
        # 12 of CRM (6 tiles) means tile 1 and tile 2
        parent[in_box] += ((j+1) * 10.0 ** (digits[in_box])).astype(int)
        digits[in_box] += ndigits

    # plt.hist(parent, bins=len(np.unique(parent)))
    # np.savetxt('thalweg_parent.xyz', np.c_[pts[:,:2], parent])

    return parent


def tile2dem_file(dem_dict, dem_order, tile_code):
    '''
    Interpret tile code to DEM file name
    '''

    dem_id, tile_id = int(tile_code.real), int(tile_code.imag)
    if tile_id != -1:
        return dem_dict[dem_order[dem_id]]['file_list'][tile_id]
    else:
        return None


def find_thalweg_tile(
    dems_json_file='dems.json',
    thalweg_shp_fname='/sciclone/schism10/feiye/STOFS3D-v5/Inputs/v14/GA_riverstreams_cleaned_utm17N.shp',
    thalweg_buffer=1000, cache_folder=None, silent=True
):
    '''
    Assign thalwegs to DEM tiles
    '''
    # read DEMs
    with open(dems_json_file, encoding='utf-8') as d:
        dem_dict = json.load(d)

    # get the box of each tile of each DEM
    dem_order = []
    for k, _ in dem_dict.items():
        if not silent:
            print(f"reading dem bounding box: {dem_dict[k]['name']}")
        dem_order.append(k)
        if cache_folder is None:  # set it to *.shp's folder
            cache_folder = os.path.dirname(os.path.abspath(dem_dict[k]['glob_pattern']))

        # if glob pattern returns any files, put them into the file_list
        dem_dict[k]['file_list'].extend(glob(dem_dict[k]['glob_pattern']))
        # get unique file names
        dem_dict[k]['file_list'] = list(set(dem_dict[k]['file_list']))

        # get box of each tile
        dem_dict[k]['boxes'] = [get_tif_box(x) for x in dem_dict[k]['file_list']]

    # read thalwegs
    print(f'Reading thalwegs from {thalweg_shp_fname} ...')
    xyz, l2g, _, perp = get_all_points_from_shp(thalweg_shp_fname)

    # find DEM tiles for all thalwegs' points
    print('Finding DEM tiles for each thalweg ...')
    x_cpp, y_cpp = lonlat2cpp(xyz[:, 0], xyz[:, 1])
    xt_right = x_cpp + thalweg_buffer * np.cos(perp)
    yt_right = y_cpp + thalweg_buffer * np.sin(perp)
    xt_left = x_cpp + thalweg_buffer * np.cos(perp + np.pi)
    yt_left = y_cpp + thalweg_buffer * np.sin(perp + np.pi)
    # find thalweg itself and two search boundaries (one on each side)
    thalwegs2dems = [find_parent_box(xyz[:, :2], dem_dict[k]['boxes']) for k in dem_dict.keys()]
    thalwegs_right2dems = [
        find_parent_box(np.array(cpp2lonlat(xt_right, yt_right)).T, dem_dict[k]['boxes']) for k in dem_dict.keys()]
    thalwegs_left2dems = [
        find_parent_box(np.array(cpp2lonlat(xt_left, yt_left)).T, dem_dict[k]['boxes']) for k in dem_dict.keys()]

    # how many digits in tile numbers
    dems_tile_digits = [int(math.log10(len(dem_dict[k]['boxes'])))+1 for k in dem_dict.keys()]

    # use cpp projection hereafter, because meter unit is easier for parameterization
    xyz[:, 0], xyz[:, 1] = x_cpp, y_cpp

    thalwegs = []
    thalwegs_parents = []
    for i, idx in enumerate(l2g):  # enumerate thalwegs
        line = xyz[idx, :]  # one segment of a thalweg
        thalwegs.append(line)

        thalweg_parents = []  # one thalweg can have parent tiles from all DEM sources
        for i_dem, [thalwegs2dem, thalwegs_left2dem, thalwegs_right2dem] in enumerate(
            zip(thalwegs2dems, thalwegs_right2dems, thalwegs_left2dems)
        ):
            # find all DEM tiles that a thalweg (including its left and right search boundaries) touches
            thalweg2dem = np.unique(
                np.r_[thalwegs2dem[idx], thalwegs_left2dem[idx], thalwegs_right2dem[idx]]).tolist()
            for dem_code in thalweg2dem:
                thalweg_parents += [complex(i_dem, x) for x in parse_dem_tiles(dem_code, dems_tile_digits[i_dem])]
        thalwegs_parents.append(thalweg_parents)

    # Group thalwegs: thalwegs from the same group have the same parent tiles
    print('Grouping thalwegs ...')
    groups = []
    group_id = 0
    thalweg2group = -np.ones((len(thalwegs)), dtype=int)
    for i, thalweg_parents in enumerate(thalwegs_parents):
        if thalweg_parents not in groups:
            groups.append(thalweg_parents)
            thalweg2group[i] = group_id
            group_id += 1
        else:
            for j, x in enumerate(groups):
                if x == thalweg_parents:
                    thalweg2group[i] = j

    # reduce groups: merge smaller groups into larger groups
    groups = np.array(groups, dtype=object)
    ngroup = len(groups)
    grp2large_grp = ngroup * np.ones((ngroup+1,), dtype=int)  # add a dummy mapping at the end
    for i1, group1 in enumerate(groups):
        for i2, group2 in enumerate(groups):
            if len(group1) < len(group2) and all(elem in group2 for elem in group1):
                # print(f'{group1} is contained in {group2}')
                grp2large_grp[i1] = i2
                break

    # But some large groups are still contained in larger groups,
    # get to the bottom of the family tree (e.g., parent's parent's parent ...)
    parents = np.squeeze(grp2large_grp[grp2large_grp])  # parent's parent
    idx = parents != len(groups)  # where parent's parent exists
    while any(idx):
        grp2large_grp[idx] = parents[idx]  # reset parent to parent's parent
        parents = parents[parents]  # advance family tree
        idx = parents != len(groups)  # get the idx where parent's parent still exists

    idx = grp2large_grp == len(groups)  # where parent's parent is non-existent
    grp2large_grp[idx] = np.arange(len(groups)+1)[idx]  # parent group is self
    grp2large_grp = grp2large_grp[:-1]  # remove the dummy group at the end

    large_groups = groups[np.unique(grp2large_grp)]
    if not silent:
        print(f'number of groups after reduction: {len(large_groups)}')
    group_lens = [len(x) for x in large_groups]
    if not silent:
        print(f'group lengths: min {min(group_lens)}; max {max(group_lens)}; mean {np.mean(group_lens)}')

    thalweg2group = grp2large_grp[thalweg2group]
    map_grp = dict(zip(np.unique(grp2large_grp), np.arange(len(np.unique(grp2large_grp)))))
    thalweg2large_group = np.array([map_grp[x] for x in thalweg2group])

    large_group2thalwegs = [[] for _ in range(len(large_groups))]
    for i, x in enumerate(thalweg2large_group):
        large_group2thalwegs[x].append(i)

    large_groups_files = copy.deepcopy(large_groups)
    for i, group in enumerate(large_groups):
        for j, tile_code in enumerate(group):
            large_groups_files[i][j] = tile2dem_file(dem_dict=dem_dict, dem_order=dem_order, tile_code=tile_code)

    # histogram
    # plt.hist(thalweg2large_group, bins=len(np.unique(thalweg2large_group)))
    # plt.show()

    return thalweg2large_group, large_groups_files, np.array(large_group2thalwegs, dtype=object)


def sample_rasterize_polygons(input_shp_fname):
    '''
    Make a splitter shapefile for splitting a large raster/vector shapefile into smaller tiles.
    Then split the large shapefile into smaller tiles.
    Finally, rasterize the smaller shapefiles into tif files.
    '''
    outdir = Path(f'{input_shp_fname.parent}/{input_shp_fname.stem}_split/')
    outdir.mkdir(parents=True, exist_ok=True)

    dem_box = gpd.read_file(input_shp_fname).total_bounds
    _, splitter_gdf = gen_splitter(dem_box, dl=0.5, overlap_ratio=0.01)

    splitter_gdf.to_file(f'{outdir}/splitter.shp')
    split_shps = split_vector_shp(input_shp_fname, splitter_gdf, outdir)
    for split_shp in split_shps:
        rasterize_shp(split_shp)
    print('Done!')


if __name__ == '__main__':
    SHP_FNAME = Path('/sciclone/schism10/Hgrid_projects/STOFS3D-v8/v46/Shapefiles/nhd_area_clipped.shp')
    sample_rasterize_polygons(SHP_FNAME)
