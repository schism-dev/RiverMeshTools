"""
This script provides a driver for grouping thalwegs based on their parent DEM tiles,
then allocate groups to mpi cores,
and finally calls the function "make_river_map" to sequentially process each group on each core

Usage:
Import and call the function "river_map_mpi_driver",
see sample_parallel.py in the installation directory.
"""


import os
import time
from mpi4py import MPI
from glob import glob
import numpy as np
import pickle
from pathlib import Path
import geopandas as gpd
from shapely.ops import polygonize
from RiverMapper.river_map_tif_preproc import find_thalweg_tile, Tif2XYZ
from RiverMapper.make_river_map import make_river_map, clean_intersections, geos2SmsArcList, Geoms_XY, clean_arcs
from RiverMapper.config_river_map import ConfigRiverMap
from RiverMapper.SMS import merge_maps, SMS_MAP
from RiverMapper.util import silentremove
import warnings

warnings.filterwarnings("error", category=UserWarning)


def my_mpi_idx(N, size, rank):
    '''
    Distribute N tasks to {size} ranks.
    The return value is a bool vector of the shape (N, ),
    with True indices indicating tasks for the current rank.
    '''
    i_my_groups = np.zeros((N, ), dtype=bool)
    groups = np.array_split(range(N), size)  # n_per_rank, _ = divmod(N, size)
    my_group_ids = groups[rank]
    i_my_groups[my_group_ids] = True
    return my_group_ids, i_my_groups

def merge_outputs(output_dir):
    print(f'\n------------------ merging outputs from all cores --------------\n')
    time_merge_start = time.time()

    # sms maps
    total_arcs_map = merge_maps(f'{output_dir}/*_total_arcs.map', merged_fname=f'{output_dir}/total_arcs.map')

    total_intersection_joints = merge_maps(f'{output_dir}/*intersection_joints*.map', merged_fname=f'{output_dir}/total_intersection_joints.map')
    if total_intersection_joints is not None:
        total_intersection_joints = total_intersection_joints.detached_nodes

    total_river_map = merge_maps(f'{output_dir}/*river_arcs.map', merged_fname=f'{output_dir}/total_river_arcs.map')

    total_dummy_map = merge_maps(f'{output_dir}/*dummy_arcs.map', merged_fname=f'{output_dir}/total_dummy_arcs.map')

    total_river_arcs = None
    if total_river_map is not None:
        total_river_arcs = total_river_map.arcs

    # for feeder channels, "this_nrow_arcs" is saved as z values in the map file
    merge_maps(f'{output_dir}/*river_arcs_extra.map', merged_fname=f'{output_dir}/total_river_arcs_extra.map')

    total_centerlines = merge_maps(f'{output_dir}/*centerlines.map', merged_fname=f'{output_dir}/total_centerlines.map')
    merge_maps(f'{output_dir}/*bank_final*.map', merged_fname=f'{output_dir}/total_banks_final.map')

    # # shapefiles
    river_outline_files = glob(f'{output_dir}/*_river_outline.shp')
    if len(river_outline_files) > 0:
        gpd.pd.concat([gpd.read_file(x).to_crs('epsg:4326') for x in river_outline_files]).to_file(f'{output_dir}/total_river_outline.shp')

    bomb_polygon_files = glob(f'{output_dir}/*_bomb_polygons.shp')
    if len(bomb_polygon_files) > 0:
        gpd.pd.concat([gpd.read_file(x).to_crs('epsg:4326') for x in bomb_polygon_files]).to_file(f'{output_dir}/total_bomb_polygons.shp')

    print(f'Merging outputs took: {time.time()-time_merge_start} seconds.')
    return [total_arcs_map, total_intersection_joints, total_river_arcs, total_centerlines, total_dummy_map]


def final_clean_up(output_dir, total_arcs_map, snap_points, i_blast_intersection=False, total_river_arcs=None):
    pass

def river_map_mpi_driver(
    dems_json_file = './dems.json',  # files for all DEM tiles
    thalweg_shp_fname='',
    output_dir = './',
    river_map_config = None,
    min_thalweg_buffer = 1000,
    cache_folder = './Cache/',
    comm = MPI.COMM_WORLD
):
    '''
    Driver for the parallel execution of make_river_map.py

    Thalwegs are grouped based on the DEM tiles associated with each thalweg.
    For each thalweg, its associated DEM tiles are those needed for determining
    the elevations on all thalweg points, as well as
    the elevations within a buffer zone along the thalweg
    (within which the positions of left and right banks will be searched)

    One core can be responsible for one or more thalweg groups,
    which are fed to make_river_map.py one at a time

    Summary of the input parameters:
    river_map_config: a ConfigRiverMap object to pass the arguments to make_river_map.py
    min_thalweg_buffer: in meters. This is the minimum search range on either side of the thalweg.
                    Because banks will be searched within this range,
                    its value is needed now to identify parent DEM tiles of each thalweg
    cache_folder: folder for saving the cache file for thalweg grouping
    '''

    thalweg_buffer = max(min_thalweg_buffer, river_map_config.optional['river_threshold'][-1])

    # deprecated (fast enough without caching)
    i_thalweg_cache = False  # Whether or not to read thalweg info from cache.
                             # The cache file saves coordinates, index, curvature, and direction at all thalweg points
    # i_grouping_cache: Whether or not to read grouping info from cache,
    #                   which is useful when the same DEMs and thalweg_shp_fname are used.
    #                   A cache file named "dems_json_file + thalweg_shp_fname_grouping.cache" will be saved
    #                   regardless of the option value (the option only controls cache reading).
    #                   This is usually fast even without reading cache.
    i_grouping_cache = True
    cache_folder = Path(cache_folder)
    thalweg_shp_fname = Path(thalweg_shp_fname)
    output_dir = Path(output_dir)
    dems_json_file = Path(dems_json_file)

    # configurations (parameters) for make_river_map()
    if river_map_config is None:
        river_map_config = ConfigRiverMap()  # use default configurations

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        time_start = time_grouping_start = time.time()
        print('\n---------------------------------grouping thalwegs---------------------------------\n')

    comm.Barrier()

    thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs = None, None, None
    if rank == 0:
        print(f'A total of {size} core(s) used.')
        silentremove(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if i_grouping_cache:
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_name = Path(f'{cache_folder}/{dems_json_file.name}_{thalweg_shp_fname.name}_grouping.cache')
            try:
                with open(cache_name, 'rb') as file:
                    print(f'Reading grouping info from cache ...')
                    thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs = pickle.load(file)
            except FileNotFoundError:
                print(f"Grouping cache does not exist at {cache_folder}. Cache will be generated after grouping.")

        if thalwegs2tile_groups is None:
            thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs = find_thalweg_tile(
                dems_json_file=dems_json_file,
                thalweg_shp_fname=thalweg_shp_fname,
                thalweg_buffer = thalweg_buffer,
                iNoPrint=bool(rank), # only rank 0 prints to screen
                i_thalweg_cache=i_thalweg_cache
            )
            if i_grouping_cache:
                with open(cache_name, 'wb') as file:
                    pickle.dump([thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs], file)

    if np.all(np.equal(tile_groups_files, None)):
        raise ValueError(f'No DEM tiles found for the thalwegs in {thalweg_shp_fname}')

    thalwegs2tile_groups = comm.bcast(thalwegs2tile_groups, root=0)
    tile_groups_files = comm.bcast(tile_groups_files, root=0)
    tile_groups2thalwegs = comm.bcast(tile_groups2thalwegs, root=0)

    if rank == 0:
        print(f'Thalwegs are divided into {len(tile_groups2thalwegs)} groups.')
        for i, tile_group2thalwegs in enumerate(tile_groups2thalwegs):
            print(f'[ Group {i+1} ]-----------------------------------------------------------------------\n' + \
                  f'Group {i+1} includes the following thalwegs (idx starts from 0): {tile_group2thalwegs}\n' + \
                  f'Group {i+1} needs the following DEMs: {tile_groups_files[i]}\n')
        print(f'Grouping took: {time.time()-time_grouping_start} seconds')

    comm.barrier()
    if rank == 0: print('\n---------------------------------caching DEM tiles---------------------------------\n')
    comm.barrier()

    # Leveraging MPI to cache DEM tiles in parallel.
    # This is useful when the DEMs are large and there are many of them.
    # And when there are existing cache files, try reading from them;
    # if the reading is not successful, then regenerate the cache for the corresponding DEM tiles.
    # The actual reading of DEM caches is done in make_river_map()
    # This is why both this driver and make_river_map() have the same i_DEM_cache option
    if river_map_config.optional['i_DEM_cache']:
        unique_tile_files = []
        for group in tile_groups_files:
            for file in group:
                if (file not in unique_tile_files) and (file is not None):
                    unique_tile_files.append(file)
        unique_tile_files = np.array(unique_tile_files)

        for tif_fname in unique_tile_files[my_mpi_idx(len(unique_tile_files), size, rank)[0]]:
            _, is_new_cache = Tif2XYZ(tif_fname=tif_fname)
            if is_new_cache:
                print(f'[Rank: {rank} cached DEM {tif_fname}')
            else:
                print(f'[Rank: {rank} validated existing cache for {tif_fname}')

    comm.Barrier()
    if rank == 0: print('\n---------------------------------assign groups to each core---------------------------------\n')
    comm.Barrier()

    my_group_ids, i_my_groups = my_mpi_idx(N=len(tile_groups_files), size=size, rank=rank)
    my_tile_groups = tile_groups_files[i_my_groups]
    my_tile_groups_thalwegs = tile_groups2thalwegs[i_my_groups]
    print(f'Rank {rank} handles Group {np.squeeze(np.argwhere(i_my_groups))}\n')

    comm.Barrier()
    if rank == 0: print('\n---------------------------------beginning map generation---------------------------------\n')
    comm.Barrier()
    time_all_groups_start = time.time()

    for i, (my_group_id, my_tile_group, my_tile_group_thalwegs) in enumerate(zip(my_group_ids, my_tile_groups, my_tile_groups_thalwegs)):
        time_this_group_start = time.time()
        print(f'Rank {rank}: Group {i+1} (global: {my_group_id}) started ...')
        # update some parameters in the config file
        river_map_config.optional['output_prefix'] = f'Group_{my_group_id}_{rank}_{i}_'
        river_map_config.optional['mpi_print_prefix'] = f'[Rank {rank}, Group {i+1} of {len(my_tile_groups)}, global: {my_group_id}] '
        river_map_config.optional['selected_thalweg'] = my_tile_group_thalwegs
        make_river_map(
            tif_fnames = my_tile_group,
            thalweg_shp_fname = thalweg_shp_fname,
            output_dir = output_dir,
            **river_map_config.optional,  # pass all optional parameters with a dictionary
        )

        print(f'Rank {rank}: Group {i+1} (global: {my_group_id}) run time: {time.time()-time_this_group_start} seconds.')

    print(f'Rank {rank}: total run time: {time.time()-time_all_groups_start} seconds.')

    comm.Barrier()

    # finalize
    if rank == 0:
        # merge outputs from all ranks
        total_arcs_map, total_intersection_joints, total_river_arcs, total_centerlines, total_dummy_map = merge_outputs(output_dir)

        print(f'\n--------------- final clean-ups --------------------------------------------------------\n')
        time_final_cleanup_start = time.time()

        total_arcs_cleaned = [arc for arc in total_arcs_map.to_GeoDataFrame().geometry.unary_union.geoms]
        if not river_map_config.optional['i_blast_intersection']:
            if os.path.exists(f'{output_dir}/total_bomb_polygons.shp'):
                bomb_polygons = gpd.read_file(f'{output_dir}/total_bomb_polygons.shp')
            else:
                bomb_polygons = None
            total_arcs_cleaned = clean_intersections(
                arcs=total_arcs_cleaned, target_polygons=bomb_polygons, snap_points=total_intersection_joints,
                i_OCSMesh=river_map_config.optional['i_OCSMesh'],
                idummy=river_map_config.optional['i_pseudo_channel']==1,
            )
        total_arcs_cleaned = clean_arcs(
            total_arcs_cleaned, i_real_clean=river_map_config.optional['i_real_clean'],
            snap_point_reso_ratio=river_map_config.optional['snap_point_reso_ratio'],
            snap_arc_reso_ratio=river_map_config.optional['snap_arc_reso_ratio'],
        )

        # merge dummy arcs into total_arcs_cleaned
        if len(total_dummy_map.arcs) > 0:
            total_arcs_cleaned = total_arcs_cleaned + total_dummy_map.arcs
            total_arcs_cleaned = [arc for arc in total_arcs_map.to_GeoDataFrame().geometry.unary_union.geoms]

        SMS_MAP(arcs=geos2SmsArcList(total_arcs_cleaned)).writer(filename=f'{output_dir}/total_arcs.map')

        gpd.GeoDataFrame(
            index=range(len(total_arcs_cleaned)), crs='epsg:4326', geometry=total_arcs_cleaned
        ).to_file(filename=f'{output_dir}/total_arcs.shp', driver="ESRI Shapefile")

        # outputs for OCSMesh
        if river_map_config.optional['i_OCSMesh']:
            total_arcs_cleaned_polys = [poly for poly in polygonize(gpd.GeoSeries(total_arcs_cleaned))]
            gpd.GeoDataFrame(
                index=range(len(total_arcs_cleaned_polys)), crs='epsg:4326', geometry=total_arcs_cleaned_polys
            ).to_file(filename=f'{output_dir}/total_polys_for_OCSMesh.shp', driver="ESRI Shapefile")

        # river_arcs_cleaned = clean_river_arcs(total_river_arcs, total_arcs_cleaned)
        # total_river_outline_polys = generate_river_outline_polys(river_arcs_cleaned)
        # if len(total_river_outline_polys) > 0:
        #     gpd.GeoDataFrame(
        #         index=range(len(total_river_outline_polys)), crs='epsg:4326', geometry=total_river_outline_polys
        #     ).to_file(filename=f'{output_dir}/total_river_outline.shp', driver="ESRI Shapefile")
        # else:
        #     print(f'Warning: total_river_outline_polys empty')

        print(f'Final clean-ups took: {time.time()-time_final_cleanup_start} seconds.')

        # delete per-core outputs
        silentremove(glob(f'{output_dir}/Group*'))
        print(f'>>>>>>>> Total run time: {time.time()-time_start} seconds >>>>>>>>')
