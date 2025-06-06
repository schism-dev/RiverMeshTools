"""
This script provides a driver for grouping thalwegs based on their parent DEM tiles,
then allocate groups to mpi cores,
and finally calls the function "make_river_map" to sequentially process each group on each core

Usage:
Import and call the function "river_map_mpi_driver",
see sample_parallel.py in the sample application:
http://ccrm.vims.edu/yinglong/feiye/Public/RiverMapper_Samples.tar
"""


# Standard Library Imports
import time
from glob import glob
import pickle
from pathlib import Path

# Third-Party Imports
from mpi4py import MPI
import numpy as np
import geopandas as gpd

# Application-Specific Imports
from RiverMapper.config_logger import logger
from RiverMapper.config_river_map import ConfigRiverMap
from RiverMapper.river_map_tif_preproc import find_thalweg_tile, Tif2XYZ
from RiverMapper.make_river_map import make_river_map, geos2SmsArcList, clean_arcs, output_ocsmesh
from RiverMapper.SMS import merge_maps, SMS_MAP, get_all_points_from_shp
from RiverMapper.util import silentremove

# import warnings
# warnings.filterwarnings("error", category=UserWarning)
logger.setLevel('INFO')


def my_mpi_idx(ntasks, size, rank):
    '''
    Distribute ntasks tasks to {size} ranks.
    The return value is a bool vector of the shape (ntasks, ),
    with True indices indicating tasks for the current rank.
    '''
    i_my_groups = np.zeros((ntasks, ), dtype=bool)
    groups = np.array_split(range(ntasks), size)  # n_per_rank, _ = divmod(ntasks, size)
    my_group_ids = groups[rank]
    i_my_groups[my_group_ids] = True
    return my_group_ids, i_my_groups


def rename_single_core_outputs(output_dir):
    '''
    Rename the output files from a single-core run
    '''
    logger.info('\n------------------ renaming single-core outputs --------------\n')
    # rename the output files
    for file in glob(f'{output_dir}/Group_0_0_0_*'):
        if 'total' in file:
            new_name = file.replace('Group_0_0_0_', '')
        else:
            new_name = file.replace('Group_0_0_0_', 'total_')
        Path(file).rename(new_name)


def merge_dry_run_outputs(output_dir):
    '''
    Merge outputs from all cores for dry run
    '''
    logger.info('\n------------------ Merging outputs from all cores for dry run --------------\n')
    time_merge_start = time.time()
    
    # shapefiles
    valid_thalwegs = glob(f'{output_dir}/*valid_thalwegs.shp')
    if len(valid_thalwegs) > 0:
        gpd.pd.concat([gpd.read_file(x).to_crs('epsg:4326') for x in valid_thalwegs]).to_file(
            f'{output_dir}/total_valid_thalwegs.shp')
    logger.info('Merging outputs took: %s seconds.', time.time()-time_merge_start)


def merge_outputs(output_dir):
    '''
    Merge outputs from all cores
    '''
    logger.info('\n------------------ Merging outputs from all cores --------------\n')
    time_merge_start = time.time()

    # sms maps
    total_arcs_map = merge_maps(mapfile_glob_str=f'{output_dir}/*_total_arcs.map',
                                merged_fname=f'{output_dir}/total_arcs.map')

    total_intersection_joints = merge_maps(
        mapfile_glob_str=f'{output_dir}/*intersection_joints*.map',
        merged_fname=f'{output_dir}/total_intersection_joints.map')
    if total_intersection_joints is not None:
        total_intersection_joints = total_intersection_joints.detached_nodes

    total_river_map = merge_maps(f'{output_dir}/*river_arcs.map', merged_fname=f'{output_dir}/total_river_arcs.map')
    # total_inner_map = merge_maps(f'{output_dir}/*inner_arcs.map', merged_fname=f'{output_dir}/total_inner_arcs.map')
    total_dummy_map = merge_maps(f'{output_dir}/*dummy_arcs.map', merged_fname=f'{output_dir}/total_dummy_arcs.map')

    total_river_arcs = None
    if total_river_map is not None:
        total_river_arcs = total_river_map.arcs

    # for feeder channels, "this_nrow_arcs" is saved as z values in the map file
    merge_maps(f'{output_dir}/*river_arcs_extra.map', merged_fname=f'{output_dir}/total_river_arcs_extra.map')
    merge_maps(f'{output_dir}/*river_arcs_z.map', merged_fname=f'{output_dir}/total_river_arcs_z.map')

    total_centerlines = merge_maps(f'{output_dir}/*centerlines.map', merged_fname=f'{output_dir}/total_centerlines.map')
    merge_maps(f'{output_dir}/*bank_final*.map', merged_fname=f'{output_dir}/total_banks_final.map')
    merge_maps(f'{output_dir}/*original_banks.map', merged_fname=f'{output_dir}/total_original_banks.map')
    merge_maps(f'{output_dir}/*final_thalweg.map', merged_fname=f'{output_dir}/total_final_thalweg.map')
    merge_maps(f'{output_dir}/*corrected_thalweg.map', merged_fname=f'{output_dir}/total_corrected_thalweg.map')

    # # shapefiles
    river_outline_files = glob(f'{output_dir}/*_river_outline.shp')
    if len(river_outline_files) > 0:
        gpd.pd.concat([gpd.read_file(x).to_crs('epsg:4326') for x in river_outline_files]).to_file(
            f'{output_dir}/total_river_outline.shp')

    logger.info('Merging outputs took: %s seconds.', time.time()-time_merge_start)
    return [total_arcs_map, total_intersection_joints, total_river_arcs, total_centerlines, total_dummy_map]


def river_map_mpi_driver(
    dems_json_file='./dems.json',  # files for all DEM tiles
    thalweg_shp_fname='',
    output_dir='./',
    river_map_config=None,
    min_thalweg_buffer=1000,
    cache_folder='./Cache/',
    comm=MPI.COMM_WORLD
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

    # i_grouping_cache: Whether or not to read grouping info from cache,
    # which is useful when the same DEMs and thalweg_shp_fname are used.
    # A cache file named "dems_json_file + thalweg_shp_fname_grouping.cache" will be saved
    # regardless of the option value (the option only controls cache reading).
    # This is usually fast even without reading cache.
    i_grouping_cache = True
    cache_folder = Path(cache_folder)
    thalweg_shp_fname = Path(thalweg_shp_fname)
    output_dir = Path(output_dir)

    # Sometimes dems_json_file is not provided, e.g., for levees
    if dems_json_file is not None:
        dems_json_file = Path(dems_json_file)

    # configurations (parameters) for make_river_map()
    if river_map_config is None:
        river_map_config = ConfigRiverMap()  # use default configurations

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        time_start = time_grouping_start = time.time()
        logger.info('\n---------------------------------grouping thalwegs---------------------------------\n')

    comm.Barrier()

    thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs = None, None, None

    if rank == 0:
        if dems_json_file is not None:
            logger.info('A total of %s core(s) used.', size)
            silentremove(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if i_grouping_cache:
                cache_folder.mkdir(parents=True, exist_ok=True)
                cache_name = Path(
                    f'{cache_folder}/{dems_json_file.name}_{thalweg_shp_fname.name}_grouping.cache')
                try:
                    with open(cache_name, 'rb') as file:
                        logger.info('Reading grouping info from cache ...')
                        thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs = pickle.load(file)
                except FileNotFoundError:
                    logger.info(
                        "Grouping cache does not exist at %s. Cache will be generated after grouping.",
                        cache_folder)

            if thalwegs2tile_groups is None:
                thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs = find_thalweg_tile(
                    dems_json_file=dems_json_file,
                    thalweg_shp_fname=thalweg_shp_fname,
                    thalweg_buffer=thalweg_buffer,
                    silent=bool(rank),  # only rank 0 prints to screen
                )
                if i_grouping_cache:
                    with open(cache_name, 'wb') as file:
                        pickle.dump([thalwegs2tile_groups, tile_groups_files, tile_groups2thalwegs], file)

            if np.all(np.equal(tile_groups_files, None)):
                raise ValueError(f'No DEM tiles found for the thalwegs in {thalweg_shp_fname}')
        else:  # accomodate the case when dems_json_file is not provided, e.g., for levees
            n_group = size * 10
            _, l2g, _, _ = get_all_points_from_shp(thalweg_shp_fname)
            n_thalweg = len(l2g)
            # build groups by evenly dividing n_thalweg thalwegs into n_group groups
            tile_groups2thalwegs = np.array(np.array_split(range(n_thalweg), n_group), dtype=object)
            # map each thalweg to a group
            thalwegs2tile_groups = np.zeros((n_thalweg, ), dtype=int)
            for i, tile_group2thalwegs in enumerate(tile_groups2thalwegs):
                thalwegs2tile_groups[tile_group2thalwegs.astype(int)] = i
            # dummy files for each group
            tile_groups_files = np.array([[None] for _ in range(n_group)])

    thalwegs2tile_groups = comm.bcast(thalwegs2tile_groups, root=0)
    tile_groups_files = comm.bcast(tile_groups_files, root=0)
    tile_groups2thalwegs = comm.bcast(tile_groups2thalwegs, root=0)

    if rank == 0:
        logger.info('Thalwegs are divided into %d groups.', len(tile_groups2thalwegs))
        for i, tile_group2thalwegs in enumerate(tile_groups2thalwegs):
            logger.info(
                '\n[ Group %d (group id starting from 0) ]'
                '-----------------------------------------------------------------------\n'
                'Group %d includes the following thalwegs (idx starts from 0): %s\n'
                'Group %d needs the following DEMs: %s\n',
                i, i, tile_group2thalwegs, i, tile_groups_files[i])
        logger.info('Grouping took: %s seconds', time.time()-time_grouping_start)

    comm.barrier()
    if rank == 0:
        logger.info('\n---------------------------------Caching DEM tiles---------------------------------\n')
    comm.barrier()

    # Leveraging MPI to cache DEM tiles in parallel.
    # This is useful when there a large number of large DEM tiles.
    # If there are existing cache files, try reading from them;
    # if not successful, then regenerate the cache.
    # The actual reading of DEM caches is done in make_river_map()
    # This is why both this driver and make_river_map() have the same i_DEM_cache option
    if dems_json_file is not None and river_map_config.optional['i_DEM_cache']:
        unique_tile_files = []
        for group in tile_groups_files:
            for file in group:
                if (file not in unique_tile_files) and (file is not None):
                    unique_tile_files.append(file)
        unique_tile_files = np.array(unique_tile_files)

        for tif_fname in unique_tile_files[my_mpi_idx(len(unique_tile_files), size, rank)[0]]:
            _, is_new_cache = Tif2XYZ(tif_fname=tif_fname)
            if is_new_cache:
                logger.info('[Rank: %s cached DEM %s', rank, tif_fname)
            else:
                logger.info('[Rank: %s validated existing cache for %s', rank, tif_fname)

    comm.Barrier()
    if rank == 0:
        logger.info('\n---------------------------------assign groups to each core---------------------------------\n')
    comm.Barrier()

    my_group_ids, i_my_groups = my_mpi_idx(len(tile_groups_files), size, rank)
    my_tile_groups = tile_groups_files[i_my_groups]
    my_tile_groups_thalwegs = tile_groups2thalwegs[i_my_groups]
    logger.info('Rank %s handles Group %s\n', rank, np.squeeze(np.argwhere(i_my_groups)))

    comm.Barrier()
    if rank == 0:
        logger.info('\n---------------------------------beginning map generation---------------------------------\n')
    comm.Barrier()
    time_all_groups_start = time.time()

    for i, (my_group_id, my_tile_group, my_tile_group_thalwegs) in enumerate(
        zip(my_group_ids, my_tile_groups, my_tile_groups_thalwegs)
    ):
        # if my_group_id != 45:
        #     continue  # temporary testing

        time_this_group_start = time.time()
        logger.info('Rank %s: Group %s (global: %s) started ...', rank, i, my_group_id)
        # update some parameters in the config file
        river_map_config.optional['output_prefix'] = f'Group_{my_group_id}_{rank}_{i}_'
        river_map_config.optional['mpi_print_prefix'] = (
            f'[Rank {rank}, Group {i+1} of {len(my_tile_groups)}, local id: {i}, global: {my_group_id}] ')
        river_map_config.optional['selected_thalweg'] = my_tile_group_thalwegs
        make_river_map(
            tif_fnames=my_tile_group,
            thalweg_shp_fname=thalweg_shp_fname,
            output_dir=output_dir,
            **river_map_config.optional,  # pass all optional parameters with a dictionary
        )

        logger.info(
            'Rank %s: Group %s (global: %s) run time: %s seconds.',
            rank, i, my_group_id, time.time()-time_this_group_start)

    logger.info('Rank %s: total run time: %s seconds.', rank, time.time()-time_all_groups_start)

    comm.Barrier()

    # finalize
    if rank == 0:
        if river_map_config.optional['dry_run_only']:
            merge_dry_run_outputs(output_dir)
        else:  # merge outputs from all ranks
            total_arcs_map, _, _, _, _ = merge_outputs(output_dir)

            logger.info('\n--------------- Final clean-ups --------------------------------------------------------\n')
            time_final_cleanup_start = time.time()

            total_arcs_cleaned = clean_arcs(
                [arc for arc in total_arcs_map.to_GeoDataFrame().geometry.unary_union.geoms],
                n_clean_iter=river_map_config.optional['n_clean_iter'],
                snap_point_reso_ratio=river_map_config.optional['snap_point_reso_ratio'],
                snap_arc_reso_ratio=river_map_config.optional['snap_arc_reso_ratio']
            )

            SMS_MAP(arcs=geos2SmsArcList(total_arcs_cleaned)).writer(filename=f'{output_dir}/total_arcs.map')

            gpd.GeoDataFrame(
                index=range(len(total_arcs_cleaned)), crs='epsg:4326', geometry=total_arcs_cleaned
            ).to_file(filename=f'{output_dir}/total_arcs.shp', driver="ESRI Shapefile")

            logger.info('Final clean-ups took: %s seconds.', time.time()-time_final_cleanup_start)

            # outputs for OCSMesh
            if river_map_config.optional['i_OCSMesh']:
                output_ocsmesh(output_dir, area_thres=0.85)

        # delete per-core outputs
        silentremove(glob(f'{output_dir}/Group*'))
        logger.info('>>>>>>>> Total run time: %s seconds >>>>>>>>', time.time()-time_start)
