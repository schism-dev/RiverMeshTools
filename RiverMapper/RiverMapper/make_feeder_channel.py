"""
This script generates feeder channels for the river map.
"""

from pathlib import Path
import pickle
from time import time

import numpy as np
import geopandas as gpd

from RiverMapper.util import z_decoder, z_encoder
from RiverMapper.SMS import SMS_MAP, SMS_ARC, get_perpendicular_angle


class Feeder():
    """
    A class to represent a feeder channel.
    """
    def __init__(self, points_x: np.ndarray, points_y: np.ndarray, base_id) -> None:
        # feeder channel points coordinates
        # (n_along_river_arcs, n_points_along_each_feeder_channel)
        self.points_x = points_x
        self.points_y = points_y
        self.head = None  # center of the head cross-river arc
        self.base = None  # center of the base cross-river arc
        
        # populate head and base
        self.head = np.c_[np.mean(points_x[:, 0]), np.mean(points_y[:, 0])]
        # adjust base_id to be within the point range of the feeder channel
        base_id = min(self.points_x.shape[1] - 1, base_id)
        self.base = np.c_[np.mean(points_x[:, base_id]), np.mean(points_y[:, base_id])]


def find_inlets_outlets2(line_map: SMS_MAP, boundary_gdf: gpd.GeoDataFrame, reverse_arc=False):
    '''
    Find the index of the last inlet point (index is outside, index-1 is inside)
    of all rivers. The projection of the river and the polygon must be consistent.

    Assuming the river points are arranged from downstream to upstream.
    Otherwise, set reverse_arc=True.
    '''

    timer = time()

    # ------------------------- find in-grid river points ---------------------------
    point_array = line_map.xyz[:, :2]
    points_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(point_array[:, 0], point_array[:, 1]), crs='EPSG:4326')
    points_within_polygon = gpd.sjoin(points_gdf, boundary_gdf, predicate='within')
    inside = points_gdf.index.isin(points_within_polygon.index)
    print(f'finding inside points took {time()-timer} seconds')

    timer = time()

    inlets = -np.ones(len(line_map.l2g), dtype=int)  # same length as the number of rivers
    outlets = -np.ones(len(line_map.l2g), dtype=int)
    for i, ids in enumerate(line_map.l2g):  # for each river
        if reverse_arc:
            ids = ids[::-1]  # reverse the order of the river points
        arc_points_inside = inside[ids]
        if np.any(arc_points_inside) and not np.all(arc_points_inside):  # crossing mesh boundary
            # last inlet, assuming the river points are arranged from down to upstream
            inlet = np.argwhere(arc_points_inside == 0)[0][0]
            if inlet > 0:
                inlets[i] = inlet
            # last outlet, assuming the river points are arranged from down to upstream
            outlet = np.argwhere(arc_points_inside)[0][0]
            if outlet > 0:
                outlets[i] = outlet

    print(f'{len(line_map.l2g)} rivers, {sum(inlets>0)} inlets, {sum(outlets>0)} outlets')
    return inlets, outlets


def make_feeder_channel():
    '''
    This function generates feeder channels for the river map to be used in SMS.

    It depends on one of the diagnostic maps generated by RiverMapper/make_river_map.py,
    which contains encoded information in the z-field of total_river_arcs_extra.map.
    The number of inner arcs is recorded as the first 2 decimals of the z-field of the
    toal_river_arcs_extra.

    The grid_boundary_shp_fname is the shapefile of the grid boundary,
    which should be generated manually from the "lbnd + other" coverage
    to form the boundary of the grid.
    For SECOFS, this is lbnd_coastal;
    For STOFS, this is lbnd_ocean.
    '''

    # ------------------- inputs -------------------
    # output_dir = '/sciclone/schism10/feiye/STOFS3D-v5/Inputs/v14/Parallel/SMS_proj/feeder/'
    # rivermap_fname = f'{output_dir}/total_inner_arcs.map'
    # grid_fname = '/sciclone/schism10/feiye/STOFS3D-v5/Inputs/v14/Parallel/SMS_proj/feeder/hgrid.ll'

    # output_dir = '/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.2/Feeder/'
    # rivermap_fname = f'/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.2/Feeder/total_river_arcs1.map'
    # grid_fname = '/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.2/Feeder/hgrid.ll'
    # grid_boundary_shp_fname = '/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16/Shapefiles/boundary.shp'

    # output_dir = '/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v18_subset/Feeder/'
    # rivermap_fname = f'/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.2/Feeder/total_river_arcs1.map'
    # grid_fname = '/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v17_subset/Feeder/hgrid.ll'
    # grid_boundary_shp_fname = '/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v17_subset/Shapefiles/boundary.shp'

    # output_dir = '/sciclone/schism10/Hgrid_projects/STOFS3D-v7/v20.0/Feeder/'
    # rivermap_fname = f'/sciclone/schism10/Hgrid_projects/STOFS3D-v7/v20.0/Feeder/total_river_arcs_extra.map'
    # grid_boundary_shp_fname = '/sciclone/schism10/Hgrid_projects/STOFS3D-v7/v20.0/Feeder/grid_boundary.shp'  # in esri:102008

    # output_dir = '/sciclone/schism10/Hgrid_projects/SECOFS/new22_JZ/Feeder'
    # rivermap_fname = '/sciclone/schism10/Hgrid_projects/SECOFS/new22_JZ/Feeder/total_river_arcs_extra.map'
    # grid_boundary_shp_fname = '/sciclone/schism10/Hgrid_projects/SECOFS/new22_JZ/Feeder/grid_boundary.shp'  # in esri:102008
    # reverse_arc = False

    # output_dir = '/sciclone/schism10/Hgrid_projects/STOFS3D-v8/v23.1/Feeder/'
    # rivermap_fname = f'{output_dir}/total_river_arcs_extra.map'
    # grid_boundary_shp_fname = f'{output_dir}/grid_bnd.shp'  # no need to dissolve, must be in esri:102008
    # reverse_arc = True  # used for NHD based river arcs

    output_dir = '/sciclone/schism10/Hgrid_projects/STOFS3D-v8/v24.4/Feeder/'
    rivermap_fname = f'{output_dir}/total_river_arcs_extra.map'
    grid_boundary_shp_fname = f'{output_dir}/grid_bnd.shp'  # no need to dissolve, must be in esri:102008
    reverse_arc = True  # used for NHD based river arcs

    # -------------------- end inputs ----------------

    gdf = gpd.read_file(grid_boundary_shp_fname)
    mesh_bnd_gdf = gdf.dissolve().to_crs("EPSG:4326")

    timer = time()

    # read river map, try loading cache first
    cache_fname = f'{rivermap_fname}.pkl'
    if Path(cache_fname).exists():
        with open(cache_fname, 'rb') as file:
            river_map = pickle.load(file)
    else:
        river_map = SMS_MAP(rivermap_fname)
        with open(cache_fname, 'wb') as file:
            pickle.dump(river_map, file)

    # calculate centerlines for each river
    n_rivers = 0  # number of rivers
    narcs_rivers = -np.ones((0, 1), dtype=int)  # number of river arcs for each river
    n = 0
    centerlines = []
    while (n < len(river_map.arcs)):  # loop each river, which has narcs along-river arcs
        # Number of inner arcs is recorded as the first 2 decimals of
        # the z-field of the toal_river_arcs_extra.
        arc_info = z_decoder(river_map.arcs[n].points[0, -1])
        # number of inner arcs is the same for all points along an arc, read the first one
        narcs = arc_info[0][0]

        # calculate centerline point coordinates by averaging river arcs
        centerline_points = river_map.arcs[n].points * 0.0
        for m in range(n, n+narcs):
            centerline_points += river_map.arcs[m].points
        centerline_points /= narcs

        # diagnostic z
        # centerline_points[:, -1] = np.arange(1, len(centerline_points)+1)

        centerlines.append(SMS_ARC(points=centerline_points, src_prj='epsg:4326'))

        n += narcs
        narcs_rivers = np.append(narcs_rivers, narcs)
        n_rivers += 1
    centerline_map = SMS_MAP(arcs=np.array(centerlines).reshape((-1, 1)))
    centerline_map.writer(filename=f'{output_dir}/centerline.map')
    centerline_map.get_xyz()

    print(f'reading river map took {time()-timer} seconds')
    timer = time()

    # find inlets and outlets
    # inlets[i] is the index of the first point after an arc goes from inside to outside of the grid
    inlets, outlets = find_inlets_outlets2(
        line_map=centerline_map, boundary_gdf=mesh_bnd_gdf, reverse_arc=reverse_arc)
    print(f'finding inlets/outlets took {time()-timer} seconds')
    timer = time()

    # Configure a fake channel from the head of the real part of the feeder
    # extending further upstream by a few rows of channel elements.
    # This is to ensure there are a few cross sections in a straight channel to allow flow ramp-up.
    feeder_channel_extension = np.array([-10.0, -5.0, 0])

    # i_inlet_option is the index of the inlet point to be used for the feeder channel
    # the number means how many points to go upstream from the last inlet point
    i_inlet_options = [2, 1, 0]

    # get total number of river-arcs that cross the boundary
    n_inlets = sum(inlets > -1)
    narcs_rivers_inlets = sum(narcs_rivers[inlets > -1])

    max_n_follow = 10  # follow the real river channel into the model domain for a few cross sections

    # the number of feeder arcs = along-river arcs + cross-river arcs
    feeder_arcs = np.empty((
        narcs_rivers_inlets + n_inlets * (max_n_follow + len(feeder_channel_extension)), 1
    ), dtype=object)

    # the number of outlet arcs (only considering the base cross-river arc, since no pseudo channel is needed)
    # is set as the same of the number of inlets for convenience, but most inlets don't have an outlet
    outlet_arcs = np.empty((n_inlets, 1), dtype=object)  # left bank and right bank for each thalweg

    i = 0
    i_river = 0
    n_feeder = 0
    n_outlet = 0
    feeders = []
    while (i < len(river_map.arcs)):
        this_inlets = inlets[i_river]  # inlets[range(i, i+narcs_rivers[i_river])]
        if this_inlets > 0:  # any(this_inlets>0)
            for i_inlet_option in i_inlet_options:
                # follow river arcs inside the domain as an extension of the feeder channel,
                # in order to avoid isolated feeder channels
                feeder_base_pts = np.zeros((0, 3), dtype=float)
                feeder_follow_pts = np.zeros((0, 3), dtype=float)

                inlet = min(this_inlets + i_inlet_option, len(river_map.arcs[i].points) - 1)
                i_follow = np.arange(inlet-1, max(inlet-max_n_follow, -1), -1)  # [0, inlet-1], reversed
                n_follow = len(i_follow)

                for j in range(i, i+narcs_rivers[i_river]):  # along-channel arcs of a river
                    river_points = river_map.arcs[j].points
                    if reverse_arc:
                        river_points = river_points[::-1, :]
                    feeder_base_pts = np.r_[feeder_base_pts, river_points[inlet, :].reshape(1,3)]
                    feeder_follow_pts = np.r_[feeder_follow_pts, river_points[i_follow, :].reshape(-1,3)]

                perp = np.mean(get_perpendicular_angle(line=feeder_base_pts[[1, -1], :2]))
                if reverse_arc:  # the cross channel arc is also reversed
                    perp += np.pi

                width = ((feeder_base_pts[0, 0] - feeder_base_pts[1, 0])**2 + (feeder_base_pts[0, 1] - feeder_base_pts[1, 1])**2) ** 0.5
                feeder_channel_length = feeder_channel_extension * width

                # coordinates of feeder channel points,
                # the 1st dimension is the number of along-river arcs and
                # the 2nd dimension is the number of points along each feeder channel arc
                xt = np.zeros((narcs_rivers[i_river], len(feeder_channel_extension)+n_follow), dtype=float)
                yt = np.zeros((narcs_rivers[i_river], len(feeder_channel_extension)+n_follow), dtype=float)
                for k in range(len(feeder_channel_extension)):
                    xt[:, k] = feeder_base_pts[:, 0] + feeder_channel_length[k] * np.cos(perp)
                    yt[:, k] = feeder_base_pts[:, 1] + feeder_channel_length[k] * np.sin(perp)

                point_array = np.c_[xt[:, i_inlet_option:len(feeder_channel_extension)].reshape(-1, 1),  # used to be i_inlet_option+1, needs to be checked
                                    yt[:, i_inlet_option:len(feeder_channel_extension)].reshape(-1, 1)]
                if len(point_array) == 0:
                    break
                points_gdf = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(point_array[:, 0], point_array[:, 1]),
                    crs='EPSG:4326'
                )
                points_within_polygon = gpd.sjoin(points_gdf, mesh_bnd_gdf, predicate='within')

                ingrid_feeders = points_gdf.index.isin(points_within_polygon.index)

                if sum(ingrid_feeders) == 0:
                    break  # found clean feeder (all feeder channel points are outside the grid boundary)

                # the worse case is none of the i_inlet_option returns clean feeders,
                # in this case the feeders are still valid and the last one is kept.

                print(f'unclean connection at arc {i+1}')
                print(f'inlet location: {feeder_base_pts[0, 0]}, {feeder_base_pts[0, 1]}')

            if n_follow > 0:
                xt[:, -n_follow:] = feeder_follow_pts[:, 0].reshape(-1, n_follow)
                yt[:, -n_follow:] = feeder_follow_pts[:, 1].reshape(-1, n_follow)

            for k in range(xt.shape[0]):
                feeder_arcs[n_feeder] = SMS_ARC(points=np.c_[xt[k, :], yt[k, :]], src_prj='epsg:4326')
                n_feeder += 1

            for k in range(xt.shape[1]):
                feeder_arcs[n_feeder] = SMS_ARC(points=np.c_[xt[:, k], yt[:, k]], src_prj='epsg:4326')
                n_feeder += 1

            feeders.append(Feeder(points_x=xt, points_y=yt, base_id=len(feeder_channel_extension)+1))

        this_outlets = outlets[i_river]  # outlets[range(i, i+narcs_rivers[i_river])]
        if this_outlets > 0:  # any(this_outlets>0):
            outlet_base_pts = np.zeros((0, 3), dtype=float)
            outlet = this_outlets  # max(0, np.min(this_outlets[this_outlets!=-1]))
            for j in range(i, i+narcs_rivers[i_river]):  # inner arcs of a river
                river_points = river_map.arcs[j].points
                if reverse_arc:
                    river_points = river_points[::-1, :]
                outlet_base_pts = np.r_[outlet_base_pts, river_points[outlet, :].reshape(1,3)]
                outlet_arcs[n_outlet] = SMS_ARC(points=outlet_base_pts, src_prj='epsg:4326')
                n_outlet += 1

        i += narcs_rivers[i_river]
        i_river += 1

    if len(feeders) != n_inlets:
        raise Exception("Inconsistent number of inlets and feeder channels")

    # -----------------------   write outputs   -----------------------
    feeders_map = SMS_MAP(arcs=feeder_arcs.reshape((-1, 1)))
    feeders_map.writer(filename=f'{output_dir}/feeders.map')

    feeders_shp = feeders_map.to_GeoDataFrame()
    feeders_shp.set_crs("epsg:4326", inplace=True)
    feeders_shp.to_file(f'{output_dir}/feeders.shp')

    # clip feeder channels outside the grid boundary
    feeders_outside = gpd.overlay(feeders_shp, mesh_bnd_gdf, how='difference')
    feeders_outside.to_file(f'{output_dir}/feeders_outside.shp')


    SMS_MAP(arcs=outlet_arcs.reshape((-1, 1))).writer(filename=f'{output_dir}/outlets.map')

    # save feeders info in a *.pkl
    feeder_heads = np.zeros((len(feeders), 3), dtype=float)
    feeder_bases = np.zeros((len(feeders), 3), dtype=float)
    feeder_l2g = [None] * len(feeders); npts = 0
    feeder_points = np.empty((0, 2), dtype=float)
    feeder_arrays_x = [None] * len(feeders)
    feeder_arrays_y = [None] * len(feeders)
    for i, feeder in enumerate(feeders):
        feeder_heads[i, :2] = feeder.head[:]
        feeder_bases[i, :2] = feeder.base[:]
        feeder_l2g[i] = np.array(np.arange(npts, npts+feeder.points_x.size))
        feeder_points = np.r_[feeder_points, np.c_[feeder.points_x.reshape(-1,1), feeder.points_y.reshape(-1,1)]]
        feeder_arrays_x[i] = feeder.points_x
        feeder_arrays_y[i] = feeder.points_y
        npts += feeder.points_x.size

    # write outputs
    with open(f'{output_dir}/feeder.pkl', 'wb') as file:
        pickle.dump([feeder_l2g, feeder_points, feeder_heads, feeder_bases], file)
    
    np.savetxt(f'{output_dir}/feeder_heads_bases.xy', np.c_[feeder_heads[:, :2], feeder_bases[:, :2]])

    with open(f'{output_dir}/feeder_arrays.pkl', 'wb') as file:
        pickle.dump([feeder_arrays_x, feeder_arrays_y], file)

    SMS_MAP(detached_nodes=np.r_[feeder_heads, feeder_bases]).writer(filename=f'{output_dir}/feeder_hb.map')


if __name__ == '__main__':
    make_feeder_channel()
