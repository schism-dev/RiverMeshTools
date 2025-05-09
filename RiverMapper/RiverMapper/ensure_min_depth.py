'''
This script helps ensure a minimum depth in rivers.
The depth is measured from the higher bank elevation to any point in the river.
If the depth is less than the specified minimum depth, the point is dredged to
satisfy the minimum depth requirement.
'''

import numpy as np
from copy import deepcopy
from sklearn.neighbors import KDTree
import geopandas as gpd

from RiverMapper.SMS import SMS_MAP
from RiverMapper.util import z_decoder
from pylib import read as read_schism_hgrid  # pip install pylibs-ocean
from pylib_experimental.schism_file import cread_schism_hgrid as read_schism_hgrid

class Rivers():
    '''
    Class to handle river information from RiverMapper's output: total_river_arcs_extra.map,
    which contains extra information in the z-field.
    It groups river arcs into rivers and provides methods to manipulate river information.
    '''
    def __init__(self, river_arc_info: SMS_MAP, crs='epsg:4326') -> None:
        self.river_arc_info = river_arc_info  # original river arc information in an SMS_MAP object

        self.river_arc_gdf = river_arc_info.to_GeoDataFrame()  # river arcs in a GeoDataFrame
        self.river_arc_gdf.set_crs(crs, inplace=True)

        # list of river arcs grouped by rivers
        self.rivers_arcs, self.riverarcs_grouping = self.group_river_arcs()

        # index of rivers that are in the region of interest
        self.idx = np.ones(len(river_arc_info.arcs), dtype=bool)  # all rivers initially

        self.arcs_z = None  # z values of river arcs' vertices
        self.rivers_coor = None

        self.rivers_transects = None
        self.rivers_centerline_coor = None

        self.riverhead_transects = None
        self.riverhead_transects_center = None

    def group_river_arcs(self, nrow_info_position: int = 0) -> list:
        '''
        Arrange river information into a list of grouped river arcs.
        Each group corresponds to a river segment.

        river_arc_info: SMS_MAP object with river arc information coded in the z-field
        nrow_info_position: position of the z-field to decode the number of cross-channel nodes
            e.g., z=2.0401 and nrow_row_info_position=0:
            2 means there are 2 pieces of information in the z-field,
            "04" means 4 rows of arcs (including bank arcs) parrallel to the banks,
            "01" is another piece of information, in this case an indicator of outer arc
        '''
        n_arc = 0
        group = []
        group_idx = []
        while n_arc < len(self.river_arc_info.arcs):
            # nrow (number of arcs) in the cross-channel direction,
            # coded as the first two decimal digits of the z-field.
            # nrow is the same for all points along an arc, so just use the first one
            nrow = z_decoder(self.river_arc_info.arcs[n_arc].points[:, -1])[0][nrow_info_position]

            group.append(self.river_arc_info.arcs[n_arc:n_arc+nrow])
            group_idx.append(np.arange(n_arc, n_arc+nrow))

            n_arc += nrow
        return group, group_idx

    def get_river_arcs_z(self) -> list:
        '''get z values of river arcs vertices from the mesh'''
        raise NotImplementedError('Not implemented yet, use mesh_dp2riverarc_z() instead')

    def get_rivers_coor(self) -> list:
        '''get river vertices coordinates (x,y,z) grouped by rivers'''
        if self.arcs_z is None:
            self.get_river_arcs_z()

        rivers_coor = [None] * len(self.riverarcs_grouping)
        for k, arcs_ids in enumerate(self.riverarcs_grouping):
            if any(self.idx[arcs_ids]):
                rivers_coor[k] = np.stack([
                    np.c_[self.river_arc_info.arcs[i].points[:, :2], self.arcs_z[i]]
                    for i in arcs_ids
                ])
        self.rivers_coor = rivers_coor
        return rivers_coor

    def get_rivers_transects(self) -> list:
        '''get the cross-sections of rivers grouped by rivers'''

        if self.arcs_z is None:
            self.get_river_arcs_z()

        self.rivers_transects = [None] * len(self.riverarcs_grouping)
        for k, arcs_ids in enumerate(self.riverarcs_grouping):
            if any(self.idx[arcs_ids]):
                river_points = np.stack([
                    np.c_[self.river_arc_info.arcs[i].points[:, :2], self.arcs_z[i]]
                    for i in arcs_ids
                ])
                self.rivers_transects[k] = np.array([
                    river_points[:, i, :] for i in range(river_points.shape[1])
                ])  # transects along the river; i is along-river index
        return self.rivers_transects

    def get_rivers_centerline(self) -> list:
        ''' get river center coordinates grouped by rivers '''
        self.rivers_centerline_coor = []
        for _, river_arcs in enumerate(self.rivers_arcs):
            self.rivers_centerline_coor.append(np.mean([arc.points for arc in river_arcs], axis=0))

        return self.rivers_centerline_coor

    def mesh_dp2riverarc_z(self, hgrid_obj) -> list:
        '''
        get z from the nearest mesh nodes;
        most river arc points are mesh nodes except for those subject to cleaning

        :return: arcs_z: list of z values of river arcs' vertices
        '''
        xyz, l2g = self.river_arc_info.get_xyz()
        _, map2mesh = KDTree(np.c_[hgrid_obj.x, hgrid_obj.y]).query(xyz[:, :2])
        arcs_z = []
        for arc_idx in l2g:
            arcs_z.append(hgrid_obj.z[map2mesh[arc_idx]])
        self.arcs_z = arcs_z
        return arcs_z

    def set_region_of_interest(self, region_gdf: gpd.GeoDataFrame) -> np.ndarray:
        '''
        Find the index of rivers that are in the region of interest.
        Indices of the points inside region are saved in self.idx
        '''
        in_region = gpd.sjoin(
            self.river_arc_gdf.to_crs(region_gdf.crs), region_gdf,
            how='inner', predicate='intersects'
        ).index
        idx = np.zeros(len(self.river_arc_info.arcs), dtype=bool)
        idx[in_region] = True
        self.idx = idx

    def dredge_inner_arcs(
        self, min_channel_depth=1, region_gdf=None, diag_output_dir=None
    ) -> np.ndarray:
        '''
        Dredge the inner longitudinal transects based on the difference between
        the highest bank elevation and a user-defined depth
        Use the original river_arc_*.map as inputs to retain the original river arc order
        Use the valid_idx to filter out the rivers that are not in the region of interest

        :param min_channel_depth: float, depth to dredge the inner transects,
             measured from the higher bank
        :param region_gdf: gpd.GeoDataFrame, region of interest
        :param diag_output_dir: str, directory to save diagnostic files

        :return: dredged_points: np.ndarray, shape=(n_points, 3),
            x, y, z coordinates of dredged points
        '''
        if self.rivers_coor is None:
            self.get_rivers_coor()
        if region_gdf is not None:
            self.set_region_of_interest(region_gdf=region_gdf)

        dredged_points = np.zeros((0, 3), dtype=float)
        for k, arcs_id in enumerate(self.riverarcs_grouping):
            if any(self.idx[arcs_id]):
                # inside the watershed, i.e., where river arc points coorespond to mesh nodes

                # Measure target dp from the higher bank's dp.
                # rivers_coor: Left bank and right bank; along-river index; z of xyz
                bank_dp = np.min(self.rivers_coor[k][[0, -1], :, 2], axis=0)
                target_thalweg_dp = bank_dp + min_channel_depth  # target thalweg dp, positive downward

                # dredge the inner transects; todo: consider the outer arcs
                self.rivers_coor[k][1:-1, :, 2] = np.maximum(self.rivers_coor[k][1:-1, :, 2], target_thalweg_dp)
                dredged_points = np.r_[dredged_points, self.rivers_coor[k][1:-1, :, :].reshape(-1, 3)]

        dredged_points_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(dredged_points[:, 0], dredged_points[:, 1], dredged_points[:, 2]),
            crs='EPSG:4326')
        # add z as a column
        dredged_points_gdf['z'] = dredged_points[:, 2]

        if region_gdf is not None:  # clip the dredged points to the watershed
            idx = gpd.sjoin(
                dredged_points_gdf.to_crs(region_gdf.crs), region_gdf, how='inner', predicate='intersects'
            ).index
            dredged_points = dredged_points[idx, :]
            dredged_points_gdf = dredged_points_gdf.iloc[idx]

        if diag_output_dir is not None:
            dredged_points_gdf.to_file(diag_output_dir + './dredged_points.shp')

        return dredged_points

    def match_transect(self, poi_xy: np.ndarray, diag_output_dir=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        find the nearest resolved river transect of each point of interest
        :param nwm_gdf: gpd.GeoDataFrame, NWM hydrofabric
        :param poi_xy: np.ndarray, shape=(n_points, 2), x, y coordinates of points of interest
        :param diag_output_dir: str, directory to save diagnostic files, None if not saving

        :return:
        riverhead_transects: list of np.ndarray, shape=(n_transect_points, 3),
           x, y, z coordinates of river head transects
        riverhead_transects_center: list of np.ndarray, shape=(3, ),
           x, y, z coordinates of river head transects center
        '''

        rivercenter_coor = np.vstack(self.rivers_centerline_coor)
        # initialize the global to local mapping to -1
        rivercenter_vertices_g2l = np.zeros((rivercenter_coor.shape[0], 2), dtype=int) - 1
        n = 0
        for i in range(len(self.rivers_arcs)):  # group index
            # vertices index within a group (in this case only one arc, i.e., the centerline arc)
            for j in range(len(self.rivers_centerline_coor[i])):
                rivercenter_vertices_g2l[n] = [i, j]
                n += 1
        if n != len(rivercenter_coor):
            raise ValueError('Inconsistent number of river center vertices')

        _, idx = KDTree(rivercenter_coor[:, :2]).query(poi_xy)

        riverhead_transects = []
        riverhead_transects_center = []
        riverheads_in_arcgroup = rivercenter_vertices_g2l[np.squeeze(idx)]  # 2d array: [river index, vertices index]
        for i, indicies in enumerate(riverheads_in_arcgroup):  # riverhead: [river index, vertices index]
            river_idx, vertex_idx = indicies
            transect_coor = np.array([arc.points[vertex_idx, :] for arc in self.rivers_arcs[river_idx]])
            transect_z = np.array([self.arcs_z[arc_idx][vertex_idx] for arc_idx in self.riverarcs_grouping[river_idx]])
            riverhead_transects.append(np.c_[transect_coor[:, :2], transect_z])
            riverhead_transects_center.append(np.mean(transect_coor, axis=0))

        if diag_output_dir is not None:
            gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(
                    x=np.vstack(riverhead_transects)[:, 0], y=np.vstack(riverhead_transects)[:, 1], crs='EPSG:4326')
            ).to_file(f'{diag_output_dir}/river_head_transect.shp')

        self.riverhead_transects = riverhead_transects
        self.riverhead_transects_center = riverhead_transects_center

        return riverhead_transects, riverhead_transects_center

    def match_river_center(self, poi_xy: np.ndarray) -> np.ndarray:
        '''find the nearest river center for each point of interest'''
        if self.rivers_centerline_coor is None:
            self.get_rivers_centerline()

        rivercenter_coor = np.vstack(self.rivers_centerline_coor)
        _, idx = KDTree(rivercenter_coor[:, :2]).query(poi_xy)

        return rivercenter_coor[np.squeeze(idx)]

# -------------------------------------------------------------------------------------
# ------------------------------end class definition-----------------------------------
# -------------------------------------------------------------------------------------


def dredge_river_transects(
    rivers: Rivers,
    region_gdf: gpd.GeoDataFrame = None,
    hgrid_obj=None,  # schism_hgrid object read by the read() function in pylib
    min_channel_depth=1.0,
    output_dir='./'
):
    '''
    Dredge the inner arcs of each river transect to maintain a minimum elevation drop
    from bank to thalweg, thus maintaining channel connectivity

    Inputs:
    - rivers: Rivers object from reading the RiverMapper diagnostic output file
    - region_gdf: gpd.GeoDataFrame, region of interest, in which the dredging is performed.
        It must be in the same coordinate system as the mesh.
    - hgrid_obj: schism_hgrid object, mesh object
    - min_channel_depth: float, minimum channel depth to dredge.
        The depth is measured from the higher bank elevation to an inner arc node.
    - output_dir: str, directory to save the dredged mesh and diagnostic files
    '''

    # get the river arcs' z values from the mesh
    rivers.mesh_dp2riverarc_z(hgrid_obj)

    # find the inner arcs and dredge within the watershed
    dredged_points = rivers.dredge_inner_arcs(
        region_gdf=region_gdf, min_channel_depth=min_channel_depth)

    # map dredged points to mesh nodes
    _, idx = KDTree(np.c_[hgrid_obj.x, hgrid_obj.y]).query(dredged_points[:, :2])
    # update the mesh
    hgrid_dredged = deepcopy(hgrid_obj)
    hgrid_dredged.dp[np.squeeze(idx)] = np.maximum(
        hgrid_dredged.dp[np.squeeze(idx)], dredged_points[:, 2]
    )

    # output
    hgrid_dredged.grd2sms(output_dir + '/hgrid_dredged.2dm')  # SMS format
    hgrid_dredged.save(output_dir + '/hgrid_dredged.gr3', fmt=1)  # SCHISM format


def sample_usage_dredge_river_transects():
    '''
    Sample usage of dredge_river_transects()
    '''
    # Load extra information from the river arcs
    # You should have this file under the RiverMapper output directory;
    # if not, configure RiverMapper to output this file by setting i_DiagnosticOutput
    rivers = Rivers(SMS_MAP(
        '/sciclone/schism10/Hgrid_projects/STOFS3D-v7/v19_RiverMapper/Outputs/'
        'bora_v19.1.v19_ie_v18_3_nwm_clipped_in_cudem_missing_tiles_20-core/'
        'total_river_arcs_extra.map'
    ))
    # Define region of interest
    region_gdf = gpd.read_file(
        '/sciclone/schism10/feiye/STOFS3D-v8/I15_v7/Bathy_edit/RiverArc_Dredge/watershed_ME.shp'
    )
    # schism mesh object with bathymetry
    hgrid_obj = read_schism_hgrid(
        '/sciclone/schism10/feiye/STOFS3D-v8/I15a_v7/Bathy_edit/RiverArc_Dredge/hgrid.ll')

    output_dir = '/sciclone/schism10/feiye/STOFS3D-v8/I15a_v7/Bathy_edit/RiverArc_Dredge/'

    # Dredge the river transects
    dredge_river_transects(
        rivers, region_gdf, hgrid_obj,
        min_channel_depth=1.0, output_dir=output_dir
    )


if __name__ == '__main__':
    sample_usage_dredge_river_transects()
