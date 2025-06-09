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
from RiverMapper.util import Rivers
from pylib import read as read_schism_hgrid  # pip install pylibs-ocean
# from pylib_experimental.schism_file import cread_schism_hgrid as read_schism_hgrid


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
        It must have a coordinate reference system (CRS) defined.
    - hgrid_obj: schism_hgrid object with bathymetry loaded, assuming lon/lat
    - min_channel_depth: float, minimum channel depth to dredge.
        The depth is measured from the higher bank elevation to an inner arc node.
    - output_dir: str, directory to save the dredged mesh and diagnostic files
    '''

    print('getting river arcs z from the mesh ...')
    rivers.mesh_dp2riverarc_z(hgrid_obj)

    print('dredging river transects ...')
    dredged_points = rivers.dredge_inner_arcs(
        region_gdf=region_gdf, min_channel_depth=min_channel_depth)

    print('mapping dredged points to the mesh ...')
    _, idx = KDTree(np.c_[hgrid_obj.x, hgrid_obj.y]).query(dredged_points[:, :2])
    # update the mesh
    hgrid_dredged = deepcopy(hgrid_obj)
    hgrid_dredged.dp[np.squeeze(idx)] = np.maximum(
        hgrid_dredged.dp[np.squeeze(idx)], dredged_points[:, 2]
    )

    print('saving dredged mesh ...')
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
    ))  # default crs is 'epsg:4326', which is also the default for RiverMapper
    # # Define region of interest
    # region_gdf = gpd.read_file(
    #     '/sciclone/schism10/Hgrid_projects/STOFS3D-v8/v31/Clip/outputs/watershed.shp'
    # ).to_crs('epsg:4326')
    # # schism mesh object with bathymetry
    # hgrid_obj = read_schism_hgrid(
    #     '/sciclone/schism10/feiye/STOFS3D-v8/I15a_v7/Bathy_edit/RiverArc_Dredge/hgrid.ll')

    # output_dir = '/sciclone/schism10/feiye/STOFS3D-v8/I15a_v7/Bathy_edit/RiverArc_Dredge_test/'
    watershed_origional = gpd.read_file(
        '/sciclone/schism10/Hgrid_projects/STOFS3D-v8/v31/Clip/outputs/watershed.shp'
    )
    watershed = gpd.overlay(
        watershed_origional,
        gpd.read_file(
            '/sciclone/schism10/feiye/STOFS3D-v8/I15_v7/Bathy_edit/RiverArc_Dredge/watershed_ME.shp'
        ).to_crs(watershed_origional.crs),
        how='difference'
    )

    hgrid_obj = read_schism_hgrid(
        '/sciclone/schism10/feiye/STOFS3D-v8/I15a_v7/Bathy_edit/RiverArc_Dredge/hgrid.ll')

    output_dir = '/sciclone/schism10/feiye/STOFS3D-v8/I15a_v7/Bathy_edit/RiverArc_Dredge/'

    # Dredge the river transects
    dredge_river_transects(
        rivers, region_gdf=watershed, hgrid_obj=hgrid_obj,
        min_channel_depth=1.0, output_dir=output_dir
    )


if __name__ == '__main__':
    sample_usage_dredge_river_transects()
