"""
This script provides a class to configure the parameters of make_river_map.

Parameter presets are also provided  as class methods for different usage scenarios.
"""

# global variables
cpp_crs = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

class ConfigRiverMap():
    '''A class to handle the configuration of the river map generation,
    i.e., processing the parameters for the function make_river_map()
    and storing the factory settings for the river map generation.
    Note that only optional parameters are handled here,
    users should still provide the required inputs (paths to necessary files)
    for the function make_river_map().

    Use unpacking to pass the optional parameters to the function make_river_map(),
    Example usage:
    my_config = ConfigRiverMap.OuterArcs()
    make_river_map(
        tif_fnames = ['./Inputs/DEMs/GA_dem_merged_ll.tif'],
        thalweg_shp_fname = './Inputs/Shapefiles/GA_local.shp',
        output_dir = './Outputs/',
         **my_config.optional,
    )
    '''
    def __init__(self,
        i_DEM_cache = True, selected_thalweg = None,
        MapUnit2METER = 1, river_threshold = (5, 400), min_arcs = 3,
        elev_scale = 1.0,
        outer_arcs_positions = (), R_coef=0.4, length_width_ratio = 6.0,
        along_channel_reso_thres = (5, 300),
        i_blast_intersection = False, blast_radius_scale = 0.5, bomb_radius_coef = 0.3,
        i_real_clean = False, projection_for_cleaning = cpp_crs,
        i_close_poly = True, i_smooth_banks = True,
        snap_point_reso_ratio = 0.3, snap_arc_reso_ratio = 0.2,
        output_prefix = '', mpi_print_prefix = '', i_OCSMesh = False, i_DiagnosticOutput = False,
        i_pseudo_channel = 0, pseudo_channel_width = 18, nrow_pseudo_channel = 4,
    ):
        # see a description of the parameters in the function make_river_map()
        self.optional = {
            'selected_thalweg': selected_thalweg,
            'output_prefix': output_prefix,
            'mpi_print_prefix': mpi_print_prefix,
            'MapUnit2METER': MapUnit2METER,
            'river_threshold': river_threshold,
            'min_arcs': min_arcs,
            'elev_scale': elev_scale,
            'outer_arcs_positions': outer_arcs_positions,
            'R_coef': R_coef,
            'length_width_ratio': length_width_ratio,
            'along_channel_reso_thres': along_channel_reso_thres,
            'i_close_poly': i_close_poly,
            'i_blast_intersection': i_blast_intersection,
            'i_real_clean': i_real_clean,
            'projection_for_cleaning': projection_for_cleaning,
            'blast_radius_scale': blast_radius_scale,
            'bomb_radius_coef': bomb_radius_coef,
            'snap_point_reso_ratio': snap_point_reso_ratio,
            'snap_arc_reso_ratio': snap_arc_reso_ratio,
            'i_smooth_banks': i_smooth_banks,
            'i_DEM_cache': i_DEM_cache,
            'i_OCSMesh': i_OCSMesh,
            'i_DiagnosticOutput': i_DiagnosticOutput,
            'i_pseudo_channel': i_pseudo_channel,
            'pseudo_channel_width': pseudo_channel_width,
            'nrow_pseudo_channel': nrow_pseudo_channel,
        }

    @classmethod
    def LooselyFollowRivers(cls):
        '''Small-scale river curvatures may not be exactly followed,
        but channel connectivity is still preserved.'''
        return cls(length_width_ratio = 30.0)

    @classmethod
    def OuterArcs(cls):
        '''
        Outer arcs are generated at the specified positions parallel to the bank arcs
        and on both sides of the river.
        The values of outer_arcs_positions are the relative posititions of the river width,
        '''
        return cls(outer_arcs_positions = (0.1, 0.2), i_real_clean = False)

    @classmethod
    def BombedIntersections(cls):
        '''
        Blast intersection arcs and insert feature points to pave river confluence.
        Note that an additional output file "intersection_joints.map" will be generated;
        this file contains the coordinates of the intersection points.
        '''
        return cls(i_blast_intersection = True, blast_radius_scale = 0.5, bomb_radius_coef = 0.3)

    @classmethod
    def FurtherCleanIntersections(cls):
        '''
        Compared to the default parameter settings, which only snap close-by points,
        this option (i_real_clean = True) further cleans nearby arcs closer than
        the threshold (snap_arc_reso_ratio * river width).
        However, the efficiency still needs to be improved.
        '''
        return cls(blast_radius_scale = 0.0, i_real_clean = True)

    @classmethod
    def BarrierIsland(cls):
        '''
        Barrier islands can be treated as a special case of channels
        if the z values of the DEM are inverted.
        Explanation on the changes from the default settings:
        1. river_threshold: the default value is (5, 400), which is too small for barrier islands.
        2. elev_scale: the default value is 1.0; a value of -1.0 is used here to invert the DEM.
        3. i_real_clean: the default value is False; a value of True is used here to further clean the river map.
        4. R_coef: the default value is 0.4; a larger value is needed here to make the barrier map more smooth,
              because capturing the curvature of barrier islands is not as important as the capturing barrier height.
        5. length_width_ratio: the default value is 6.0; a larger value is used here to make the barrier map more smooth.
        6. along_channel_reso_thres: the default value is (5, 300); a larger value is used here to fit a nicer
              transition between the barrier and the ocean (which often have a resolution of 2000 m).
        '''
        return cls(river_threshold=(5, 1000), elev_scale=-1.0, i_real_clean=True,
                   R_coef=20, length_width_ratio=100, along_channel_reso_thres=(5, 1000))

    @classmethod
    def Levees(cls):
        '''
        Expand all thalwegs (in this case levee centerlines)
        to channels (in this case levees with two feet and a flat top)
        of a fixed width (in this case foot-to-foot width)
        '''
        return cls(i_pseudo_channel = 1, pseudo_channel_width = 18,
                   nrow_pseudo_channel = 4, length_width_ratio = 50.0,
                   i_DiagnosticOutput = True, i_real_clean = True,
                   snap_point_reso_ratio = -1e-5,  # negative value means abosolute value (lon/lat)
                   i_smooth_banks = False, river_threshold = (18, 18), min_arcs = 4)

    @classmethod
    def PseudoChannels(cls):
        '''
        Expand all thalwegs (in this case levee centerlines)
        to channels (in this case levees with two feet and a flat top)
        of a fixed width (in this case foot-to-foot width)
        '''
        return cls(i_pseudo_channel = 2, pseudo_channel_width = 30, nrow_pseudo_channel = 4)
