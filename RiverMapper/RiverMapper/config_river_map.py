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

    # class constants defining the default values of the optional parameters,
    # also used by make_river_map()
    DEFAULT_i_DEM_cache = True
    DEFAULT_selected_thalweg = None
    DEFAULT_river_threshold = (5, 600)  # used to be (16, 400) around the time of STOFS-3D-Atl v6 (schism git f10c3eab)
    DEFAULT_min_arcs = 3
    DEFAULT_width2narcs_option = 'regular'
    DEFAULT_custom_width2narcs = None
    DEFAULT_elev_scale = 1.0
    DEFAULT_outer_arcs_positions = ()
    DEFAULT_R_coef=0.4
    DEFAULT_length_width_ratio = 6.0
    DEFAULT_along_channel_reso_thres = (5, 300)
    DEFAULT_snap_point_reso_ratio = 0.3
    DEFAULT_snap_arc_reso_ratio = 0.2
    DEFAULT_n_clean_iter = 5
    DEFAULT_i_close_poly = True
    DEFAULT_i_smooth_banks = True
    DEFAULT_output_prefix = ''
    DEFAULT_mpi_print_prefix = ''
    DEFAULT_i_OCSMesh = True
    DEFAULT_i_DiagnosticOutput = False
    DEFAULT_i_pseudo_channel = 2
    DEFAULT_pseudo_channel_width = 18
    DEFAULT_nrow_pseudo_channel = 4

    def __init__(self,
        i_DEM_cache = DEFAULT_i_DEM_cache,
        selected_thalweg = DEFAULT_selected_thalweg,
        river_threshold = DEFAULT_river_threshold,
        min_arcs = DEFAULT_min_arcs,
        width2narcs_option = DEFAULT_width2narcs_option,
        custom_width2narcs = DEFAULT_custom_width2narcs,
        elev_scale = DEFAULT_elev_scale,
        outer_arcs_positions = DEFAULT_outer_arcs_positions,
        R_coef= DEFAULT_R_coef,
        length_width_ratio = DEFAULT_length_width_ratio,
        along_channel_reso_thres = DEFAULT_along_channel_reso_thres,
        snap_point_reso_ratio = DEFAULT_snap_point_reso_ratio,
        snap_arc_reso_ratio = DEFAULT_snap_arc_reso_ratio,
        n_clean_iter = DEFAULT_n_clean_iter,
        i_close_poly = DEFAULT_i_close_poly,
        i_smooth_banks = DEFAULT_i_smooth_banks,
        output_prefix = DEFAULT_output_prefix,
        mpi_print_prefix = DEFAULT_mpi_print_prefix,
        i_OCSMesh = DEFAULT_i_OCSMesh,
        i_DiagnosticOutput = DEFAULT_i_DiagnosticOutput,
        i_pseudo_channel = DEFAULT_i_pseudo_channel,
        pseudo_channel_width = DEFAULT_pseudo_channel_width,
        nrow_pseudo_channel = DEFAULT_nrow_pseudo_channel,
    ):
        # see a description of the parameters in the function make_river_map()
        self.optional = {
            'i_DEM_cache': i_DEM_cache,
            'selected_thalweg': selected_thalweg,
            'river_threshold': river_threshold,
            'min_arcs': min_arcs,
            'elev_scale': elev_scale,
            'outer_arcs_positions': outer_arcs_positions,
            'R_coef': R_coef,
            'length_width_ratio': length_width_ratio,
            'along_channel_reso_thres': along_channel_reso_thres,
            'snap_point_reso_ratio': snap_point_reso_ratio,
            'snap_arc_reso_ratio': snap_arc_reso_ratio,
            'n_clean_iter': n_clean_iter,
            'i_close_poly': i_close_poly,
            'i_smooth_banks': i_smooth_banks,
            'output_prefix': output_prefix,
            'mpi_print_prefix': mpi_print_prefix,
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
        return cls(outer_arcs_positions = (0.1, 0.2))

    @classmethod
    def BarrierIsland(cls):
        '''
        Barrier islands can be treated as a special case of channels
        if the z values of the DEM are inverted.
        Explanation on the changes from the default settings:
        1. river_threshold: the default value is (5, 400), which is too small for barrier islands.
        2. elev_scale: the default value is 1.0; a value of -1.0 is used here to invert the DEM.
        3. R_coef: the default value is 0.4; a larger value is needed here to make the barrier map more smooth,
              because capturing the curvature of barrier islands is not as important as the capturing barrier height.
        4. length_width_ratio: the default value is 6.0; a larger value is used here to make the barrier map more smooth.
        5. along_channel_reso_thres: the default value is (5, 300); a larger value is used here to fit a nicer
              transition between the barrier and the ocean (which often have a resolution of 2000 m).
        '''
        return cls(
            river_threshold=(5, 1000), elev_scale=-1.0, R_coef=20,
            length_width_ratio=100, along_channel_reso_thres=(5, 1000)
        )

    @classmethod
    def Levees(cls):
        '''
        Expand all thalwegs (in this case levee centerlines)
        to channels (in this case levees with two feet and a flat top)
        of a fixed width (in this case foot-to-foot width)
        '''
        return cls(
            i_pseudo_channel = 1, pseudo_channel_width = 25,
            nrow_pseudo_channel = 4, length_width_ratio = 80.0,
            snap_point_reso_ratio = 0.1, snap_arc_reso_ratio = 0.1,
            i_smooth_banks = False, river_threshold = (18, 18), min_arcs = 4,
            i_DiagnosticOutput = True, n_clean_iter = 3,
        )
    
    @classmethod
    def STOFS_3D_Atlantic(cls):
        '''
        Settings for the STOFS 3D Atlantic domain
        '''
        return cls(
            length_width_ratio = 20.0,
            river_threshold = (5, 400),
            width2narcs_option = 'insensitive',
            i_DiagnosticOutput = True,
        )

