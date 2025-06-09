"""
This is the core script for generating river maps.

In serial mode, import the function "make_river_map" as follows:
'from RiverMapper.make_river_map import make_river_map'
, and directly call the function "make_river_map()".

In parallel mode, call the same function through
the parallel driver "river_map_mpi_driver.py".

Download sample applications including sample scripts and inputs for
both the serial mode and the parallel mode here:
http://ccrm.vims.edu/yinglong/feiye/Public/RiverMapper_Samples.tar
"""


# Standard library imports
from builtins import ValueError
from copy import deepcopy
import math
import os
import sys
import time
from glob import glob
from tqdm import tqdm

# Related third-party imports
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import zscore
from scipy.spatial import cKDTree, KDTree
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union
from shapely.validation import explain_validity
from rtree import index
from geopandas.tools import sjoin
from sklearn.neighbors import NearestNeighbors

# Local application/library specific imports
from RiverMapper.config_logger import logger
from RiverMapper.SMS import (
    SMS_ARC, SMS_MAP, curvature,
    dl_lonlat2cpp, get_all_points_from_shp, get_perpendicular_angle,
    lonlat2cpp, write_river_shape_extra
)
from RiverMapper.SMS import cpp2lonlat
from RiverMapper.river_map_tif_preproc import Tif2XYZ, get_elev_from_tiles
from RiverMapper.config_river_map import ConfigRiverMap
from RiverMapper.util import CPP_CRS, z_encoder

# np.seterr(all='raise')  # Needs more attention, see Issue #1


class Geoms_XY():
    '''Class to handle the geometric processing of a list of LineString objects'''

    def __init__(self, geom_list, crs='epsg:4326', add_z=False):
        if not isinstance(geom_list, list):
            raise TypeError('Input must be a list of LineString objects')
        if not all(isinstance(geom, LineString) for geom in geom_list):
            raise TypeError('None LineString object found in the list')

        self.geom_list = [geom for geom in geom_list]
        self.add_z = add_z
        self.xy, self.xy_idx = self.geomlist2xy()
        self.crs = crs

    def geomlist2xy(self):
        '''
        Convert a list of LineString objects to a 2D numpy array of coordinates
        return:
            geoms_xy: 2D numpy array of coordinates
            geoms_xy_idx: 2D numpy array of indices for each LineString object
        '''

        ncol = 3 if self.add_z else 2

        geoms_xy_idx = np.ones((len(self.geom_list), ncol), dtype=int) * -9999

        # record the start and end indices of each LineString's vertices
        idx = 0
        for i, geom in enumerate(self.geom_list):
            geoms_xy_idx[i, 0] = idx
            geoms_xy_idx[i, 1] = idx + len(geom.xy[0])
            idx += len(geom.xy[0])

        # concatenate the coordinates of all vertices of LineString objects
        geoms_xy = np.empty((geoms_xy_idx[-1, 1], ncol), dtype=float)
        for i, geom in enumerate(self.geom_list):
            geoms_xy[geoms_xy_idx[i, 0]:geoms_xy_idx[i, 1], :] = np.array(geom.coords)[:, :ncol]

        return geoms_xy, geoms_xy_idx

    def xy2geomlist(self):
        '''re-construct LineString objects from the 2D numpy array of coordinates'''
        for i, _ in enumerate(self.geom_list):
            self.geom_list[i] = LineString(self.xy[self.xy_idx[i, 0]:self.xy_idx[i, 1], :])

    def update_coords(self, new_coords: np.ndarray):
        """update the coordinates of the LineString objects by
        replacing the original coordinates with new ones in a 2D numpy array"""
        iupdate = False
        if not np.array_equal(self.xy, new_coords):
            self.xy[:, :] = new_coords[:, :]
            self.xy2geomlist()
            iupdate = True
        return iupdate

    def snap_to_points(self, snap_points, target_poly_gdf=None):
        """Snap targeted points to the specified locations (i.e., snap_points)
        , optionally target polygons are provided to limit the snapping to within the polygons"""
        geoms_xy_gdf = points2GeoDataFrame(self.xy, crs=self.crs)

        if target_poly_gdf is None:
            i_target_points = np.ones((self.xy.shape[0], ), dtype=bool)
        else:
            points_in_polys = gpd.tools.sjoin(geoms_xy_gdf, target_poly_gdf, predicate="within", how='left')
            _, idx = np.unique(points_in_polys.index, return_index=True)  # some points belong to multiple polygons
            i_target_points = np.array(~np.isnan(points_in_polys.index_right))[idx]

        _, idx = nearest_neighbour(self.xy[i_target_points, :2], np.c_[snap_points[:, 0], snap_points[:, 1]])
        self.xy[i_target_points, :] = snap_points[idx, :]

        self.xy2geomlist()

    def snap_to_self(self, tolerance):
        """Snap points of self within the specified tolerance,
        Not used in the current version"""
        # Check for approximate equality within tolerance
        unique_rows = np.unique(np.round(self.xy, decimals=int(-np.log10(tolerance))),
                                axis=0, return_index=True)[1]
        # Get the unique rows
        new_xyz = self.xy[unique_rows, :]
        self.snap_to_points(new_xyz)


# ------------------------------------------------------------------
# low level functions mainly for basic geometric processing
# ------------------------------------------------------------------
def moving_average(a, n=10, self_weights=0):
    '''
    Calculate the moving average of a 1D numpy array

    Inputs:
    - a: 1D numpy array
    - n: number of points to average
    - self_weights: weight for the original data,
        when self_weights=0, it is a simple moving average
        when self_weights >> 0, there is little change to the original data
        self_weights cannot be negative

    Outputs:
    - ret2: 1D numpy array of the moving average
    '''
    if self_weights < 0:
        raise ValueError('self_weights cannot be negative')

    if a.shape[0] <= n:
        # linearly interpolate the first and last records
        a = a[:, np.newaxis]
        ret2 = a[0, :] + (a[-1, :] - a[0, :]) * np.linspace(0, 1, len(a))[:, np.newaxis]
        return np.squeeze(ret2)
    else:
        ret = np.cumsum(a, axis=0, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[n-1:] = ret[n-1:] / n

        # re-align time series
        ret1 = ret * 0.0
        m = int(np.floor(n/2))
        ret1[m:-m] = ret[2*m:]

        # fill the first and last few records
        ret1[:m] = ret1[m]
        ret1[-m:] = ret1[-m-1]

        # put more weights on self
        ret2 = (ret1 + self_weights * a) / (1 + self_weights)

        return ret2


def get_angle(a, b, c):
    '''
    Calculate the Angle abc, where a, b, c are 2D coordinates
    The range of the angle is [0, 360) degrees
    (not used currently)
    '''
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def get_angle_diffs(xs, ys):
    '''
    Calculate the angle differences between adjacent segments of a line
    '''
    line = np.c_[xs, ys]
    line_cplx = np.squeeze(line.view(np.complex128))
    angles = np.angle(np.diff(line_cplx))
    angle_diff0 = np.diff(angles)
    angle_diff = np.diff(angles)
    angle_diff[angle_diff0 > np.pi] -= 2 * np.pi
    angle_diff[angle_diff0 < -np.pi] += 2 * np.pi

    return angle_diff


def nearest_neighbour(points_a, points_b):
    '''Find the nearest neighbour of points_a in points_b'''
    tree = cKDTree(points_b)
    return tree.query(points_a)[0], tree.query(points_a)[1]


def get_dist_increment(line):
    '''Calculate the distance increment along a line'''
    line_copy = deepcopy(line)
    line_cplx = np.squeeze(line_copy.view(np.complex128))
    dist = np.absolute(line_cplx[1:] - line_cplx[:-1])
    return dist


def intersect(A, B, C, D):
    '''
    Return true if line segments AB and CD intersect,
    where A, B, C, D are 2D coordinates
    '''
    def ccw(A, B, C):
        '''
        Check if the points A, B, C are in a counterclockwise order;
        A, B, C: 2D coordinates
        '''
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def snap_vertices(line, resolution):
    '''
    Sequentially remove the vertices of a line based on specified resolution,
    The resulting line keeps a subset of the original vertices
    '''
    increment_along_thalweg = get_dist_increment(line)

    idx = 0
    original_seg_length = increment_along_thalweg[0]
    while idx < len(increment_along_thalweg)-1:
        if original_seg_length < resolution[idx]:
            line[idx+1, :] = line[idx, :]  # snap current point to the previous one
            original_seg_length += increment_along_thalweg[idx+1]
        else:
            original_seg_length = increment_along_thalweg[idx+1]
        idx += 1  # move original arc forward

    return line


def gdf2arclist(gdf):
    '''Convert a geopandas dataframe of LineStrings to a list of LineString objects'''
    uu = gdf.unary_union
    if uu.is_empty:
        logger.warning('Warning: No arcs left after cleaning')
        return []
    elif uu.geom_type == 'LineString':
        logger.warning('Warning: Only one arc left after cleaning')
        return [uu]
    elif uu.geom_type == 'MultiLineString':
        arcs = [arc for arc in uu.geoms]
    else:
        raise TypeError('Unexpected geometry type after cleaning')
    return arcs


def geos2SmsArcList(geoms=None):
    '''
    Convert a geopandas dataframe of LineStrings to a list of SMS_ARC objects
    For the purpose of this code, source projection is always "epsg:4326"
    '''
    return [SMS_ARC(points=np.array(line.coords), src_prj='epsg:4326') for line in geoms]


# ------------------------------------------------------------------
# higher level functions for specific tasks involved in river map generation
# ------------------------------------------------------------------
def smooth_thalweg(
    line, ang_diff_shres=np.pi/2.4, nmax=100, smooth_coef=0.2,
    pinned_points=None, option=2
):
    '''Smooth the thalweg by removing sharp turns'''

    xs = line[:, 0]
    ys = line[:, 1]

    if pinned_points is None:
        pinned_points = np.zeros((line.shape[0], ), dtype=bool)

    n = 0
    while n < nmax:
        # pad with zeros at both ends, so end points will not be identified as sharp turns
        angle_diffs = np.r_[0.0, get_angle_diffs(line[:, 0], line[:, 1]), 0.0]
        sharp_turn_idx = abs(angle_diffs) > ang_diff_shres
        sharp_turns = np.argwhere(sharp_turn_idx)[:, 0]
        if not sharp_turns.size:
            break
        else:
            if option == 1:  # deprecated
                # Option 1: move the sharp turn point to a new point, based on
                # the average coordinates of the two adjacent points and the original location.
                # This is not a good option because it may deviate from the real thalweg
                line[sharp_turns, 0] = (
                    np.array((xs[sharp_turns-1] + xs[sharp_turns+1]) / 2) * smooth_coef
                    + xs[sharp_turns] * (1-smooth_coef)
                )
                line[sharp_turns, 1] = (
                    np.array((ys[sharp_turns-1] + ys[sharp_turns+1]) / 2) * smooth_coef
                    + ys[sharp_turns] * (1-smooth_coef)
                )
            elif option == 2:
                # Option 2: remove the sharp turn points.
                # The end points are preserved because the angles are padded with zeros at both ends
                # do not remove pinned points
                remove_idx = np.logical_and(sharp_turn_idx, ~pinned_points.astype(bool))
                line = line[~remove_idx, :]
                pinned_points = pinned_points[~remove_idx]

            n += 1

    if n == nmax:
        logger.warning('warning: smooth_thalweg did not converge in %d steps\n', n)
    else:
        logger.debug('smooth_thalweg converged in %d steps\n', n)

    perp = get_perpendicular_angle(line)

    return line, perp


def river_quality(xs, ys, idx=None):
    '''
    Check river quality based on the number of sharp turns.
    Rivers with exessive number of sharp turns are likely
    from ambiguous channels in the DEM
    '''
    if idx is None:
        idx = np.ones((xs.shape[1], ), dtype=bool)  # all points are valid
    elif sum(idx) < 2:
        return False

    for [x, y] in zip(xs, ys):
        angle_diffs = abs(np.r_[0.0, get_angle_diffs(x[idx], y[idx]), 0.0])
        if np.mean(angle_diffs) > 1.8:
            logger.warning('discarding arc based on the number of sharp turns\n')
            return False

    return True


def smooth_bank(line, xs, ys, xs_other_side, ys_other_side, ang_diff_shres=np.pi/2.4, nmax=100, smooth_coef=0.2):
    '''
    Smooth the bank by removing sharp turns.
    Use with caution, because the smoothed bank may deviate from the real bank.
    '''
    n = 0
    while n < nmax:
        angle_diffs = np.r_[0.0, get_angle_diffs(xs, ys), 0.0]
        sharp_turns = np.argwhere(abs(angle_diffs) > ang_diff_shres)[:, 0]

        if len(sharp_turns) == 0:
            break
        # Step 1: plan to move the sharp turn point to a new point, based on
        # the average coordinates of the two adjacent points and the original location
        xs_moved = (
            np.array((xs[sharp_turns-1] + xs[sharp_turns+1]) / 2) * smooth_coef
            + xs[sharp_turns] * (1-smooth_coef)
        )
        ys_moved = (
            np.array((ys[sharp_turns-1] + ys[sharp_turns+1]) / 2) * smooth_coef
            + ys[sharp_turns] * (1-smooth_coef)
        )

        # Step 2: decide if the planned move is too far (i.e., across the thalweg)
        insert_turns_idx = []
        for i, [x_moved, y_moved, sharp_turn] in enumerate(zip(xs_moved, ys_moved, sharp_turns)):
            invalid_move = (
                intersect(
                    [xs[sharp_turn], ys[sharp_turn]], [x_moved, y_moved],
                    line[sharp_turn, :], line[sharp_turn-1, :])
                + intersect(
                    [xs[sharp_turn], ys[sharp_turn]], [x_moved, y_moved],
                    line[sharp_turn, :], line[sharp_turn+1, :])
            )
            if invalid_move:
                # prepare to insert 2 more points around the sharp turn
                insert_turns_idx.append(i)
            else:
                xs[sharp_turns] = xs_moved
                ys[sharp_turns] = ys_moved

        if len(insert_turns_idx) > 0:
            # replace the point at sharp bend with two more points adjacent to it
            idx = sharp_turns[insert_turns_idx]

            tmp = np.c_[line, xs, ys, xs_other_side, ys_other_side]
            tmp_original = deepcopy(tmp)
            tmp[idx, :] = (tmp_original[idx, :] + tmp_original[idx+1, :])/2
            tmp = np.insert(tmp, idx, (tmp_original[idx, :] + tmp_original[idx-1, :])/2, axis=0)

            if line.shape[1] != 3:
                raise ValueError(f'Error in smooth_bank: line should have three columns for x, y, z, but got {line}')
            line = tmp[:, :3]
            xs = tmp[:, 3]
            ys = tmp[:, 4]
            xs_other_side = tmp[:, 5]
            ys_other_side = tmp[:, 6]

        n += 1

    if n == nmax:
        logger.warning('warning: smooth_bank did not converge in %d steps\n', n)
    else:
        logger.debug('smooth_bank converged in %d steps\n', n)

    perp = get_perpendicular_angle(line)

    return line, xs, ys, xs_other_side, ys_other_side, perp


def nudge_bank(line, perp, xs, ys, dist=None):
    '''
    Nudge banks in the perpendicular direction to the thalweg
    so that the width of the channel is within a specified range
    '''
    if dist is None:
        dist = np.array([35, 500])

    ds = ((line[:, 0] - xs)**2 + (line[:, 1] - ys)**2)**0.5

    idx = ds < dist[0]
    xs[idx] = line[idx, 0] + dist[0] * np.cos(perp[idx])
    ys[idx] = line[idx, 1] + dist[0] * np.sin(perp[idx])

    idx = ds > dist[1]
    xs[idx] = line[idx, 0] + dist[1] * np.cos(perp[idx])
    ys[idx] = line[idx, 1] + dist[1] * np.sin(perp[idx])

    return xs, ys


def fill_missing_values(arr, idx):
    """
    Replace outliers in arr with linearly interpolated values,
    for points outside the range of the linear interpolation,
    the values are replaced with the nearest non-outlier values.
    """

    if len(idx) == 0:
        return arr

    data = arr.copy()

    # If all indices are outliers, return the original data
    if len(idx) == len(data):
        return data

    # Indices of the non-outliers
    non_outlier_indices = np.setdiff1d(np.arange(data.size), idx)

    # Linear interpolation for in-range points
    interpolated_values = np.interp(idx, non_outlier_indices, data[non_outlier_indices])

    # Replace outliers with interpolated values
    data[idx] = interpolated_values

    # Handling outliers at the beginning
    if idx[0] == 0:
        first_non_outlier = non_outlier_indices[0]
        data[0:first_non_outlier] = data[first_non_outlier]

    # Handling outliers at the end
    if idx[-1] == len(data) - 1:
        last_non_outlier = non_outlier_indices[-1]
        data[last_non_outlier + 1:] = data[last_non_outlier]

    return data


def set_eta_thalweg(x, y, z, coastal_z=None, const_depth=1.0):
    """
    Estimate the water surface elevation along the thalweg based on the bathymetry
    """

    if coastal_z is None:
        coastal_z = [0.0, 3.0]

    # approximation based on bathymetry
    eta = np.zeros(x.shape, dtype=float)

    # smooth bathymetry along thalweg because the elevation is smoother than bathymetry
    mean_dl = np.mean(get_dist_increment(np.c_[x, y]))
    z_fixed = fill_missing_values(z, np.argwhere(zscore(z) > 2)[:, 0])  # remove outliers, nEw
    if z_fixed is None:
        logger.error('Failed to fill missing values, z: %s; zscore(z): %s', z, zscore(z))
        raise ValueError('Failed to fill missing values in bathymetry')

    # at least 10 points average to make it smooth
    z_smooth = moving_average(z_fixed, n=int(max(100.0/(mean_dl+1e-6), 10)), self_weights=0)

    # coastal (deep): assume zero
    idx = z_smooth <= coastal_z[0]
    # do nothing, use intial value 0

    # upland (high): assume constant depth
    idx = z_smooth >= coastal_z[1]
    eta[idx] = z_smooth[idx] + const_depth

    # transitional zone: assume linear transition
    idx = (z_smooth > coastal_z[0])*(z_smooth < coastal_z[1])
    eta[idx] = (z_smooth[idx]+const_depth) * (z_smooth[idx]-coastal_z[0])/(coastal_z[1]-coastal_z[0])

    # coastal water depths are mostly larger than the constant depth
    # force minimum depth to be the same as the constant depth,
    # this is necessary to avoid negative depths in the transitional zone
    eta = np.maximum(eta, z_smooth + const_depth)

    return eta


def get_fake_banks(thalweg, const_bank_width=3.0):
    '''
    Get fake banks for the thalweg by adding a constant width on each side
    '''
    perp = get_perpendicular_angle(thalweg)
    x_banks_right = thalweg[:, 0] + const_bank_width / 2.0 * np.cos(perp)
    y_banks_right = thalweg[:, 1] + const_bank_width / 2.0 * np.sin(perp)
    x_banks_left = thalweg[:, 0] + const_bank_width / 2.0 * np.cos(perp + np.pi)
    y_banks_left = thalweg[:, 1] + const_bank_width / 2.0 * np.sin(perp + np.pi)
    bank2bank_width = ((x_banks_left - x_banks_right)**2 + (y_banks_left - y_banks_right)**2)**0.5

    return x_banks_left, y_banks_left, x_banks_right, y_banks_right, perp, bank2bank_width


def get_two_banks(S_list, thalweg, thalweg_eta, search_length, search_steps, min_width, elev_scale):
    '''
    Get two banks for the thalweg by searching in the lateral (cross-channel) direction
    of each point on the thalweg
    '''
    status = 0
    range_arcs = []
    # find perpendicular direction along thalweg at each point
    perp = get_perpendicular_angle(thalweg)

    # find search area for a thalweg, consisting of two lines on each side
    xt_right = thalweg[:, 0] + search_length * np.cos(perp)
    yt_right = thalweg[:, 1] + search_length * np.sin(perp)
    xt_left = thalweg[:, 0] + search_length * np.cos(perp + np.pi)
    yt_left = thalweg[:, 1] + search_length * np.sin(perp + np.pi)

    # Diagnostic: save search area as SMS arcs
    range_arcs += [
        SMS_ARC(points=np.c_[xt_left, yt_left], src_prj='cpp'),
        SMS_ARC(points=np.c_[xt_right, yt_right], src_prj='cpp')
    ]

    # find two banks
    x_banks_left, y_banks_left = get_bank(
        S_list, thalweg[:, 0], thalweg[:, 1], thalweg_eta, xt_left, yt_left, search_steps, elev_scale)
    x_banks_right, y_banks_right = get_bank(
        S_list, thalweg[:, 0], thalweg[:, 1], thalweg_eta, xt_right, yt_right, search_steps, elev_scale)

    if x_banks_left is None or x_banks_right is None:
        logger.warning('warning: elev value is None, failed to find banks ... ')
        status = -2  # status = -2 means bank search failure
        return None, None, None, None, None, None, status

    bank2bank_width = ((x_banks_left - x_banks_right)**2 + (y_banks_left - y_banks_right)**2)**0.5

    # deal with very small widths, temporary test, nEw
    ismall = bank2bank_width < min_width
    if sum(ismall) / len(ismall) > 0.3:
        status = -1  # status = -1 means narrow channel

    x_banks_right[ismall] = thalweg[ismall, 0] + min_width/2 * np.cos(perp[ismall])
    y_banks_right[ismall] = thalweg[ismall, 1] + min_width/2 * np.sin(perp[ismall])
    x_banks_left[ismall] = thalweg[ismall, 0] + min_width/2 * np.cos(perp[ismall] + np.pi)
    y_banks_left[ismall] = thalweg[ismall, 1] + min_width/2 * np.sin(perp[ismall] + np.pi)
    bank2bank_width = ((x_banks_left - x_banks_right)**2 + (y_banks_left - y_banks_right)**2)**0.5

    # last return value is status, 0 means success
    return x_banks_left, y_banks_left, x_banks_right, y_banks_right, perp, bank2bank_width, status


def redistribute_arc(
    line, line_smooth, channel_width, this_nrow_arcs,
    smooth_option=1, R_coef=0.4, length_width_ratio=6.0,
    reso_thres=(5, 300), endpoints_scale=1.0, idryrun=False,
    pinned_points=None,
):
    '''
    Redistribute points along the line according to the curvature
    and the channel width, so that finer resolution is used in the
    curvy and narrow sections.
    Optionally, increase the resolution near the endpoints for better intersections

    pinned_points: a boolean array to indicate which points are retained regardless of the resolution
    '''

    if pinned_points is None:
        pinned_points = np.zeros((line.shape[0]), dtype=bool)  # no pinned points

    # initialize retained_points, all points are retained at first but may be removed later
    # , except for the pinned points which are always retained
    retained_points = np.ones((line.shape[0]), dtype=bool)

    cross_channel_length_scale = channel_width / (this_nrow_arcs-1)

    if line.shape[0] < 2:
        logger.warning('warning: line only has one point, no need for redistributing')
        return line, line_smooth, np.zeros(line.shape[0], dtype=float), retained_points

    # along-river distance, for redistribution
    increment_along_thalweg = get_dist_increment(line[:, :2])

    # # smooth curvature to get rid of small-scale zig-zags
    if smooth_option == -1:  # existing line_smooth
        curv = curvature(line_smooth)
    elif smooth_option == 0:  # no smoothing
        line_smooth = line
        curv = curvature(line)
    elif smooth_option == 1:  # Option 1: moving average
        line_smooth = moving_average(line, n=100, self_weights=2)
        curv = curvature(line_smooth)
    elif smooth_option == 2:  # Option 2: spline (slow and doesn't really work because original points are preserved)
        smooth_factor = 4
        # create spline function
        f, u, *_ = interpolate.splprep([line[:, 0], line[:, 1]], s=10, per=0)
        # create interpolated lists of points
        uint = np.interp(np.arange(0, len(u)-1+1/smooth_factor, 1/smooth_factor), np.arange(0, len(u)), u)
        xint, yint = interpolate.splev(uint, f)
        line_smooth = np.c_[xint, yint]
        curv_sp = curvature(line_smooth)
        curv = curv_sp[::smooth_factor]
    else:
        raise ValueError('unknown smooth_option')

    # plt.scatter(line_smooth[:, 0], line_smooth[:, 1])
    # plt.scatter(line[:, 0], line[:, 1], s=0.3)
    # plt.show()

    # use different interval along the line to calculate curvature
    if smooth_option != 2:
        for i in [1, 2]:  # more combinations -> more conservative, i.e., larger curvature
            for j in range(i):
                curv[j::i] = np.maximum(curv[j::i], curvature(line_smooth[j::i]))

    R = 1.0 / (curv + 1e-10)
    # resolution at points
    river_resolution = np.minimum(R_coef * R, length_width_ratio * cross_channel_length_scale)
    river_resolution = np.minimum(np.maximum(reso_thres[0], river_resolution), reso_thres[1])

    if idryrun:
        river_resolution /= 1.0  # testing

    # increase resolution near endpoints for better intersections
    if endpoints_scale != 1.0:
        for k in [0.5, 1, 2]:
            # tweak within the length of k river widths
            starting_points = np.r_[0.0, np.cumsum(increment_along_thalweg)] < k * np.mean(channel_width)
            river_resolution[starting_points] /= endpoints_scale
            ending_points = (
                np.flip(np.r_[0.0, np.cumsum(np.flip(increment_along_thalweg))]) <
                k * np.mean(channel_width)  # tweak within the length of k river widths
            )
            river_resolution[ending_points] /= endpoints_scale

    # resolution between two points
    river_resolution_seg = (river_resolution[:-1]+river_resolution[1:])/2  # resolution between two points

    # redistribute points along the line according to river_resolution_seg
    idx = 0
    this_seg_length = increment_along_thalweg[0]  # dist between pt0 and pt1
    while idx < len(increment_along_thalweg)-1:  # loop over original points
        if (this_seg_length < river_resolution_seg[idx]) and (pinned_points[idx+1] == 0):
            retained_points[idx+1] = False  # remove idx+1 if the seg between idx and idx+1 is too short
            this_seg_length += increment_along_thalweg[idx+1]  # add the next seg
        else:
            this_seg_length = increment_along_thalweg[idx+1]  # reset the seg length
        idx += 1  # move along original arc from idx to idx+1
    # last point should be retained
    retained_points[-1] = True

    return line[retained_points, :], line_smooth, river_resolution, retained_points


def get_thalweg_neighbors(thalwegs, thalweg_endpoints):
    '''
    Find the neighbors of the thalwegs
    that connect at the endpoints
    '''
    thalweg_neighbors = [None] * len(thalwegs) * 2
    for i, thalweg in enumerate(thalwegs):
        dist = thalweg_endpoints[:, :] - thalweg[0, :]
        same_points = dist[:, 0]**2 + dist[:, 1]**2 < 1e-10**2
        if sum(same_points) > 1:  # intersection point
            thalweg_neighbors[2*i] = np.argwhere(same_points)[0]

    return thalweg_neighbors


def default_width2narcs(width, min_arcs=ConfigRiverMap.DEFAULT_min_arcs, opt='regular'):
    '''
    Default function to determine the number of arcs in the cross-section based on
    the width of the river channel
    '''

    if opt == 'regular':
        if width < 30:
            nrow = min_arcs
        else:
            nrow = int(min_arcs + np.ceil(width / 500))  # add one arc for every increase of 500 m
    elif opt == 'insensitive':
        nrow = int(min_arcs + np.floor(0.35*width**0.25))  # add one arc for every increase of one order of magnitude
        # similar to nrow = min_arcs + max(0, np.floor(log10(width)-1))
    elif opt == 'sensitive':
        nrow = int(min_arcs + np.ceil(width / 100))  # add one arc for every increase of 100 m
    else:
        raise ValueError(f'unknown width2narcs option: {opt}')

    return nrow


def set_inner_arc_position(nrow_arcs, position_type='regular'):
    '''
    Set the position of the inner arcs as a list of floats.
    Each value is the fraction of the width corresponding to the
    inner arc position along the cross-section.

    position_type can take a string or a user-defined function.
    '''

    if callable(position_type):  # user-defined function
        inner_arc_position = position_type(nrow_arcs)
    elif position_type == 'regular':  # evenly spaced inner arcs
        inner_arc_position = np.linspace(0.0, 1.0, nrow_arcs)
    elif position_type == 'fake':  # default levee
        inner_arc_position = np.array([0.0, 6.75, 11.25, 18.0]) / 18
    elif position_type == 'toward_center':  # denser near center
        raise NotImplementedError('toward_center not implemented')
    elif position_type == 'toward_banks':  # denser near banks
        raise NotImplementedError('toward_banks not implemented')
    else:
        raise ValueError(f'unknown inner arc position type: {position_type}')

    return inner_arc_position


def points2GeoDataFrame(points: np.ndarray, crs='epsg:4326'):
    '''
    Takes a (n, 2) array of (lon, lat) coordinates and convert to GeoPanda's GeoDataFrame
    '''
    df = pd.DataFrame({'lon': points[:, 0], 'lat': points[:, 1]})
    df['coords'] = list(zip(df['lon'], df['lat']))
    df['coords'] = df['coords'].apply(Point)
    return gpd.GeoDataFrame(df, geometry='coords', crs=crs)


def list2gdf(obj_list, crs='epsg:4326'):
    '''
    Convert a list of objects to a GeoDataFrame
    '''
    if isinstance(obj_list, list) or isinstance(obj_list, np.ndarray):
        return gpd.GeoDataFrame(index=range(len(obj_list)), crs=crs, geometry=obj_list)
    else:
        raise TypeError('Input objects must be in a list or np.array')


def BinA(A=None, B=None):
    """
    Find the indices of the bins in A that B falls into
    """
    idx = np.argsort(A)
    sorted_A = A[idx]
    sorted_index = np.searchsorted(sorted_A, B)

    Bindex = np.take(idx, sorted_index, mode="raise")
    mask = A[Bindex] != B

    result = np.ma.array(Bindex, mask=mask)
    return result


def snap_closeby_points(pt_xyz: np.ndarray):
    """
    Snap closeby points to the same location.

    Inputs:
    - pt_xyz: a (n, 3) array of points, where the first two columns are x and y coordinates
        and the third column is the snapping threshold around each point
    """
    xyz = deepcopy(pt_xyz)
    for i in range(xyz.shape[0]):
        i_near = abs((xyz[i, 0]+1j*xyz[i, 1]) - (xyz[:, 0]+1j*xyz[:, 1])) < xyz[i, 2] * 0.3
        xyz[i_near, :] = xyz[i, :]
    return xyz


def snap_closeby_points_global(pt_xyz: np.ndarray, snap_point_reso_ratio: float, n_nei=30):
    '''
    Snap closeby points to the same location.
    A double loop is used for each point,
    because we need to update the xyz array in place upon snapping
    Points are processed in two groups to improve efficiency.
    '''

    # save pt_xyz to file for debugging
    # np.savetxt('pt_xyz.txt', pt_xyz, fmt='%.8f')

    xyz = deepcopy(pt_xyz)
    npts = xyz.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(npts, n_nei)).fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    nsnap = 0

    if snap_point_reso_ratio > 0:
        dist_thres = xyz[:, 2] * snap_point_reso_ratio
    else:
        # negative snap_point_reso_ratio means absolute distance
        dist_thres = - snap_point_reso_ratio * np.ones(xyz[:, 2].shape)

    # Two groups of points:
    # Type-I points:
    # The last of n_nei neighbors is within the search radius,
    # so there could be additional closeby neighbors not included in the n_nei neighbors
    # so we need to loop trhough all points to find them
    # Since there should not be many of them, the efficiency is not a concern
    points_with_many_neighbors = (distances[:, -1] < dist_thres)

    logger.info('%d vertices marked for cleaning I', sum(points_with_many_neighbors))
    nsnap += sum(points_with_many_neighbors)
    if sum(points_with_many_neighbors) > 0:
        points_with_many_neighbors = np.where(points_with_many_neighbors)[0]
        for i in points_with_many_neighbors:
            nearby_points = abs((xyz[i, 0]+1j*xyz[i, 1])-(xyz[:, 0]+1j*xyz[:, 1])) < dist_thres[i]
            target = min(np.where(nearby_points)[0])  # always snap to the least index to assure consistency
            xyz[nearby_points, :] = xyz[target, :]

    # Type-II points:
    # The last of n_nei neighbors is outside the search radius
    # and the closest neighbor is within the search radius,
    # so we only need to deal with at most n_nei neighbors.
    # Although a large number of points fall in this category,
    # we only need to loop through n_nei neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=min(npts, n_nei)).fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    distances[distances == 0.0] = 9999
    # for each Type-II point, find its close-by neighbors if any
    points_with_few_neighbors = (np.min(distances, axis=1) <= dist_thres)*(distances[:, -1] >= dist_thres)

    nsnap += sum(points_with_few_neighbors)
    logger.info('%d vertices marked for cleaning II', sum(points_with_few_neighbors))
    if sum(points_with_few_neighbors) > 0:
        points_with_few_neighbors = np.where(points_with_few_neighbors)[0]
        for i in points_with_few_neighbors:
            idx = indices[i, :]
            i_nearby = abs((xyz[i, 0]+1j*xyz[i, 1])-(xyz[idx, 0]+1j*xyz[idx, 1])) < dist_thres[i]
            target = min(idx[np.where(i_nearby)[0]])  # always snap to the least index to assure consistency
            xyz[idx[i_nearby], :] = xyz[target, :]

    return xyz, nsnap


def snap_closeby_lines_global(lines, snap_arc_reso_ratio):
    """
    Snap closeby lines to the same location.
    Break each LineString into segments at inserted points.
    """

    # Create a spatial index for faster querying
    idx = index.Index()
    for i, line in enumerate(lines):
        idx.insert(i, line.bounds)

    updownstreams = [[[], []] for _ in range(len(lines))]
    neighbors = [[[], []] for _ in range(len(lines))]
    for i, line in enumerate(lines):
        tolerance = np.mean(np.array(line.coords), axis=0)[-1]  # average z-coordinate
        neighbor = list(idx.intersection(line.buffer(10 * tolerance).bounds))  # only test nearby lines
        neighbor.remove(i)  # remove self
        neighbors[i] = neighbor
        for j in [0, -1]:
            point = Point(line.coords[j])
            for nei in neighbor:
                if lines[nei] is None:
                    continue
                if lines[nei].intersects(line):
                    updownstreams[i][j].append(nei)

    line_bufs = []
    line_bufs_small = []
    for i, line in enumerate(lines):
        tols = np.mean(np.array([z for x, y, z in line.coords])) * snap_arc_reso_ratio
        line_bufs.append(line.buffer(tols))
        tols = np.mean(np.array([z for x, y, z in line.coords])) * snap_arc_reso_ratio * 0.2
        line_bufs_small.append(line.buffer(tols))

    for i, line in enumerate(lines):
        neighbor = neighbors[i]

        points = [Point(x, y, z) for x, y, z in line.coords]

        for j, point in enumerate(points):
            if j == 0 or j == len(points) - 1:  # endpoints
                # check if endpoints are connected to any neighboring line
                if len(updownstreams[i][min(j, 1)]) == 0:  # hanging endpoints, remove the entire line;
                    # min(j, 1) is 0 for the first point and 1 for the last point
                    lines[i] = None
                    break  # go to next line
                else:  # connected endpoints
                    for nei in neighbor:
                        if lines[nei] is None:
                            continue
                        elif nei in updownstreams[i][min(j, 1)]:
                            continue  # skip the connected line, otherwise the endpoint is always in buffer
                        elif point.within(line_bufs_small[nei]):
                            # remove the endpoint if it is within any neighboring line's buffer
                            # a smaller buffer is used for the endpoints
                            points[j] = None
            else:
                # check if an intermediate point is within any neighboring line's buffer
                for nei in neighbor:
                    if lines[nei] is None:
                        continue  # skip the connected line, otherwise the endpoint is always in buffer
                    if point.within(line_bufs[nei]):
                        points[j] = None

            new_points = [point for point in points if point is not None]
            if len(new_points) >= 2:
                lines[i] = LineString(new_points)
            else:
                lines[i] = None

    lines = [line for line in lines if line is not None]
    lines = [line for line in gpd.GeoDataFrame(geometry=lines).unary_union.geoms]
    return lines


def snap_points_to_lines(arc_points, snap_arc_reso):
    '''
    Calculate the minimum distance between each point and all segments of all lines,
    snap the point to the nearest segment if the distance is smaller than snap_arc_reso
    '''

    # ------begin nested functions---------------------------
    # Function to calculate distance between a point and a segment
    # adapted from pylib
    def mdist(xy, lxy):
        '''
        find the minimum distance of c_[x,y] to a set of lines lxy
          lxy=c_[x1,y1,x2,y2]: each row is a line (x1,y1) -> (x2,y2)

          output: minimum distance of each point to all lines (return a matrix)
        '''

        xy = np.atleast_2d(xy)
        lxy = np.atleast_2d(lxy)
        x, y = xy.T[:, :, None]
        x1, y1, x2, y2 = lxy.T[:, None, :]

        # initialize output
        dist = np.nan * np.ones([x.size, x1.size])

        # find if the foot of perpendicular exists on the line segment
        k = -((x1-x)*(x2-x1)+(y1-y)*(y2-y1))/np.maximum(sys.float_info.epsilon, (x1-x2)**2+(y1-y2)**2)
        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1

        # foot of perpendicular is on the line segment, calculate the distance in the standard way
        fpn = (k >= 0) * (k <= 1)
        dist[fpn] = abs((x+1j*y)-(xn+1j*yn))[fpn]  # pt-line dist

        # foot of perpendicular is not on the line segment,
        # calculate the distance to the nearest end point of the line segment
        dist[~fpn] = np.array(
            np.r_[abs((x+1j*y)-(x1+1j*y1))[None, ...], abs((x+1j*y)-(x2+1j*y2))[None, ...]]
        ).min(axis=0)[~fpn]  # pt-pt dist

        return dist

    # Function to find approximate distances to nearest segments for a vertex
    def find_nearest_distances_approx(points, segments, tree, n_nearest):
        # Query the KD-tree for nearest segments within threshold
        _, segment_indices = tree.query(points, k=n_nearest)

        # Calculate distances for candidate segments
        distances = np.zeros((len(points), n_nearest))
        for i, [point, seg_ind] in enumerate(zip(points, segment_indices)):
            candidate_segments = segments[seg_ind, :]
            distances[i] = mdist(point, candidate_segments)

        return distances
    # ------end nested functions---------------------------

    points = arc_points.xy

    # split lines into segments (a segment only has two points)
    # first, find the number of segments and initialize a np array
    n_seg = 0
    for line in arc_points.geom_list:
        n_seg += len(line.coords) - 1
    segs = np.zeros((n_seg, 4), dtype=float)
    # fill in the segments
    n_seg = 0
    for pt_idx_per_line in arc_points.xy_idx:
        idx = np.arange(pt_idx_per_line[0], pt_idx_per_line[1])
        seg_idx = np.column_stack((idx[:-1], idx[1:]))  # each row is a segment
        line_xy = points[seg_idx, :2].reshape(-1, 4)  # convert to mdist's line format: x1, y1, x2, y2 for each row
        segs[n_seg:n_seg+len(idx)-1, :] = line_xy
        n_seg += len(idx) - 1
    seg_midpoints = np.c_[(segs[:, 0] + segs[:, 2])/2, (segs[:, 1] + segs[:, 3])/2]

    # build a kd-tree for faster querying
    tree = cKDTree(seg_midpoints)
    # 30 adjacent segs is enough in most cases
    dists = find_nearest_distances_approx(points[:, :2], segs, tree, n_nearest=min(30, n_seg))
    # dists is of shape (n_points, n_nearest)
    dists = np.where(dists == 0, np.nan, dists)  # ignore zero distance, i.e., a point is a segment's endpoint

    # to-do: RuntimeWarning: All-NaN slice encountered
    # the smallest non-zero distance of each row is the distance from that point to the nearest segment
    dists = np.nanmin(dists, axis=1)
    # nan is allowed in dists, because dists >= snap_arc_reso will be used to identify valid points
    target_pt_idx = dists >= snap_arc_reso  # identify valid points, which are not too close to any segment
    if not any(target_pt_idx):
        logger.warning('warning: all points are too close to segments, no snapping is performed')
    else:
        # snap to the valid points, i.e., removing the invalid points
        arc_points.snap_to_points(points[target_pt_idx])

    return arc_points


def CloseArcs(polylines: gpd.GeoDataFrame):
    '''
    Close polylines whose endpoints are not connected to other polylines.
    Not actively used; needs more testing.
    '''

    # Get all points from all polylines
    all_points = [point for line in polylines.geometry for point in line.coords]

    for k in [0, -1]:
        end_points = np.array([line.geometry.coords[k] for line in polylines.itertuples()])
        nbrs = NearestNeighbors(n_neighbors=2).fit(end_points)
        distances, indices = nbrs.kneighbors(end_points)
        i_disconnected = np.atleast_2d(np.argwhere(distances[:, 1] > 0.0)).ravel()

        for i, line in zip(i_disconnected, polylines.geometry[i_disconnected]):
            # Connect the line with the nearest point
            if k == 0:
                new_line = LineString([all_points[indices[i, 1]]] + list(line.coords))
            else:
                new_line = LineString(list(line.coords) + [all_points[indices[i, 1]]])
            polylines.loc[i, 'geometry'] = new_line

    return polylines.geometry.to_list(), polylines


def clean_intersections(
        arcs=None, target_polygons=None, snap_points: np.ndarray = None,
        buffer_coef=0.3, idummy=False, i_OCSMesh=False, projected_crs=CPP_CRS
):
    '''
    Deprecated!
    Clean arcs (LineStringList, a list of Shapely's LineString objects)
    by first intersecting them (by unary_union),
    then snapping target points (within 'target_polygons') to 'snap_points'.
    A projected_crs is required for projecting the default lon/lat coordinates
    to a projected coordinate system.
    '''

    arcs0 = deepcopy(arcs)
    if isinstance(arcs0, list):
        arcs_gdf = gpd.GeoDataFrame({'index': range(len(arcs)), 'geometry': arcs})
    elif isinstance(arcs0, gpd.GeoDataFrame):
        arcs_gdf = arcs0
    else:
        raise TypeError()

    target_poly_gdf = None  # initialize
    if isinstance(target_polygons, list):
        target_poly_gdf = list2gdf(target_polygons)
    elif isinstance(target_polygons, gpd.GeoDataFrame):
        target_poly_gdf = target_polygons
    elif target_polygons is None:
        logger.warning('Warning: target polygons do not exist, cleaning all arcs without snapping')
    else:
        raise TypeError()

    arcs = arcs_gdf.geometry.unary_union.geoms

    if idummy or target_polygons is None:
        return [arc for arc in arcs]

    if target_polygons is not None:
        # ------------------snapping -------------------------------------------
        arc_points = Geoms_XY(geom_list=arcs, crs='epsg:4326', add_z=True)
        arc_points.snap_to_points(snap_points[:, :], target_poly_gdf)

        snapped_arcs = np.array([
            arc for arc in gpd.GeoDataFrame(geometry=arc_points.geom_list).unary_union.geoms
        ], dtype=object)

        # ------------------ further cleaning intersection arcs -------------------------------------------
        cleaned_arcs_gdf = gpd.GeoDataFrame(crs='epsg:4326', index=range(len(snapped_arcs)), geometry=snapped_arcs)
        lineInPolys = gpd.tools.sjoin(cleaned_arcs_gdf, target_poly_gdf, predicate="within", how='left')
        _, idx = np.unique(lineInPolys.index, return_index=True)  # some points belong to multiple polygons
        i_cleaned_noninter_arcs = np.array(np.isnan(lineInPolys.index_right))[idx]
        i_cleaned_intersection_arcs = np.array(~np.isnan(lineInPolys.index_right))[idx]

        # keep non intersection arcs as is
        arcs_cleaned_1 = snapped_arcs[i_cleaned_noninter_arcs]
        arcs_cleaned_2 = snapped_arcs[i_cleaned_intersection_arcs]

        # clean intersection arcs that are too close to each other
        # Disabled for OCSMesh because it may cause some polylines to be not closed and thus not polygonized
        # This is not a problem for SMS, because SMS works on polylines not polygons
        if len(arcs_cleaned_2) > 0 and not i_OCSMesh:
            logger.info('cleaning arcs around river intersections ...')
            arcs_cleaned_2_gdf = gpd.GeoDataFrame(
                crs='epsg:4326', index=range(len(arcs_cleaned_2)), geometry=arcs_cleaned_2)
            tmp_gdf = arcs_cleaned_2_gdf.to_crs(projected_crs)
            reso_gs = gpd.GeoSeries(map(Point, snap_points[:, :2]), crs='epsg:4326').to_crs(projected_crs)
            _, idx = nearest_neighbour(np.c_[tmp_gdf.centroid.x, tmp_gdf.centroid.y], np.c_[reso_gs.x, reso_gs.y])

            arc_buffer = dl_lonlat2cpp(snap_points[idx, 2] * buffer_coef, snap_points[idx, 1])
            line_buf_gdf = arcs_cleaned_2_gdf.to_crs(projected_crs).buffer(distance=arc_buffer)

            arc_points = Geoms_XY(geom_list=arcs_cleaned_2, crs='epsg:4326', add_z=True)

            arc_pointsInPolys = gpd.tools.sjoin(
                points2GeoDataFrame(arc_points.xy, crs='epsg:4326').to_crs(projected_crs),
                gpd.GeoDataFrame(geometry=line_buf_gdf), predicate="within", how='left'
            )
            arcs_buffer = [None] * len(arcs_cleaned_2)
            for i, _ in enumerate(arcs_buffer):
                arcs_buffer[i] = []
            for idx, row in arc_pointsInPolys.iterrows():
                arcs_buffer[row.index_right].append(idx)

            invalid_point_idx = np.empty((0, ), dtype=int)
            for i, pt in enumerate(arcs_buffer):
                arc_points_inbuffer = arc_points.xy[pt, :]
                dist, idx = nearest_neighbour(arc_points_inbuffer[:, :2], np.array(arc_points.geom_list[i].xy).T)
                invalid = dist > 1e-10
                if np.any(invalid):
                    invalid_point_idx = np.r_[invalid_point_idx, np.array(pt)[invalid]]
            invalid_point_idx = np.unique(invalid_point_idx)
            i_invalid = np.zeros((len(arc_points.xy), ), dtype=bool)
            i_invalid[invalid_point_idx] = True
            arc_points.snap_to_points(snap_points=arc_points.xy[~i_invalid, :])

            if len(arc_points.geom_list) > 1:
                arcs_cleaned_2 = np.array([
                    arc for arc in gpd.GeoDataFrame(geometry=arc_points.geom_list).unary_union.geoms
                ], dtype=object)
            else:
                arcs_cleaned_2 = np.array([arc_points.geom_list[0]])

        if len(arcs_cleaned_1) + len(arcs_cleaned_2) > 1:
            cleaned_arcs = gpd.GeoDataFrame(geometry=np.r_[arcs_cleaned_1, arcs_cleaned_2]).unary_union.geoms
        else:
            cleaned_arcs = gpd.GeoDataFrame(geometry=np.r_[arcs_cleaned_1, arcs_cleaned_2])

    return [arc for arc in cleaned_arcs]


def clean_arcs(arcs, snap_point_reso_ratio, snap_arc_reso_ratio, n_clean_iter=5):
    '''
    Remove arc vertices that are too close to an arc or a vertex through a few iterations.
    Clean within a small neighborhood then progressively enlarges the search radius,
    which corresponds to a progressive_ratio that takes n_clean_iter iterations to reach 1.
    The maximum number of iterations is n_clean_iter+10.
    A point to point distance and a point to arc distance are used in each iteration,
    which seems to give cleaner results than using only one of them.
    '''

    if n_clean_iter < 1:
        raise ValueError('n_clean_iter must be >= 1')

    logger.info('cleaning all arcs iteratively ...')

    progressive_ratio = (np.arange(1, n_clean_iter+1) / n_clean_iter) ** 2  # small steps at the beginning
    progressive_ratio = np.r_[progressive_ratio, np.ones(10)]

    for i, pratio in enumerate(progressive_ratio):
        logger.info('-------------------Cleaning, Iteration %d -------------------', i+1)

        # points close to each other
        ratio1 = snap_point_reso_ratio * pratio
        logger.info('Snapping nearby points: ratio1 = %s', ratio1)
        arc_points = Geoms_XY(geom_list=arcs, crs='epsg:4326', add_z=True)
        xyz, nsnap = snap_closeby_points_global(arc_points.xy, snap_point_reso_ratio=ratio1)
        if nsnap == 0 and progressive_ratio[i] == max(progressive_ratio):  # no more snapping
            break
        arc_points.update_coords(xyz)
        arcs_gdf = gpd.GeoDataFrame({'index': range(len(arcs)), 'geometry': arc_points.geom_list})

        arcs = gdf2arclist(arcs_gdf)

        # points close to lines
        ratio2 = snap_arc_reso_ratio * pratio
        logger.info('Snapping points close to lines: ratio2 = %s', ratio2)
        arc_points = Geoms_XY(geom_list=arcs, crs='epsg:4326', add_z=True)
        arc_points = snap_points_to_lines(arc_points, snap_arc_reso=arc_points.xy[:, -1]*ratio2)
        arcs_gdf = gpd.GeoDataFrame({'index': range(len(arcs)), 'geometry': arc_points.geom_list})

        arcs = gdf2arclist(arcs_gdf)

    return arcs


def clean_river_arcs_for_ocsmesh(river_arcs=None, total_arcs=None):
    '''
    Snap river arcs to the nearest total arcs (deprecated)
    '''

    total_arcs_geomsxy = Geoms_XY(total_arcs, crs="epsg:4326")

    river_arcs_cleaned = deepcopy(river_arcs)
    for i, river in enumerate(river_arcs_cleaned):
        for j, arc in enumerate(river):
            if arc is not None:
                _, idx = nearest_neighbour(arc.points[:, :2], total_arcs_geomsxy.xy)
                river_arcs_cleaned[i, j].points[:, :2] = total_arcs_geomsxy.xy[idx, :]

    return river_arcs_cleaned


def output_ocsmesh(
    output_dir, arc_shp_fname='total_arcs.shp',
    outline_shp_fname='total_river_outline.shp', area_thres=0.7
):
    '''
    Output the river polygons for OCSMesh.
    It polygonizes the river arcs, then filters out land polygons that are largely outside the river outline.
    Input:
        output_dir: the directory to save the output shapefile
        arc_shp_fname: the name of the shapefile containing all arcs
        outline_shp_fname: the name of the shapefile containing the river outline
        area_thres: the threshold for the 2nd screening,
            which retains polygons that are largely inside the river outline.
            A larger value means more strict filtering.
    Output:
        total_river_polys.shp: the name of the shapefile containing the river polygons
    '''
    logger.info('Preparing OCSMesh products ...')
    time_ocsmesh_start = time.time()

    # convert linestrings to polygons, which creates extra polygons if a land area is enclosed by river arcs
    total_arcs_cleaned = gpd.read_file(f'{output_dir}/{arc_shp_fname}').geometry.tolist()
    total_arcs_cleaned_polys = [poly for poly in polygonize(gpd.GeoSeries(total_arcs_cleaned))]
    total_polys = gpd.GeoDataFrame(
        index=range(len(total_arcs_cleaned_polys)), crs='epsg:4326', geometry=total_arcs_cleaned_polys
    )
    # read the pre-generated river outline, which is used to filter out land areas
    total_polys_cpp = total_polys.to_crs(CPP_CRS)

    total_river_outline_cpp = gpd.read_file(f'{output_dir}/{outline_shp_fname}').to_crs(CPP_CRS)

    # Perform the spatial join first using 'intersects' to find potential matches
    def buffer_polygon(polygon):
        area_sqrt = np.sqrt(polygon.area)  # Calculate the square root of the polygon's area
        buffer_distance = area_sqrt * 0.1  # Buffer distance is 10% of this value
        return polygon.buffer(buffer_distance)  # Return the buffered polygon

    total_river_outline_buf = total_river_outline_cpp.copy()
    total_river_outline_buf['buffered_geometry'] = total_river_outline_buf.geometry.apply(buffer_polygon)
    possible_matches = sjoin(
        total_polys_cpp, total_river_outline_buf, how="inner", predicate="intersects"
    )
    # Initialize an empty list to store indices of strictly within polygons
    poly_idx = []
    # Iterate through each polygon in the original total_river_outline_cpp
    logger.info('Filtering out land areas that are largely outside the river outline ...')
    for idx, river_poly in tqdm(
        total_river_outline_cpp.iterrows(), total=total_river_outline_cpp.shape[0]
    ):
        # Select polygons from total_polys_cpp that intersect with the current river_poly
        intersecting_polys = possible_matches[possible_matches.index_right == idx]

        # Check if each of these intersecting polygons is largely within the current river_poly
        for _, poly in intersecting_polys.iterrows():
            try:
                # Validate geometries before performing operations
                if not poly['geometry'].is_valid:
                    warning_msg = (
                        "Invalid geometry detected in poly: "
                        f"{explain_validity(poly['geometry'])}"
                    )
                    logger.warning(warning_msg)
                    poly['geometry'] = poly['geometry'].buffer(0)  # Attempt to fix the geometry

                if not river_poly['geometry'].is_valid:
                    warning_msg = (
                        "Invalid geometry detected in river_poly: "
                        f"{explain_validity(river_poly['geometry'])}"
                    )
                    logger.warning(warning_msg)
                    river_poly['geometry'] = river_poly['geometry'].buffer(0)  # Attempt to fix the geometry

                # Calculate the intersection area
                intersection_area = poly['geometry'].intersection(river_poly['geometry']).area

                # Calculate the original area of the polygon
                original_area = poly['geometry'].area

                # Check if a big portion of the original area is within the river polygon
                if original_area > 0:  # Occasionally, original_area can be zero
                    if intersection_area / original_area > area_thres:
                        poly_idx.append(poly.name)

            except shapely.errors.GEOSException as e:
                err_msg = f"GEOSException during intersection calculation: {e}"
                logger.error(err_msg)
                continue  # Skip this polygon and move to the next one

    gpd.GeoDataFrame(
        crs=CPP_CRS, geometry=total_polys_cpp.loc[poly_idx, 'geometry']
    ).to_crs('epsg:4326').to_file(
        filename=f'{output_dir}/total_river_polys.shp', driver="ESRI Shapefile"
    )

    # write extra information to file
    logger.info('Writing extra river arc information to file ...')
    if os.path.exists(f'{output_dir}/river_arcs_extra.map'):
        extra_info_map = SMS_MAP(f'{output_dir}/river_arcs_extra.map')
    elif os.path.exists(f'{output_dir}/total_river_arcs_extra.map'):
        extra_info_map = SMS_MAP(f'{output_dir}/total_river_arcs_extra.map')
    else:
        raise FileNotFoundError(
            f'Cannot find river_arcs_extra.map or total_river_arcs_extra.map in {output_dir}'
        )
    write_river_shape_extra(extra_info_map, f'{output_dir}/total_river_arcs_extra.shp')

    logger.info('Preparing OCSMesh products took: %s seconds.', time.time()-time_ocsmesh_start)


def generate_river_outline_polys(river_arcs=None):
    """
    Generate river outline polygons from river arcs.
    Mainly used for OCSMesh.
    """
    river_outline_polygons = []
    river_polygons = [None] * river_arcs.shape[0]

    # save river polygon (enclosed by two out-most arcs and two cross-river transects at both ends)
    for i, river in enumerate(river_arcs):
        # number of valid arcs in this river segment
        idx = np.argwhere(np.not_equal(river, None)).squeeze()
        if len(idx) >= 2:  # at least two rows of arcs to make a polygon
            river_polygons[i] = []
            valid_river_arcs = river[idx]

            # close the river polygon
            mls_uu = unary_union(LineString(np.r_[
                valid_river_arcs[0].points[:, :2],  # the first arc
                np.flipud(valid_river_arcs[-1].points[:, :2]),  # the last arc in reverse order
                valid_river_arcs[0].points[0, :2].reshape(-1, 2)  # connect 1st pt, close the polygon
            ]))
            for polygon in polygonize(mls_uu):
                river_polygons[i].append(polygon)

    # convert to 1D list for conversion to shapefile later
    for river_polygon in river_polygons:
        if river_polygon is not None:
            river_outline_polygons += river_polygon

    return river_outline_polygons


def correct_thalwegs(S_list, dl, line, search_length, perp, mpi_print_prefix, elev_scale=1.0):
    '''
    Correct thalwegs by relocating each thalweg vertex to
    the lowest elevation along the cross-channel transect.
    '''
    i_corrected = False

    x = line[:, 0]
    y = line[:, 1]

    xt_right = line[:, 0] + search_length * np.cos(perp)
    yt_right = line[:, 1] + search_length * np.sin(perp)
    xt_left = line[:, 0] + search_length * np.cos(perp + np.pi)
    yt_left = line[:, 1] + search_length * np.sin(perp + np.pi)

    __search_steps = int(np.max(search_length/dl)) * 2  # scaled by 2 to refine the search
    # give it at least something to search for, i.e., the length of 5 grid points
    __search_steps = max(5, __search_steps)

    x_new = np.full((len(x), 2), np.nan, dtype=float)
    y_new = np.full((len(x), 2), np.nan, dtype=float)
    # initialized to a large number because we want to find the minimum
    elev_new = np.ones((len(x), 2), dtype=float) * 9999
    thalweg_idx = np.ones(xt_right.shape) * 9999
    for k, [xt, yt] in enumerate([[xt_left, yt_left], [xt_right, yt_right]]):
        xts = np.linspace(x, xt, __search_steps, axis=1)
        yts = np.linspace(y, yt, __search_steps, axis=1)

        elevs = get_elev_from_tiles(xts, yts, S_list, scale=elev_scale)
        if elevs is None:
            continue

        # elevs is well defined for the rest of the loop if we reach here
        if np.isnan(elevs).any():
            logger.warning(
                '%s Warning: nan found in elevs when trying to improve Thalweg: %s',
                mpi_print_prefix, np.c_[x, y])
            continue  # aborting thalweg correction for this vertex

        if elevs.shape[1] < 2:
            logger.warning(
                '%s Warning: elevs shape[1] < 2,'
                'not enough valid elevations when trying to improve Thalweg: %s',
                mpi_print_prefix, np.c_[x, y])
            continue  # aborting thalweg correction for this vertex

        # instead of using the minimum, use the median of
        # the lowest 10 elevations to avoid the noise in bathymetry
        n_low = min(10, elevs.shape[1]-1)
        low = np.argpartition(elevs, n_low, axis=1)
        thalweg_idx = np.median(low[:, :n_low], axis=1).astype(int)

        if any(thalweg_idx < 0) or any(thalweg_idx >= xts.shape[1]):
            logger.warning(
                '%s Warning: invalid thalweg index after improvement: %s',
                mpi_print_prefix, np.c_[x, y])
            continue  # aborting thalweg improvement

        x_new[:, k] = xts[range(len(x)), thalweg_idx]
        y_new[:, k] = yts[range(len(x)), thalweg_idx]
        elev_new[:, k] = elevs[range(len(x)), thalweg_idx]

        # correction successful if this point is reached
        i_corrected = True

    left_or_right = elev_new[:, 0] > elev_new[:, 1]
    x_real = x_new[range(len(x)), left_or_right.astype(int)]  # if left higher than right, then use right
    y_real = y_new[range(len(x)), left_or_right.astype(int)]

    return np.c_[x_real, y_real, line[:, 2]], i_corrected


def get_bank(S_list, x, y, thalweg_eta, xt, yt, search_steps=100, elev_scale=1.0):
    '''
    Get a bank on one side of the thalweg (x, y)
    Inputs:
        x, y, eta along a thalweg
        parameter deciding the search area: search_stps
    '''

    # form a search area between the thalweg and the search limit
    xts = np.linspace(x, xt, search_steps, axis=1)
    yts = np.linspace(y, yt, search_steps, axis=1)

    # esitmated eta along the thalweg, expanded laterally to cover the search area
    eta_stream = np.tile(thalweg_eta.reshape(-1, 1), (1, search_steps))

    # ground elevations of the search area
    elevs = get_elev_from_tiles(xts, yts, S_list, scale=elev_scale)
    if elevs is None:
        return None, None  # return None for the bank

    # elevs is well defined if this step is reached
    d_eta = elevs - eta_stream  # closeness to target depth, negative means under water

    bank_idx = np.argmax(d_eta >= 0, axis=1)

    invalid = bank_idx == 0  # thalweg is bank, meaning the width is very narrow
    bank_idx[invalid] = np.argmin(abs(d_eta[invalid, :]), axis=1)

    # special case: all water, not reaching land
    all_water = np.all(d_eta < 0, axis=1)
    bank_idx[all_water] = search_steps - 1  # last point, as wide as possible

    # R_sort_idx = np.argsort(d_eta)
    # bank_idx = np.min(R_sort_idx[:, :min(search_steps, search_tolerance)], axis=1) # search_tolerance=5

    x_banks = xts[range(len(x)), bank_idx]
    y_banks = yts[range(len(x)), bank_idx]

    return x_banks, y_banks


def make_river_map(
        tif_fnames, thalweg_shp_fname, output_dir,
        i_DEM_cache=ConfigRiverMap.DEFAULT_i_DEM_cache,
        selected_thalweg=ConfigRiverMap.DEFAULT_selected_thalweg,
        river_threshold=ConfigRiverMap.DEFAULT_river_threshold,
        min_arcs=ConfigRiverMap.DEFAULT_min_arcs,
        width2narcs_option=ConfigRiverMap.DEFAULT_width2narcs_option,
        custom_width2narcs=ConfigRiverMap.DEFAULT_custom_width2narcs,
        elev_scale=ConfigRiverMap.DEFAULT_elev_scale,
        outer_arcs_positions=ConfigRiverMap.DEFAULT_outer_arcs_positions,
        R_coef=ConfigRiverMap.DEFAULT_R_coef,
        length_width_ratio=ConfigRiverMap.DEFAULT_length_width_ratio,
        along_channel_reso_thres=ConfigRiverMap.DEFAULT_along_channel_reso_thres,
        snap_point_reso_ratio=ConfigRiverMap.DEFAULT_snap_point_reso_ratio,
        snap_arc_reso_ratio=ConfigRiverMap.DEFAULT_snap_arc_reso_ratio,
        n_clean_iter=ConfigRiverMap.DEFAULT_n_clean_iter,
        i_close_poly=ConfigRiverMap.DEFAULT_i_close_poly,
        i_nudge_banks=ConfigRiverMap.DEFAULT_i_nudge_banks,
        i_smooth_banks=ConfigRiverMap.DEFAULT_i_smooth_banks,
        output_prefix=ConfigRiverMap.DEFAULT_output_prefix,
        mpi_print_prefix=ConfigRiverMap.DEFAULT_mpi_print_prefix,
        i_OCSMesh=ConfigRiverMap.DEFAULT_i_OCSMesh,
        i_DiagnosticOutput=ConfigRiverMap.DEFAULT_i_DiagnosticOutput,
        i_pseudo_channel=ConfigRiverMap.DEFAULT_i_pseudo_channel,
        pseudo_channel_width=ConfigRiverMap.DEFAULT_pseudo_channel_width,
        pseudo_channel_dl=ConfigRiverMap.DEFAULT_pseudo_channel_dl,
        nrow_pseudo_channel=ConfigRiverMap.DEFAULT_nrow_pseudo_channel,
        dry_run_only=ConfigRiverMap.DEFAULT_dry_run_only
):
    '''
    [Core routine for making river maps]
    <Mandatory Inputs>:
    | tif_fnames (or a json file to specify file names if there are many tiles) | a list of TIF file names.
    These TIFs should cover the area of interest and be arranged by priority (higher priority ones in front) |

    | thalweg_shp_fname | name of a polyline shapefile containing the thalwegs |
    | output_dir | must specify one |

    <Optional Inputs>:
    These inputs can also be handled by the ConfigRiverMap class (recommended; see details in the online manual).
    | selected_thalweg | integer numpy array | Indices of thalwegs to be processed; mainly used by the parallel driver |
    | output_prefix | string | prefix of the output files, which can be empty |
    | mpi_print_prefix | string | prefix indentifying the rank of the calling mpi process in the output messages |
    | river_threshold | float | minimum and maximum river widths (in meters) to be resolved |
    | min_arcs | integer | minimum number of arcs to resolve a channel (bank arcs, inner arcs and outer arcs) |

    | width2narcs_option | string | pre-defined options ('regular', 'sensitive', 'insensitve')
    or 'custom' if a user-defined function is specified |

    | custom_width2narcs | a user-defined function | a function that takes one parameter 'width'
    and returns the number of arcs in the cross-channel direction |

    | elev_scale | float | scaling factor for elevations;
    a number of -1 (invert elevations) is useful for finding ridges (e.g., of a barrier island) |

    | outer_arc_positions | a tuple of floats | relative position of outer arcs,
    e.g., (0.1, 0.2) will add 2 outer arcs on each side of the river (4 in total), i.e.,
    0.1 x riverwidth and 0.2 x riverwidth from the banks. |

    | R_coef | float | coef controlling the along-channel resolutions at river bends (with a radius of R),
    a larger number leads to coarser resolutions (R * R_coef) |

    | length_width_ratio | float |  the ratio between along-channel resolution and cross-channel resolution |
    | along_channel_reso_thres | a tuple of 2 floats | the minimum and maximum along-channel resolution (in meters) |

    | snap_point_reso_ratio | float | scaling the threshold of the point snapping;
    a negtive number means absolute distance value |

    | snap_arc_reso_ratio | float | scaling the threshold of the arc snapping;
    a negtive number means absolute distance value |

    | n_clean_iter | int | number of iterations for cleaning;
    more iterations produce cleaner intersections and better channel connectivity |

    | i_close_poly | bool | whether to add cross-channel arcs to enclose river arcs into a polygon |
    | i_nudge_banks | bool | whether to nudge the river banks so that river width is within a range |
    | i_smooth_banks | bool | whether to smooth the river banks at abrupt changes of the curvature |

    | i_DEM_cache  | bool | Whether or not to read DEM info from cache.
    Reading from original tif files can be slow, so the default option is True |

    | i_OCSMesh | bool | Whether or not to generate polygon-based outputs to be used as inputs to OCSMesh |
    | i_DiagnosticsOutput | bool | whether to output diagnostic information |

    | i_pseudo_channel | int |
    0:  no pseudo channel, nrow_pseudo_channel and pseudo_channel_width are ignored;
    1: fixed-width channel with nrow elements in the cross-channel direction,
    it can also be used to generate a fixed-width levee for a given levee centerline;
    2: (default) implement a pseudo channel when the river is poorly defined in DEM

    | pseudo_channel_width | float | width of the pseudo channel (in meters) |
    | pseudo_channel_dl | float | along channel resolution of the pseudo channel (in meters) |
    | nrow_pseudo_channel |int| number of rows of elements in the cross-channel direction in the pseudo channel |

    <Outputs>:
    - total_arcs.shp: a shapefile containing all river arcs
    - total_river_polys.shp: a shapefile containing all river polygons
    - total_arcs.map: an SMS map file containing all river arcs, same as total_arcs.shp besides the format
    - other diagnostic outputs (if i_DiagnosticsOutput is True)
    '''

    # ------------------------- other input parameters not exposed to users ---------------------------
    nudge_ratio = np.array((0.3, 2.0))  # ratio between nudging distance to mean half-channel-width
    MapUnit2METER = 1.0  # ratio between actual map unit and meter; deprecated, just use lon/lat for any inputs
    # ------------------------- end other inputs ---------------------------

    # ----------------------   pre-process some inputs -------------------------
    river_threshold = np.array(river_threshold) / MapUnit2METER
    pseudo_channel_length_width_ratio = pseudo_channel_dl / (pseudo_channel_width / (nrow_pseudo_channel - 1))

    if i_pseudo_channel == 1:
        require_dem = False
        endpoints_scale = 1.0
    else:
        require_dem = True
        endpoints_scale = 1.3  # slightly refine near the endpoints to improve connectivity

    if custom_width2narcs is not None:
        if width2narcs_option != 'custom':
            logger.warning(
                '%s warning: custom_width2narcs specified but width2narcs_option is not "custom",'
                'reset to "custom"', mpi_print_prefix
            )
            width2narcs_option = 'custom'

        # decorate the function to accept the same parameters as default_width2narcs
        def decorated_custom_width2narcs(width, min_arcs=min_arcs, opt=width2narcs_option):
            if opt != 'custom':
                raise ValueError('opt must be "custom"')
            # enforce min_arcs and integer return value despite user's evaluation
            return max(min_arcs, int(custom_width2narcs(width)))

        width2narcs = decorated_custom_width2narcs
    else:
        width2narcs = default_width2narcs

    outer_arcs_positions = np.array(outer_arcs_positions).reshape(-1, )  # example: [0.1, 0.2]
    if np.any(outer_arcs_positions <= 0.0):
        err_msg = (
            f'{mpi_print_prefix} outer arcs position must > 0,'
            'a pair of arcs (one on each side of the river) are placed for each position value'
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    if len(outer_arcs_positions) > 0:  # limit cleaning threshold so that outer arcs are not removed
        min_snap_ratio = np.min(outer_arcs_positions) * 0.8
        if snap_point_reso_ratio > min_snap_ratio:
            logger.warning(
                '%s snap_point_reso_ratio %s is too large for outer arcs, reset to %s',
                mpi_print_prefix, snap_point_reso_ratio, min_snap_ratio
            )
            snap_point_reso_ratio = min_snap_ratio
        if snap_arc_reso_ratio > min_snap_ratio:
            logger.warning(
                '%s snap_arc_reso_ratio %s is too large for outer arcs, reset to %s',
                mpi_print_prefix, snap_arc_reso_ratio, min_snap_ratio
            )
            snap_arc_reso_ratio = min_snap_ratio

    # maximum number of arcs to resolve a channel (including bank arcs, inner arcs and outer arcs)
    max_nrow_arcs = (
        width2narcs(4 * river_threshold[-1], min_arcs=min_arcs, opt=width2narcs_option) +
        2 * outer_arcs_positions.size
    )  # 4*river_threshold[-1] to be safe, since 1.1 * river_threshold[-1] is the search length

    # ---------------------- end pre-processing some inputs -------------------------

    if require_dem:
        # ------------------------- read DEM ---------------------------
        main_dem_id = 1
        S_list = []

        nvalid_tile = 0
        for i, tif_fname in enumerate(tif_fnames):
            if tif_fname is None:
                continue
            else:
                nvalid_tile += 1

            S, _ = Tif2XYZ(tif_fname=tif_fname, cache=i_DEM_cache)

            info_msg = (
                f'{mpi_print_prefix} [{os.path.basename(tif_fname)}] DEM box:'
                f'{min(S.lon)}, {min(S.lat)}, {max(S.lon)}, {max(S.lat)}'
            )
            logger.info(info_msg)
            S_list.append(S)

            if nvalid_tile == main_dem_id:
                S_x, S_y = lonlat2cpp(S.lon[:2], S.lat[:2])
                dx = S_x[1] - S_x[0]
                dy = S_y[1] - S_y[0]
                dl = (abs(dx) + abs(dy)) / 2
                search_length = river_threshold[-1] * 1.1
                search_steps = int(river_threshold[-1] / dl) * 2  # scaled by 2 to refine the search

        if nvalid_tile == 0:
            raise ValueError('Fatal Error: no valid DEM tiles')

    # ------------------------- read thalweg ---------------------------
    # In the shapefile, z > 0 means a point is pinned and not to be relocated or removed.
    # The script will preserve the intersections of the input thalwegs automatically
    # by incrementing z by 1 for these points,
    # since they are important for the connectivity of the river network.
    xyz, l2g, curv, _ = get_all_points_from_shp(thalweg_shp_fname, get_z=True)

    # Find duplicated points, which are intersections of thalwegs;
    # increment z by 1 for these points
    distances, _ = KDTree(xyz).query(xyz, k=2)  # k=2 includes the point itself
    duplicates = distances[:, 1] == 0
    xyz[duplicates, 2] += 1

    xyz_lonlat = np.copy(xyz)
    xyz[:, 0], xyz[:, 1] = lonlat2cpp(xyz[:, 0], xyz[:, 1])

    # On the linestring level, additional attributes "dummy" and "keep" can be manually set in
    # the shapefile to control the processing of thalwegs.
    # Read additional field (dummy) if available. All dummy thalwegs will be preserved as is.
    thalweg_gdf = gpd.read_file(thalweg_shp_fname)

    # "dummy" arcs will be preserved as is, without searching for banks and adding inner/outer arcs
    # This is different from points with z > 0, which still searches for banks and adds inner/outer arcs
    if "dummy" in thalweg_gdf.columns:
        dummy = thalweg_gdf['dummy'].values
        logger.info(
            '%s "dummy" field found in %s, dummy thalwegs will be preserved as is',
            mpi_print_prefix, thalweg_shp_fname
        )
    else:
        dummy = np.zeros((len(l2g), ), dtype=int)

    # "keep" arcs will be processed regardless of other criteria,
    # e.g., even when the banks are not found,
    # in which case a pseudo channel will be generated
    if "keep" in thalweg_gdf.columns:
        keep = thalweg_gdf['keep'].values
        keep = np.where(np.isnan(keep), 0, keep).astype(int)  # replace NaN with 0 and convert to int
        keep = np.where(np.equal(keep, None), 0, keep).astype(int)  # replace None with 0 and convert to int
        logger.info(
            '%s "keep" field found in %s, "keep" thalwegs will be processed regardless of other criteria',
            mpi_print_prefix, thalweg_shp_fname
        )
    else:
        keep = np.zeros((len(l2g), ), dtype=int)

    # "id" field is used to identify thalwegs, which is useful for debugging
    if "id" in thalweg_gdf.columns:
        tid = thalweg_gdf['id'].values
    else:
        tid = np.arange(len(l2g))

    thalwegs = []
    thalwegs_lonlat = []
    thalwegs_smooth = []
    thalwegs_curv = []
    thalweg_endpoints = np.empty((0, 2), dtype=float)
    if selected_thalweg is None:
        selected_thalweg = np.arange(len(l2g))

    idummy_thalweg = []  # size will be the number of selected thalwegs
    ikeep_thalweg = []  # size will be the number of selected thalwegs
    thalweg_id = []
    for i, idx in enumerate(l2g):
        if i in selected_thalweg:
            idummy_thalweg.append(dummy[i])
            ikeep_thalweg.append(keep[i])
            thalweg_id.append(tid[i])
            thalwegs.append(xyz[idx, :])
            thalwegs_lonlat.append(xyz_lonlat[idx, :])
            thalwegs_curv.append(curv[idx])
            thalweg_endpoints = np.r_[thalweg_endpoints, np.reshape(xyz[idx][0, :2], (1, 2))]
            thalweg_endpoints = np.r_[thalweg_endpoints, np.reshape(xyz[idx][-1, :2], (1, 2))]
            thalwegs_smooth.append(None)  # populated later in redistribute_arc

    # --------------------------------------------------------------------------------------
    # ------Dry run (finding approximate locations of two banks) ---------------------------
    # --------------------------------------------------------------------------------------
    logger.info('\n%s  ------------------------- Dry run ------------------------- \n', mpi_print_prefix)

    thalweg_endpoints_width = np.full((len(thalwegs) * 2, 1), np.nan, dtype=float)
    # thalwegs_neighbors = np.empty((len(thalwegs), 2), dtype=object)  # [, 0] is head, [, 1] is tail
    thalweg_widths = [None] * len(thalwegs)
    valid_thalwegs = [True] * len(thalwegs)
    original_banks = np.empty((len(thalwegs), 2), dtype=object)  # left bank and right bank for each thalweg

    for i, [thalweg, curv] in enumerate(zip(thalwegs, thalwegs_curv)):
        if idummy_thalweg[i] == 1:
            continue  # still a valid thalweg, to be preserved as is (i.e., no bank search needed)

        bank_search_status = 0  # intialized to 0 (success); -1: narrower than min width; -2: not found
        if require_dem:
            elevs = get_elev_from_tiles(thalweg[:, 0], thalweg[:, 1], S_list, scale=elev_scale)
            if elevs is None:
                logger.warning(
                    "%s warning: some elevs not found on thalweg %d, the thalweg will be neglected ...",
                    mpi_print_prefix, i+1)
                valid_thalwegs[i] = False
                continue

            # set water level at each thalweg point (thalweg_eta), based on observation, simulation, estimation, etc.
            if ikeep_thalweg[i] == 1 or elev_scale < 0:
                # thalweg_eta will be set to 0 for the following cases:
                # 1) keep = 1 is reserved for NHD, where tifs are dummy with 0 (land) and -1 (water)
                # 2) barrier islands (elev_scale < 0) are always nearshore, where 0.0 is a good approximation
                thalweg_eta = 0.0 * elevs
            else:  # normal case, more sophisticated estimation for water level along thalwegs
                thalweg_eta = set_eta_thalweg(thalweg[:, 0], thalweg[:, 1], elevs)

            x_banks_left, y_banks_left, x_banks_right, y_banks_right, _, width, bank_search_status = get_two_banks(
                S_list, thalweg, thalweg_eta, search_length, search_steps,
                min_width=river_threshold[0], elev_scale=elev_scale)
        else:
            x_banks_left, y_banks_left, x_banks_right, y_banks_right, _, width = get_fake_banks(
                thalweg, const_bank_width=pseudo_channel_width)

        # preliminary check
        if bank_search_status == -2:  # failed to find banks
            thalweg_endpoints_width[i*2] = 0.0
            thalweg_endpoints_width[i*2+1] = 0.0
            valid_thalwegs[i] = False
            logger.warning("%s warning: thalweg %d out of DEM coverage, discarding ...", mpi_print_prefix, i+1)
        elif bank_search_status == -1:  # narrower than min width
            thalweg_widths[i] = pseudo_channel_width
            thalweg_endpoints_width[i*2] = pseudo_channel_width
            thalweg_endpoints_width[i*2+1] = pseudo_channel_width
            if bool(ikeep_thalweg[i]):
                logger.warning(
                    "%s warning: thalweg %d (id: %s)'s bank width is smaller than the minimum width, "
                    "it will likely fall back a pseudo channel", mpi_print_prefix, i+1, thalweg_id[i]
                )
            else:
                logger.warning(
                    "%s warning: thalweg %d (id: %s)'s bank width is smaller than the minimum width, discarding ...",
                    mpi_print_prefix, i+1, thalweg_id[i]
                )
                valid_thalwegs[i] = False
        else:  # found banks
            thalweg_widths[i] = width
            thalweg_endpoints_width[i*2] = width[0]
            thalweg_endpoints_width[i*2+1] = width[-1]

            if len(thalweg[:, 0]) < 2:
                logger.warning("%s warning: thalweg %d only has one point, discarding ...", mpi_print_prefix, i+1)
                valid_thalwegs[i] = False

            original_banks[i, 0] = SMS_ARC(points=np.c_[x_banks_left, y_banks_left], src_prj='cpp')
            original_banks[i, 1] = SMS_ARC(points=np.c_[x_banks_right, y_banks_right], src_prj='cpp')

    # output valid thalwegs
    valid_thalwegs_list = [thalwegs_lonlat[i] for i in range(len(thalwegs)) if valid_thalwegs[i]]
    gpd.GeoDataFrame(
        geometry=[LineString(thalweg) for thalweg in valid_thalwegs_list], crs="EPSG:4326"
    ).to_file(
        filename=f'{output_dir}/{output_prefix}valid_thalwegs.shp', driver="ESRI Shapefile")

    if dry_run_only:
        return

    # End of Dry run, the following information should have been gathered by this point:
    # 1) valid river segments (subject to further screening in the wet run)
    # 2) approximate channel width

    # -------------------------------------------------------------
    # ------------------------- Wet run ---------------------------
    # -------------------------------------------------------------
    logger.info('\n%s  ------------------------- Wet run ------------------------- \n', mpi_print_prefix)
    # initialize some lists and arrays
    bank_arcs = np.empty((len(thalwegs), 2), dtype=object)  # left bank and right bank for each thalweg
    # bank_arcs_raw = np.empty((len(thalwegs), 2), dtype=object)
    bank_arcs_final = np.empty((len(thalwegs), 2), dtype=object)

    cc_arcs = np.empty((len(thalwegs), 2), dtype=object)  # [, 0] is head, [, 1] is tail
    inner_arcs = np.empty((len(thalwegs), max_nrow_arcs), dtype=object)  # for storing inner arcs

    # river_arcs is the main river arc output,
    # because it has cross-channel resolution which is a basic requirement for cleaning

    # z field is cross-channel resolution
    river_arcs = np.empty((len(thalwegs), max_nrow_arcs), dtype=object)

    # the following river_arcs_* are auxiliary outputs
    river_arcs_z = np.empty((len(thalwegs), max_nrow_arcs), dtype=object)  # elevation as the z field
    river_arcs_extra = np.empty((len(thalwegs), max_nrow_arcs), dtype=object)  # extra info as the z field

    smoothed_thalwegs = [None] * len(thalwegs)
    redistributed_thalwegs_pre_correction = [None] * len(thalwegs)
    redistributed_thalwegs_after_correction = [None] * len(thalwegs)
    corrected_thalwegs = [None] * len(thalwegs)

    centerlines = [None] * len(thalwegs)
    # thalwegs_cc_reso = [None] * len(thalwegs)
    final_thalwegs = [None] * len(thalwegs)

    # --------------------------- enumerate each thalweg ---------------------------
    dummy_arcs = []
    for i, [thalweg, curv, width, valid_thalweg, thalweg_smooth] in enumerate(
        zip(thalwegs, thalwegs_curv, thalweg_widths, valid_thalwegs, thalwegs_smooth)
    ):
        # logger.info(f'{mpi_print_prefix} Wet run: Arc {i+1} of {len(thalwegs)}')

        if idummy_thalweg[i]:
            logger.info("%s Thalweg %d is dummy, skipping and keep original arcs ...", mpi_print_prefix, i)
            # use along-thalweg resolution as a substitute of cross-channel resolution
            increment_along_thalweg = get_dist_increment(thalweg[:, :2])
            dummy_arcs.append(SMS_ARC(points=np.c_[
                    thalweg[:, 0], thalweg[:, 1], np.r_[increment_along_thalweg, increment_along_thalweg[-1]]
            ], src_prj='cpp'))
            continue

        if not valid_thalweg:
            logger.info("%s Thalweg %d marked as invalid in dry run, skipping ...", mpi_print_prefix, i)
            continue

        # touch up on the estimated width to remove potential noises, nEw
        width = np.round(width, 3)  # cut off at mm to facilitate comparison
        sorted_widths = sorted(np.unique(width))  # find the two smallest widths
        if len(sorted_widths) > 2:
            if sorted_widths[1] / sorted_widths[0] > 10:
                # if the ratio between the smallest and the second smallest is too large,
                # the smallest width is likely to be a noise
                width[width == sorted_widths[0]] = np.mean(width[width > sorted_widths[0]])

        # set number of cross-channel elements
        if i_pseudo_channel == 1:  # for special features like levees and barrier islands
            this_nrow_arcs = nrow_pseudo_channel
        else:
            this_nrow_arcs = min(
                max_nrow_arcs, width2narcs(np.mean(width), min_arcs=min_arcs, opt=width2narcs_option)
            )

        # Redistribute thalwegs vertices
        thalweg, thalweg_smooth, _, retained_idx = redistribute_arc(
            thalweg, thalweg_smooth, width, this_nrow_arcs,
            R_coef=R_coef, length_width_ratio=length_width_ratio, reso_thres=along_channel_reso_thres,
            smooth_option=1, endpoints_scale=endpoints_scale, idryrun=True, pinned_points=thalweg[:, 2]
        )
        smoothed_thalwegs[i] = SMS_ARC(points=np.c_[thalweg_smooth[:, 0], thalweg_smooth[:, 1]], src_prj='cpp')
        redistributed_thalwegs_pre_correction[i] = SMS_ARC(points=np.c_[thalweg[:, 0], thalweg[:, 1]], src_prj='cpp')

        if len(thalweg[:, 0]) < 2:
            logger.warning(
                "%s warning: thalweg %d only has one point after redistribution, neglecting ...",
                mpi_print_prefix, i+1)
            continue

        # Find bank arcs and inner arcs
        # Under 4 cases (Case #), the quality_controlled flag is set to True
        inner_arc_position = None
        quality_controlled = False
        if i_pseudo_channel == 1:  # for special features like levees and barrier islands
            quality_controlled = True  # (Case 1) always true for pseudo channel
            x_banks_left, y_banks_left, x_banks_right, y_banks_right, _, width = get_fake_banks(
                thalweg, const_bank_width=pseudo_channel_width)
            inner_arc_position = set_inner_arc_position(nrow_arcs=nrow_pseudo_channel, position_type='fake')
        else:  # real channels, try to find banks first, even if dry run suggests pseudo channel
            # update thalweg info
            elevs = get_elev_from_tiles(thalweg[:, 0], thalweg[:, 1], S_list, scale=elev_scale)
            if elevs is None:
                raise ValueError(f"{mpi_print_prefix} error: some elevs not found on thalweg {i+1} ...")

            if ikeep_thalweg[i] == 1 or elev_scale < 0:
                # keep = 1 is reserved for NHD, where tifs are dummy with 0 (land) and -1 (water)
                # barrier islands (when elev_scale=-1) are always nearshore, where 0.0 is a good approximation
                thalweg_eta = 0.0 * elevs  # set uniform water level at 0.0, same as land level
            else:  # normal case, more sophisticated methods for water level approximation
                thalweg_eta = set_eta_thalweg(thalweg[:, 0], thalweg[:, 1], elevs)

            # re-make banks based on redistributed thalweg
            x_banks_left, y_banks_left, x_banks_right, y_banks_right, perp, width, bank_search_status = get_two_banks(
                S_list, thalweg, thalweg_eta, search_length, search_steps,
                min_width=river_threshold[0], elev_scale=elev_scale
            )
            if bank_search_status == -2:  # failed to find banks
                logger.warning(
                    "%s warning: banks not found for thalweg %s after redistribution, skipping ...",
                    mpi_print_prefix, i+1
                )
                continue

            # correct thalwegs
            if elev_scale < 0:  # invert z for barrier island
                search_length_for_correction = river_threshold[-1] * 1.1
            else:
                # normal case: assuming the original average width is correct, which is not always true;
                # however, a larger search length can lead to over-correcting the thalwegs to adjacent channels
                search_length_for_correction = moving_average(width, n=10) * 0.5

            thalweg, is_corrected = correct_thalwegs(
                S_list, dl, thalweg, search_length_for_correction, perp, mpi_print_prefix, elev_scale=elev_scale
            )

            if not is_corrected:
                logger.warning(
                    "%s warning: thalweg %s (head: %s; tail: %s) failed to correct, using original thalweg ...",
                    mpi_print_prefix, i+1, thalweg[0], thalweg[-1]
                )
            corrected_thalwegs[i] = SMS_ARC(points=np.c_[thalweg[:, 0], thalweg[:, 1]], src_prj='cpp')

            # update thalweg z
            elevs = get_elev_from_tiles(thalweg[:, 0], thalweg[:, 1], S_list, scale=elev_scale)
            if elevs is None:
                raise ValueError(f"{mpi_print_prefix} error: some elevs not found on thalweg {i+1} ...")
            if ikeep_thalweg[i] == 1 or elev_scale < 0:
                # keep = 1 is reserved for NHD, where tifs are dummy with 0 (land) and -1 (water)
                # barrier islands (when elev_scale=-1) are always nearshore, where 0.0 is a good approximation
                thalweg_eta = 0.0 * elevs  # set uniform water level at 0.0, same as land level
            else:  # normal case, more sophisticated methods for water level approximation
                thalweg_eta = set_eta_thalweg(thalweg[:, 0], thalweg[:, 1], elevs)
            # re-make banks based on corrected thalweg
            x_banks_left, y_banks_left, x_banks_right, y_banks_right, perp, width, bank_search_status = get_two_banks(
                S_list, thalweg, thalweg_eta, search_length, search_steps,
                min_width=river_threshold[0], elev_scale=elev_scale
            )

            if bank_search_status == -2:  # failed to find banks
                logger.warning(
                    "%s warning: banks not found for thalweg %s after correction, skipping ...",
                    mpi_print_prefix, i+1
                )
                continue

            # Redistribute thalwegs vertices
            this_nrow_arcs = min(
                max_nrow_arcs, width2narcs(np.mean(width), min_arcs=min_arcs, opt=width2narcs_option))
            thalweg, thalweg_smooth, _, retained_idx = redistribute_arc(
                thalweg, thalweg_smooth[retained_idx], width, this_nrow_arcs,
                R_coef=R_coef, length_width_ratio=length_width_ratio,
                reso_thres=along_channel_reso_thres, smooth_option=1,
                endpoints_scale=endpoints_scale, pinned_points=thalweg[:, 2]
            )
            smoothed_thalwegs[i] = SMS_ARC(
                points=np.c_[thalweg_smooth[:, 0], thalweg_smooth[:, 1]], src_prj='cpp')
            redistributed_thalwegs_after_correction[i] = SMS_ARC(
                points=np.c_[thalweg[:, 0], thalweg[:, 1]], src_prj='cpp')

            # Smooth thalweg
            thalweg, perp = smooth_thalweg(thalweg, ang_diff_shres=np.pi/2.4, pinned_points=thalweg[:, 2])
            final_thalwegs[i] = SMS_ARC(points=np.c_[thalweg[:, 0], thalweg[:, 1]], src_prj='cpp')

            # update thalweg z
            elevs = get_elev_from_tiles(thalweg[:, 0], thalweg[:, 1], S_list, scale=elev_scale)
            if elevs is None:
                raise ValueError(f"{mpi_print_prefix} error: some elevs not found on thalweg {i+1} ...")
            if ikeep_thalweg[i] == 1 or elev_scale < 0:
                # keep = 1 is reserved for NHD, where tifs are dummy with 0 (land) and -1 (water)
                # barrier islands (when elev_scale=-1) are always nearshore, where 0.0 is a good approximation
                thalweg_eta = 0.0 * elevs  # set uniform water level at 0.0, same as land level
            else:  # normal case, more sophisticated methods for water level approximation
                thalweg_eta = set_eta_thalweg(thalweg[:, 0], thalweg[:, 1], elevs)
            # re-make banks based on redistributed thalweg
            x_banks_left, y_banks_left, x_banks_right, y_banks_right, perp, width, bank_search_status = get_two_banks(
                S_list, thalweg, thalweg_eta, search_length, search_steps,
                min_width=river_threshold[0], elev_scale=elev_scale
            )

            # final touch-ups
            if bank_search_status < 0:  # degenerate case, no further touch-ups needed
                logger.warning(
                    '\n%s warning: degenerate case during final touch-ups,'
                    '(cannot find banks or narrower than min_width)'
                    'for thalweg %s after redistribution', mpi_print_prefix, i+1)
                if i_pseudo_channel == 0:
                    logger.warning('%s warning: neglecting the thalweg ...\n', mpi_print_prefix)
                    continue
                # i_pseudo_channel == 1 already handled above
                elif i_pseudo_channel == 2:
                    # (Case 2): Real river but no banks found, implement a pseudo channel as a fallback
                    quality_controlled = True  # quality check waived for pseudo channel
                    logger.warning('implementing a pseudo channel of width %s ...\n', pseudo_channel_width)

                    # nEw, add an option to coarsen the along-channel resolution
                    # redistibute thalweg vertices again as a pseudo channel, which
                    # typically has a fixed width and a coarser along-channel resolution
                    thalweg, thalweg_smooth, _, retained_idx = redistribute_arc(
                        thalweg, thalweg_smooth[retained_idx],
                        pseudo_channel_width, nrow_pseudo_channel, R_coef=R_coef,
                        length_width_ratio=pseudo_channel_length_width_ratio,
                        reso_thres=(0.3*pseudo_channel_dl, 1.1*pseudo_channel_dl),
                        smooth_option=1, endpoints_scale=endpoints_scale, pinned_points=thalweg[:, 2]
                    )
                    smoothed_thalwegs[i] = SMS_ARC(
                        points=np.c_[thalweg_smooth[:, 0], thalweg_smooth[:, 1]], src_prj='cpp')
                    redistributed_thalwegs_after_correction[i] = SMS_ARC(
                        points=np.c_[thalweg[:, 0], thalweg[:, 1]], src_prj='cpp')

                    x_banks_left, y_banks_left, x_banks_right, y_banks_right, _, width = get_fake_banks(
                        thalweg, const_bank_width=pseudo_channel_width)
                    this_nrow_arcs = nrow_pseudo_channel
                    inner_arc_position = set_inner_arc_position(nrow_arcs=nrow_pseudo_channel, position_type='fake')
            else:  # normal case: touch-ups on the two banks
                if i_nudge_banks:
                    x_banks_left, y_banks_left = nudge_bank(
                        thalweg, perp+np.pi, x_banks_left, y_banks_left, dist=nudge_ratio*0.5*np.mean(width))
                    x_banks_right, y_banks_right = nudge_bank(
                        thalweg, perp, x_banks_right, y_banks_right, dist=nudge_ratio*0.5*np.mean(width))

                # smooth banks
                if i_smooth_banks:
                    thalweg, x_banks_left, y_banks_left, x_banks_right, y_banks_right, perp = smooth_bank(
                        thalweg, x_banks_left, y_banks_left, x_banks_right, y_banks_right)
                    if thalweg is None:
                        continue
                    thalweg, x_banks_right, y_banks_right, x_banks_left, y_banks_left, perp = smooth_bank(
                        thalweg, x_banks_right, y_banks_right, x_banks_left, y_banks_left)
                    if thalweg is None:
                        continue

                # update width
                width = ((x_banks_left-x_banks_right)**2 + (y_banks_left-y_banks_right)**2)**0.5

                # get actual resolution along redistributed/smoothed thalweg
                # thalweg_resolutions[i] = np.c_[(thalweg[:-1, :]+thalweg[1:, :])/2, get_dist_increment(thalweg)]

                # make inner arcs between two banks
                this_nrow_arcs = min(
                    max_nrow_arcs, width2narcs(np.mean(width), min_arcs=min_arcs, opt=width2narcs_option))
                inner_arc_position = set_inner_arc_position(nrow_arcs=this_nrow_arcs, position_type='regular')
            # end if degenerate case
        # end if pseudo channel or real river arcs

        if inner_arc_position is None:  # all cases should already set inner_arc_position, but just in case
            raise ValueError(f"{mpi_print_prefix} error: inner_arc_position is None ...")
        # ------------------------- assemble arcs ---------------------------
        arc_position = np.r_[
            sorted(-outer_arcs_positions), inner_arc_position, 1.0+outer_arcs_positions
        ].reshape(-1, 1)

        x_river_arcs = x_banks_left.reshape(1, -1) + np.matmul(
            arc_position, (x_banks_right-x_banks_left).reshape(1, -1))
        y_river_arcs = y_banks_left.reshape(1, -1) + np.matmul(
            arc_position, (y_banks_right-y_banks_left).reshape(1, -1))

        # record cross-channel (cc) resolution at each thalweg point
        cross_channel_reso = width/(this_nrow_arcs-1)
        for k, line in enumerate(
            [np.c_[x_banks_left, y_banks_left], np.c_[x_banks_right, y_banks_right]]
        ):
            bank_arcs[i, k] = SMS_ARC(points=np.c_[line[:, 0], line[:, 1]], src_prj='cpp')

        # ------------------------- further quality control ---------------------------
        if quality_controlled:
            pass  # pre-qualified as pseudo channels (either preset in the shapefile or as fallback)
        else:  # real river arcs
            if river_quality(x_river_arcs, y_river_arcs):  # quality check passed
                quality_controlled = True  # Case 3: real river, quality check passed
            elif i_pseudo_channel == 2:  # quality check failed, fall back to pseudo channel
                logger.warning(
                    '%s warning: thalweg %d failed quality check, falling back to pseudo channel ...',
                    mpi_print_prefix, i+1)
                x_banks_left, y_banks_left, x_banks_right, y_banks_right, _, width = get_fake_banks(
                    thalweg, const_bank_width=pseudo_channel_width)

                # re-assemble arcs with pseudo channel
                this_nrow_arcs = nrow_pseudo_channel
                inner_arc_position = set_inner_arc_position(nrow_arcs=nrow_pseudo_channel, position_type='fake')

                arc_position = np.r_[
                    sorted(-outer_arcs_positions), inner_arc_position, 1.0+outer_arcs_positions
                ].reshape(-1, 1)

                x_river_arcs = x_banks_left.reshape(1, -1) + np.matmul(
                    arc_position, (x_banks_right-x_banks_left).reshape(1, -1))
                y_river_arcs = y_banks_left.reshape(1, -1) + np.matmul(
                    arc_position, (y_banks_right-y_banks_left).reshape(1, -1))

                # Case 4: real river, banks found in the previous step but with bad quality,
                # falling back to pseudo channel
                quality_controlled = True
            else:
                logger.warning('%s warning: thalweg %d failed quality check, skipping ...', mpi_print_prefix, i+1)
                quality_controlled = False

        # ------------------------- assemble additional arcs ---------------------------
        if quality_controlled:
            for k, [x_river_arc, y_river_arc] in enumerate(zip(x_river_arcs, y_river_arcs)):
                line = np.c_[x_river_arc, y_river_arc]
                if len(line) > 0:
                    # snap vertices too close to each other;
                    # although there will be a final cleanup,
                    # this is necessary because it make the intersection cleaner.
                    line = snap_vertices(line, cross_channel_reso * snap_point_reso_ratio)
                    # ----------Save-------
                    # save final bank arcs
                    if k == 0:  # left bank
                        bank_arcs_final[i, 0] = SMS_ARC(
                            points=np.c_[line[:, 0], line[:, 1], cross_channel_reso[:]], src_prj='cpp')
                    elif k == len(x_river_arcs) - 1:  # right bank
                        bank_arcs_final[i, 1] = SMS_ARC(
                            points=np.c_[line[:, 0], line[:, 1], cross_channel_reso[:]], src_prj='cpp')
                    else:  # inner arcs, k == 0 or k == len(x_river_arcs)-1 are empty
                        inner_arcs[i, k] = SMS_ARC(
                            points=np.c_[line[:, 0], line[:, 1], cross_channel_reso[:]], src_prj='cpp')

                    # save river arcs, these are not subject to cleaning,
                    # thus keeping the original pairing
                    river_arcs[i, k] = SMS_ARC(
                        points=np.c_[line[:, 0], line[:, 1], cross_channel_reso[:]], src_prj='cpp')
                    # save elevation in the z field
                    if require_dem:
                        elevs = get_elev_from_tiles(line[:, 0], line[:, 1], S_list, scale=elev_scale)
                    else:
                        elevs = np.zeros(line.shape[0])
                    if elevs is None:
                        raise ValueError(
                            f"{mpi_print_prefix} error: some elevs not found on river arc {i}, {k}: "
                            f"{cpp2lonlat(line[:, 0], line[:, 1])}\n"
                        )
                        # river_arcs_z[i, k] = None
                    else:
                        river_arcs_z[i, k] = SMS_ARC(
                            points=np.c_[line[:, 0], line[:, 1], elevs], src_prj='cpp', proj_z=False)

                    # save extra information in the z field
                    # at most 6 pieces of info are allowed to be saved
                    # (1) number of along-channel arcs, including outer arcs
                    # (2) if this is an outer-most arc: 1 for outer-most, 0 for inner arcs
                    z_info = np.c_[
                        np.ones(x_river_arc.shape, dtype=int) * len(x_river_arcs),
                        (
                            np.zeros(x_river_arc.shape) + (k == 0) + (k == len(x_river_arcs) - 1)
                        ).astype(bool).astype(int)
                    ]
                    # Note: this_nrow_arcs is the number of inner arcs, not including outer arcs
                    z_encoded = z_encoder(z_info)
                    river_arcs_extra[i, k] = SMS_ARC(
                        points=np.c_[line[:, 0], line[:, 1], z_encoded], src_prj='cpp', proj_z=False)

                # save centerlines
                if len(x_river_arcs) % 2 == 1:  # odd number of arcs
                    k = int((len(x_river_arcs)-1)/2)
                    line = np.c_[x_river_arcs[k], y_river_arcs[k]]
                else:  # even number of arcs
                    k1 = int(len(x_river_arcs)/2 - 1)
                    k2 = int(len(x_river_arcs)/2)
                    line = np.c_[(x_river_arcs[k1]+x_river_arcs[k2])/2, (y_river_arcs[k1]+y_river_arcs[k2])/2]
                centerlines[i] = SMS_ARC(
                    points=np.c_[line[:, 0], line[:, 1], cross_channel_reso[:]], src_prj='cpp')

            if len(x_river_arcs) > 0:
                # assemble cross-channel arcs
                for j in [0, -1]:
                    cc_arcs[i, j] = SMS_ARC(points=np.c_[
                        x_river_arcs[:, :][:, j],
                        y_river_arcs[:, :][:, j],
                        np.tile(cross_channel_reso[:][j], arc_position.size)
                    ], src_prj='cpp')

    # end loop i, enumerating each thalweg

    # -----------------------------------diagnostic outputs ----------------------------
    if any(river_arcs.flatten()):  # not all arcs are None
        # important diagnostic outputs are always generated
        SMS_MAP(arcs=river_arcs_extra.reshape((-1, 1))).writer(
            filename=f'{output_dir}/{output_prefix}river_arcs_extra.map')
        if i_DiagnosticOutput:  # other diagnostic outputs
            SMS_MAP(arcs=river_arcs.reshape((-1, 1))).writer(
                filename=f'{output_dir}/{output_prefix}river_arcs.map')
            SMS_MAP(arcs=river_arcs_z.reshape((-1, 1))).writer(
                filename=f'{output_dir}/{output_prefix}river_arcs_z.map')
            SMS_MAP(arcs=inner_arcs.reshape((-1, 1))).writer(
                filename=f'{output_dir}/{output_prefix}inner_arcs.map')
            SMS_MAP(arcs=bank_arcs.reshape((-1, 1))).writer(
                filename=f'{output_dir}/{output_prefix}bank.map')
            SMS_MAP(arcs=original_banks.reshape((-1, 1))).writer(
                filename=f'{output_dir}/{output_prefix}original_banks.map')

            if i_pseudo_channel != 1:  # skip the following outputs if it is a pseudo channel
                SMS_MAP(arcs=cc_arcs.reshape((-1, 1))).writer(
                    filename=f'{output_dir}/{output_prefix}cc_arcs.map')
                SMS_MAP(arcs=smoothed_thalwegs).writer(
                    filename=f'{output_dir}/{output_prefix}smoothed_thalweg.map')
                SMS_MAP(arcs=redistributed_thalwegs_pre_correction).writer(
                    filename=f'{output_dir}/{output_prefix}redist_thalweg_pre_correction.map')
                SMS_MAP(arcs=redistributed_thalwegs_after_correction).writer(
                    filename=f'{output_dir}/{output_prefix}redist_thalweg_after_correction.map')
                SMS_MAP(arcs=corrected_thalwegs).writer(
                    filename=f'{output_dir}/{output_prefix}corrected_thalweg.map')
    else:
        logger.info('%s No arcs found, aborted writing to *.map', mpi_print_prefix)

    del smoothed_thalwegs[:], redistributed_thalwegs_pre_correction[:]
    del redistributed_thalwegs_after_correction[:], corrected_thalwegs[:]

    # ------------------------- Clean up and finalize ---------------------------
    # assemble arcs groups for cleaning,
    # use river arcs with cross-channel resolution as a base of cleaning threshold
    arc_groups = [arc for river in river_arcs for arc in river if arc is not None]
    if i_close_poly:  # one cc (cross-channel) arc at each end of each river
        arc_groups += [arc for river in cc_arcs for arc in river if arc is not None]
    if any(idummy_thalweg):  # some thalwegs are dummy, so un-processed
        arc_groups += [arc for arc in dummy_arcs if arc is not None]

    # convert to linestrings
    total_arcs_cleaned = [LineString(arc.points[:, :]) for arc in arc_groups if arc is not None]

    if len(total_arcs_cleaned) > 0:
        total_arcs_cleaned = clean_arcs(
            arcs=total_arcs_cleaned, n_clean_iter=n_clean_iter,
            snap_point_reso_ratio=snap_point_reso_ratio, snap_arc_reso_ratio=snap_arc_reso_ratio
        )
    else:
        logger.warning('%s Warning: total_arcs empty', mpi_print_prefix)

    # ------------------------- main outputs ---------------------------
    if any(idummy_thalweg):  # some thalwegs are dummy
        SMS_MAP(arcs=dummy_arcs).writer(filename=f'{output_dir}/{output_prefix}dummy_arcs.map')

    if len(total_arcs_cleaned) > 0:
        SMS_MAP(arcs=centerlines).writer(filename=f'{output_dir}/{output_prefix}centerlines.map')
        del centerlines[:]
        del centerlines

        SMS_MAP(arcs=final_thalwegs).writer(filename=f'{output_dir}/{output_prefix}final_thalweg.map')
        del final_thalwegs[:]
        del final_thalwegs

        SMS_MAP(arcs=bank_arcs_final.reshape((-1, 1))).writer(
            filename=f'{output_dir}/{output_prefix}bank_final.map')

        if len(total_arcs_cleaned) > 0:
            SMS_MAP(arcs=geos2SmsArcList(geoms=total_arcs_cleaned)).writer(
                filename=f'{output_dir}/{output_prefix}total_arcs.map'
            )
            gpd.GeoDataFrame(
                index=range(len(total_arcs_cleaned)), crs='epsg:4326', geometry=total_arcs_cleaned
            ).to_file(filename=f'{output_dir}/{output_prefix}total_arcs.shp', driver="ESRI Shapefile")

        if i_OCSMesh:
            total_river_outline_polys = generate_river_outline_polys(river_arcs=river_arcs)

            if len(total_river_outline_polys) > 0:
                gpd.GeoDataFrame(
                    index=range(len(total_river_outline_polys)),
                    crs='epsg:4326', geometry=total_river_outline_polys
                ).to_file(
                    filename=f'{output_dir}/{output_prefix}river_outline.shp', driver="ESRI Shapefile"
                )
            else:
                logger.warning('%s Warning: total_river_outline_polys empty', mpi_print_prefix)

            # output if in serial mode, otherwise, the output is handled by the mpi driver
            if output_prefix == '':  # serial
                output_ocsmesh(output_dir, 'total_arcs.shp', 'river_outline.shp')

            del total_river_outline_polys

    else:
        logger.warning('%s Warning: total_sms_arcs_cleaned empty, skip writing to *.map', mpi_print_prefix)

    del total_arcs_cleaned[:]
    del total_arcs_cleaned


if __name__ == "__main__":
    print('done')
