#!/usr/bin/env python
"""
This script provides methods of dealing with SMS maps for RiverMapper

<Sample SMS map only containing arcs>
MAP VERSION 8
BEGCOV
COVFLDR "Area Property"
COVNAME "Area Property"
COVELEV 0.000000
COVID 26200
COVGUID 57a1fdc1-d908-44d3-befe-8785288e69e7
COVATTS VISIBLE 1
COVATTS ACTIVECOVERAGE Area Property
COV_WKT GEOGCS["GCS_WGS_1984",DATUM["WGS84",SPHEROID["WGS84",6378137,298.257223563]],\
PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]END_COV_WKT
COV_VERT_DATUM 0
COV_VERT_UNITS 0
COVATTS PROPERTIES MESH_GENERATION
NODE
XY -28.76 73.25 0.0
ID 1
END
NODE
XY -49.76 43.09 0.0
ID 2
END
NODE
XY -24.91 56.11 0.0
ID 3
END
NODE
XY -31.34 44.28 0.0
ID 4
END
ARC
ID 1
ARCELEVATION 0.000000
NODES        1        2
ARCVERTICES 6
-4.020000000000001 75.980000000000004 0.000000000000000
27.030000000000001 68.600000000000009 0.000000000000000
45.730000000000004 45.450000000000003 0.000000000000000
40.829999999999998 26.420000000000002 0.000000000000000
8.830000000000000 18.789999999999999 0.000000000000000
-37.609999999999999 30.300000000000001 0.000000000000000
DISTNODE 0
NODETYPE 0
ARCBIAS 1
MERGED 0 0 0 0
END
ARC
ID 2
ARCELEVATION 0.000000
NODES        3        4
ARCVERTICES 3
-4.300000000000000 57.250000000000000 0.000000000000000
-0.160000000000000 45.130000000000003 0.000000000000000
-20.059999999999999 39.300000000000004 0.000000000000000
DISTNODE 0
NODETYPE 0
ARCBIAS 1
MERGED 0 0 0 0
END
ENDCOV
BEGTS
LEND
"""


import os
import glob
import re

import numpy as np
from shapely.geometry import LineString
import geopandas as gpd

from RiverMapper.util import silentremove, z_decoder


def lonlat2cpp(lon, lat, lon0=0):
    """
    Convert lon, lat to Cartesian coordinates in meters
    """
    earth_radius = 6378206.4

    lon_radian, lat_radian = lon/180*np.pi, lat/180*np.pi
    lon0_radian = lon0 / 180 * np.pi

    xout = earth_radius * (lon_radian - lon0_radian)  # * np.cos(lat0_radian)
    yout = earth_radius * lat_radian

    return [xout, yout]


def cpp2lonlat(x, y, lon0=0, lat0=0):
    '''
    Convert Cartesian coordinates in meters to lon, lat
    '''
    earth_radius = 6378206.4

    lon0_radian, lat0_radian = lon0 / 180*np.pi, lat0 / 180*np.pi

    lon_radian = lon0_radian + x / earth_radius / np.cos(lat0_radian)
    lat_radian = y / earth_radius

    lon, lat = lon_radian * 180 / np.pi, lat_radian * 180 / np.pi

    return [lon, lat]


def dl_cpp2lonlat(dl, lat0=0):
    '''
    Convert Cartesian distance to lon, lat distance
    '''
    earth_radius = 6378206.4
    lat0_radian = lat0 / 180 * np.pi
    dlon_radian = dl / earth_radius / np.cos(lat0_radian)
    dlon = dlon_radian * 180 / np.pi
    return dlon

    # x0 = 0.0
    # x1 = dl
    # y0 = 0.0
    # y1 = 0.0
    # lon0, lat0 = cpp2lonlat(x0, y0)
    # lon1, lat1 = cpp2lonlat(x1, y1)
    # return abs((lon0-lon1)+1j*(lat0-lat1))


def dl_lonlat2cpp(dl, lat0=0):
    '''
    Convert lon, lat distance to Cartesian distance
    '''
    earth_radius = 6378206.4
    lat0_radian = lat0 / 180 * np.pi
    dl_radian = dl * np.pi / 180
    dl_cpp = dl_radian * earth_radius * np.cos(lat0_radian)
    return dl_cpp


def curvature(pts):
    '''Calculate curvature of a line defined by points,
    Curvature is 0 for a line with less than 3 points'''

    if len(pts[:, 0]) < 3:
        cur = np.zeros((len(pts[:, 0])))
    else:
        dx = np.gradient(pts[:, 0])  # first derivatives
        dy = np.gradient(pts[:, 1])

        d2x = np.gradient(dx)  # second derivatives
        d2y = np.gradient(dy)

        cur = np.abs(dx * d2y - d2x * dy) / ((dx * dx + dy * dy)**1.5 + 1e-20)

    return cur


def get_perpendicular_angle(line):
    '''Get the angle of the perpendicular line at each point of the line'''
    line_cplx = np.squeeze(line[:, :2].copy().view(np.complex128))
    angles = np.angle(np.diff(line_cplx))
    angle_diff0 = np.diff(angles)
    angle_diff = np.diff(angles)
    angle_diff[angle_diff0 > np.pi] -= 2 * np.pi
    angle_diff[angle_diff0 < -np.pi] += 2 * np.pi
    perp = angles[:-1] + angle_diff / 2 - np.pi / 2
    perp = np.r_[angles[0] - np.pi / 2, perp, angles[-1] - np.pi / 2]

    return perp


def normalize_vec(x, y):
    """Normalize a vector by its length"""
    distance = np.sqrt(x*x+y*y)
    return x/distance, y/distance


def make_offset_poly(old_x, old_y, offset, outer_ccw=1):
    """Make a polygon from a polyline by offsetting it
    in the perpendicular direction by a distance on both sides"""
    num_points = len(old_x)
    new_x = []
    new_y = []

    for curr in range(num_points):
        prev_pt = (curr + num_points - 1) % num_points
        next_pt = (curr + 1) % num_points

        vn_x = old_x[next_pt] - old_x[curr]
        vn_y = old_y[next_pt] - old_y[curr]
        vnn_x, vnn_y = normalize_vec(vn_x, vn_y)
        nnn_x = vnn_y
        nnn_y = - vnn_x

        vp_x = old_x[curr] - old_x[prev_pt]
        vp_y = old_y[curr] - old_y[prev_pt]
        vpn_x, vpn_y = normalize_vec(vp_x, vp_y)
        npn_x = vpn_y * outer_ccw
        npn_y = -vpn_x * outer_ccw

        bis_x = (nnn_x + npn_x) * outer_ccw
        bis_y = (nnn_y + npn_y) * outer_ccw

        bisn_x, bisn_y = normalize_vec(bis_x,  bis_y)
        bislen = offset / np.sqrt(1 + nnn_x * npn_x + nnn_y * npn_y)

        new_x.append(old_x[curr] + bislen * bisn_x)
        new_y.append(old_y[curr] + bislen * bisn_y)

    return new_x, new_y


def redistribute(x, y, length=None, num_points=None):
    '''
    Redistribute points along a line to have equal distance between them

    Inputs:
    - x, y: coordinates of the line
    - length: the desired distance between points
    - num_points: the desired number of points (if length is not specified)
    '''
    line = LineString(np.c_[x, y])

    if length is None and num_points is None:
        raise ValueError('either length or num_points must be specified')

    if length is not None:
        num_points = max(2, int(line.length / length))

    new_points = [line.interpolate(i/float(num_points - 1), normalized=True) for i in range(num_points)]
    x_subsampled = [p.x for p in new_points]
    y_subsampled = [p.y for p in new_points]

    # if iplot:
    #     plt.plot(x, y, '+')
    #     plt.plot(x_subsampled, y_subsampled, 'o')
    #     plt.axis('equal')
    #     plt.show()

    return x_subsampled, y_subsampled, new_points


def merge_maps(mapfile_glob_str, merged_fname):
    '''Merge multiple SMS maps into one map file'''

    if merged_fname is not None:
        silentremove(merged_fname)

    map_file_list = glob.glob(mapfile_glob_str)
    if len(map_file_list) > 0:
        map_list = [SMS_MAP(filename=map_file) for map_file in map_file_list]

        total_map = map_list[0]
        for this_map in map_list[1:]:
            total_map += this_map
        total_map.writer(merged_fname)
    else:
        # print(f'warning: outputs do not exist: {mapfile_glob_str}, abort writing to map')
        return None

    return total_map


class SMS_ARC():
    '''class for manipulating arcs in SMS maps'''
    def __init__(self, points=None, node_idx=None, src_prj=None, dst_prj='epsg:4326', proj_z=True):
        """
        points: 2D array of shape (n_points, 3) or (n_points, 2)
        node_idx: 1D array, normally the first and last points of the arc
        proj_z: whether to treate z values as a measure on the 2D-plane and project it to dst_prj
          note: z values, i.e., points[:, 2], are used to store information of the arc,
          e.g., width (needs projection to be meaningful in the new coordinate system)
                or elevation (no need to project because it is a measure in the vertical dimension)
        """

        # self.isDummy = (len(points) == 0)
        if node_idx is None:
            node_idx = [0, -1]

        if src_prj is None:
            raise ValueError('source projection not specified when initializing SMS_ARC')

        if src_prj == 'cpp' and dst_prj == 'epsg:4326':
            points[:, 0], points[:, 1] = cpp2lonlat(points[:, 0], points[:, 1])
            if points.shape[1] == 3 and proj_z:
                points[:, 2] = dl_cpp2lonlat(points[:, 2], lat0=points[:, 1])

        npoints, ncol = points.shape
        self.points = np.zeros((npoints, 3), dtype=float)
        self.points[:, :min(3, ncol)] = points[:, :min(3, ncol)]

        self.nodes = self.points[node_idx, :]
        self.arcvertices = np.delete(self.points, node_idx, axis=0)
        self.arcnode_glb_ids = np.empty(self.nodes[:, 0].shape, dtype=int)

        self.arc_hats = np.zeros((4, 3), dtype=float)
        self.arc_hat_length = -1

    def make_hats(self, arc_hat_length=-1):
        '''
        Make hats at the ends of the arc (deprecated)
        '''
        if arc_hat_length <= 0:
            raise ValueError('Arc hat length <= 0')
        else:
            self.arc_hat_length = arc_hat_length

        # make hats (a perpendicular line at each of the arc ends)
        for i, [x0, y0, xx, yy] in enumerate([
            [self.points[0, 0], self.points[0, 1], self.points[1, 0], self.points[1, 1]],
            [self.points[-1, 0], self.points[-1, 1], self.points[-2, 0], self.points[-2, 1]],
        ]):
            xt = xx - x0
            yt = yy - y0
            st = (xt**2 + yt**2)**0.5
            xt = xt/st*arc_hat_length/2
            yt = yt/st*arc_hat_length/2

            self.arc_hats[2*i, 0] = x0 - yt
            self.arc_hats[2*i, 1] = y0 + xt
            self.arc_hats[2*i+1, 0] = x0 + yt
            self.arc_hats[2*i+1, 1] = y0 - xt

        # import matplotlib.pyplot as plt
        # plt.scatter(self.points[:, 0], self.points[:, 1])
        # plt.scatter(self.arc_hats[:, 0], self.arc_hats[:, 1])
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

        return [SMS_ARC(points=self.arc_hats[:2, :]), SMS_ARC(points=self.arc_hats[2:, :])]


class SMS_MAP():
    '''class for manipulating SMS maps'''
    def __init__(self, filename=None, arcs=None, detached_nodes=None, epsg=4326):
        # main attributes
        self.epsg = None
        self.arcs = []  # expecting to be a list of SMS_ARC
        self.nodes = None  # expecting to be a 2D array of shape (n_nodes, 3)
        self.detached_nodes = None  # expecting to be a 2D array of shape (n_detached_nodes, 3)
        self.valid = True
        self.n_xyz = None  # number of all points in all arcs and detached nodes
        self.xyz = None  # coordinates of all arcs and detached nodes, shape (n_xyz, 3)
        self.l2g = None  # local-to-global node indices mapping, i.e.,
        # l2g[i] contains a list of global indices of the ith arc

        # read from file if filename is provided
        if filename is not None:
            self.reader(filename=filename)
            return

        # otherwise, initialize from arcs and detached_nodes
        if arcs is None:
            self.arcs = []
        elif isinstance(arcs, list):
            arcs = [arc for arc in arcs if arc is not None]
            self.arcs = arcs
        elif isinstance(arcs, np.ndarray):
            self.arcs = np.squeeze(arcs).tolist()

        if detached_nodes is None:
            self.detached_nodes = np.zeros((0, 3), dtype=float)
        elif isinstance(detached_nodes, list):
            detached_nodes = [node for node in detached_nodes if node is not None]
            self.detached_nodes = np.array(detached_nodes)
        elif isinstance(detached_nodes, np.ndarray):
            self.detached_nodes = detached_nodes

        self.epsg = epsg

        if self.arcs == [] and len(self.detached_nodes) == 0:
            self.valid = False
        elif np.all(np.array(self.arcs) is None) and np.all(self.detached_nodes is None):
            self.valid = False
        else:
            self.valid = True

    def __add__(self, other):
        self.arcs = self.arcs + other.arcs
        self.detached_nodes = np.r_[self.detached_nodes, other.detached_nodes]
        return SMS_MAP(arcs=self.arcs, detached_nodes=self.detached_nodes, epsg=self.epsg)

    def get_xyz(self):
        """Get xyz coordinates of all arcs and detached nodes,
        and local-to-global node indices mapping"""

        self.n_xyz = 0
        self.l2g = []

        for arc in self.arcs:
            self.l2g.append(np.array(np.arange(self.n_xyz, self.n_xyz+len(arc.points))))
            self.n_xyz += len(arc.points)

        self.xyz = np.zeros((self.n_xyz, 3), dtype=float)
        for ids, arc in zip(self.l2g, self.arcs):
            self.xyz[ids, :] = arc.points

        return self.xyz, self.l2g

    def reader(self, filename='test.map'):
        '''Read SMS map file and store information in the class attributes'''

        self.n_glb_nodes = 0
        self.n_arcs = 0
        self.n_detached_nodes = 0

        arc_nodes = []
        self.detached_nodes = np.zeros((0, 3), dtype=float)
        self.nodes = np.zeros((0, 3), dtype=float)
        with open(filename, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                strs = re.split(' +', line.strip())
                if strs[0] == 'COV_WKT':
                    if "WGS_1984" in line:
                        self.epsg = 4326
                    else:
                        raise ValueError(f'Projection not supported: {line}')
                elif strs[0] == 'NODE':
                    line = f.readline()
                    strs = re.split(' +', line.strip())
                    self.n_glb_nodes += 1
                    self.nodes = np.append(
                        self.nodes,
                        np.reshape([float(strs[1]), float(strs[2]), float(strs[3])], (1, 3)), axis=0)
                elif line.strip() == 'POINT':
                    line = f.readline()
                    strs = re.split(' +', line.strip())
                    self.n_detached_nodes += 1
                    self.detached_nodes = np.append(
                        self.detached_nodes,
                        np.reshape([float(strs[1]), float(strs[2]), float(strs[3])], (1, 3)), axis=0)
                elif line.strip() == 'ARC':
                    self.n_arcs += 1
                elif strs[0] == 'NODES':
                    this_arc_node_idx = np.array([int(strs[1]), int(strs[2])])-1
                elif strs[0] == 'ARCVERTICES':
                    this_arc_nvert = int(strs[1])
                    this_arc_verts = np.zeros((this_arc_nvert, 3), dtype=float)
                    for i in range(this_arc_nvert):
                        strs = f.readline().strip().split(' ')
                        this_arc_verts[i, :] = np.array([strs[0], strs[1], strs[2]])
                    node_1 = np.reshape(self.nodes[this_arc_node_idx[0], :], (1, 3))
                    node_2 = np.reshape(self.nodes[this_arc_node_idx[1], :], (1, 3))
                    this_arc = SMS_ARC(points=np.r_[node_1, this_arc_verts, node_2], src_prj=f'epsg: {self.epsg}')
                    self.arcs.append(this_arc)
                    arc_nodes.append(this_arc_node_idx[0])
                    arc_nodes.append(this_arc_node_idx[1])

    def writer(self, filename='test.map'):
        '''Write SMS map file from the class attributes'''

        if not self.valid:
            print(f'No features found in map, aborting writing to {filename}')
            return

        fpath = os.path.dirname(filename)
        if not os.path.exists(fpath):
            os.makedirs(fpath, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            # write header
            f.write('MAP VERSION 8\n')
            f.write('BEGCOV\n')
            # f.write('COVFLDR "Area Property"\n')
            # f.write('COVNAME "Area Property"\n')
            # f.write('COVELEV 0.000000\n')
            f.write('COVID 26200\n')
            f.write('COVGUID 57a1fdc1-d908-44d3-befe-8785288e69e7\n')
            f.write('COVATTS VISIBLE 1\n')
            f.write('COVATTS ACTIVECOVERAGE Area Property\n')
            if self.epsg == 4326:
                f.write('COV_WKT GEOGCS["GCS_WGS_1984",DATUM["WGS84",SPHEROID["WGS84",6378137,298.257223563]],'
                        'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]END_COV_WKT \n')

            elif self.epsg == 26918:
                f.write('COV_WKT PROJCS["NAD83 / UTM zone 18N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
                        'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],'
                        'TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6269"]],'
                        'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                        'AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],'
                        'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],'
                        'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
                        'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
                        'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26918"]]END_COV_WKT')
            else:
                raise ValueError(f'Projection not supported: {self.epsg}')
            f.write('COV_VERT_DATUM 0\n')
            f.write('COV_VERT_UNITS 0\n')
            f.write('COVATTS PROPERTIES MESH_GENERATION\n')

            node_counter = 0
            for i, arc in enumerate(self.arcs):
                if arc is None:
                    continue
                for j, node in enumerate(arc.nodes):
                    node_counter += 1
                    self.arcs[i].arcnode_glb_ids[j] = node_counter
                    f.write('NODE\n')
                    f.write(f'XY {node[0]} {node[1]} {node[2]}\n')
                    f.write(f'ID {node_counter}\n')
                    f.write('END\n')

            for i, node in enumerate(self.detached_nodes):
                if node is None:
                    continue
                node_counter += 1
                f.write('POINT\n')
                f.write(f'XY {node[0]} {node[1]} {node[2]}\n')
                f.write(f'ID {node_counter}\n')
                f.write('END\n')

            for i, arc in enumerate(self.arcs):
                if arc is None:
                    continue
                f.write('ARC\n')
                f.write(f'ID {i+1}\n')
                f.write('ARCELEVATION 0.00\n')
                f.write(f'NODES {" ".join(arc.arcnode_glb_ids.astype(str))}\n')
                f.write(f'ARCVERTICES {len(arc.arcvertices)}\n')
                for vertex in arc.arcvertices:
                    f.write(f'{vertex[0]} {vertex[1]} {vertex[2]}\n')
                f.write('END\n')

            f.write('ENDCOV\n')
            f.write('BEGTS\n')
            f.write('LEND\n')

    def to_GeoDataFrame(self):
        '''Convert the SMS map to a GeoDataFrame'''
        return gpd.GeoDataFrame(geometry=[LineString(line.points) for line in self.arcs if line is not None])

    def to_LineStringList(self):
        '''Convert the SMS map to a list of LineStrings'''
        return [LineString(line.points) for line in self.arcs if line is not None]


class Levee_SMS_MAP(SMS_MAP):
    '''class for manipulating levee maps, which are derived from SMS maps'''
    def __init__(self, arcs=None, epsg=4326):
        if arcs is None:
            arcs = []

        super().__init__(arcs=arcs, epsg=epsg)
        self.centerline_list = arcs
        self.subsampled_centerline_list = []
        self.offsetline_list = []

    def make_levee_maps(self, offset_list=None, subsample=None):
        '''Make levee maps from the centerline arcs by subsampling and offsetting'''
        if offset_list is None:
            offset_list = [-5, 5, -15, 15]
        if subsample is None:
            subsample = [300, 10]

        for arc in self.centerline_list:
            x_sub, y_sub, _ = redistribute(x=arc.points[:, 0], y=arc.points[:, 1], length=subsample[0])
            self.subsampled_centerline_list.append(SMS_ARC(points=np.c_[x_sub, y_sub]))

            for offset in offset_list:
                x_off, y_off = make_offset_poly(x_sub, y_sub, offset)
                self.offsetline_list.append(SMS_ARC(points=np.c_[x_off, y_off]))
        return SMS_MAP(arcs=self.subsampled_centerline_list), SMS_MAP(arcs=self.offsetline_list)


def get_all_points_from_shp(fname, silent=True, get_z=False):
    '''Read all points from a shapefile and calculate curvature and perpendicular angles at each point'''

    if not silent:
        print(f'reading shapefile: {fname}')

    # using geopandas, which seems more efficient than pyshp
    shapefile = gpd.read_file(fname)
    npts = 0
    nvalid_shps = 0
    for i in range(shapefile.shape[0]):
        if shapefile.iloc[i, :]['geometry'] is None:
            raise ValueError(f"shape {i+1} of {fname} is invalid")
        # cannot take multiparts
        if shapefile.iloc[i, :]['geometry'].geom_type == 'MultiLineString':
            raise ValueError("MultiLineString not supported")
        try:
            shp_points = np.array(shapefile.iloc[i, :]['geometry'].coords.xy).shape[1]
        except AttributeError: # nEw
            print(f"warning: shape {i+1} of {shapefile.shape[0]} is invalid")
            continue
        except NotImplementedError:  # nEw
            print(f"warning: shape {i+1} of {shapefile.shape[0]} is invalid")
            continue
        npts += shp_points
        nvalid_shps += 1

    if get_z:
        xyz = np.zeros((npts, 3), dtype=float)
    else:
        xyz = np.zeros((npts, 2), dtype=float)

    shape_pts_l2g = [None] * nvalid_shps
    ptr = 0
    ptr_shp = 0
    for i in range(shapefile.shape[0]):
        try:
            shp_points = np.array(shapefile.iloc[i, :]['geometry'].coords.xy).shape[1]
        except AttributeError:  # nEw
            print(f"warning: shape {i+1} of {shapefile.shape[0]} is invalid")
            continue
        except NotImplementedError:  # nEw
            print(f"warning: shape {i+1} of {shapefile.shape[0]} is invalid")
            continue

        if get_z:
            xyz[ptr:ptr+shp_points] = np.array(shapefile.iloc[i, :]['geometry'].coords)
        else:
            xyz[ptr:ptr+shp_points] = np.array(shapefile.iloc[i, :]['geometry'].coords.xy).T
            # todo, this is more efficient: xyz = np.array(shapefile.iloc[i, :]['geometry'].coords)[:, :2]
        shape_pts_l2g[ptr_shp] = np.array(np.arange(ptr, ptr+shp_points))
        ptr += shp_points
        ptr_shp += 1
    if ptr != npts or ptr_shp != nvalid_shps:
        raise ValueError("number of shapes/points does not match")

    curv = np.empty((npts, ), dtype=float)
    perp = np.empty((npts, ), dtype=float)
    for i, _ in enumerate(shape_pts_l2g):
        line = xyz[shape_pts_l2g[i], :]
        curv[shape_pts_l2g[i]] = curvature(line)
        perp[shape_pts_l2g[i]] = get_perpendicular_angle(line)

    return xyz, shape_pts_l2g, curv, perp


def extract_quad_polygons(input_fname='test.map', output_fname=None):
    """extract quad polygons from a SMS map file and write to a new file"""

    if output_fname is None:
        output_fname = os.path.splitext(input_fname)[0] + '.quad.map'

    with open(output_fname, 'w', encoding='utf-8') as fout:
        lines_buffer = []
        is_patch = False

        with open(input_fname, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                strs = re.split(' +', line.strip())

                if strs[0] != 'POLYGON':
                    fout.write(line)
                else:
                    lines_buffer.append(line)
                    while True:
                        line = f.readline()
                        strs = re.split(' +', line.strip())
                        lines_buffer.append(line)
                        if strs[0] == 'PATCH':
                            is_patch = True
                        elif strs[0] == 'END':
                            if is_patch:
                                for line_buffer in lines_buffer:
                                    fout.write(line_buffer)
                                is_patch = False
                                fout.flush()
                            lines_buffer = []
                            break


def test():
    """common usage scenarios of the SMS_MAP class"""

    # sample 1
    # my_map = SMS_MAP(filename='test_z.map')
    # my_map.get_xyz()
    # my_map.writer('./test.map')

    # sample 2
    # merge_maps(
    #     mapfile_glob_str='/sciclone/schism10/feiye/STOFS3D-v5/Inputs/v14/Parallel/Outputs/'
    #                      'CUDEM_merged_thalwegs_1e6_single_fix_simple_sms_cleaned_32cores/*corrected_thalweg*.map',
    #     merged_fname='/sciclone/schism10/feiye/STOFS3D-v5/Inputs/v14/Parallel/Outputs/'
    #                  'CUDEM_merged_thalwegs_1e6_single_fix_simple_sms_cleaned_32cores/total_corrected_thalwegs.map'
    # )

    # sample 3
    # extract_quad_polygons(input_fname='/sciclone/schism10/feiye/STOFS3D-v7/Inputs/I18c/tvd_polygons.map')

    # sample 4
    # wdir = ('/sciclone/schism10/feiye/STOFS3D-v7/v19_RiverMapper/Outputs/'
    #         'bora_v19.1.v19_ie_v18_3_nwm_clipped_in_cudem_missing_tiles_20-core/')
    wdir = ('/sciclone/schism10/Hgrid_projects/STOFS3D-v8/v20p2s2_RiverMapper/Outputs/'
            'bora_v20p2s2v21.nhdflowline_ms_la_clipped4_4-core/')
    my_map = SMS_MAP(f'{wdir}/total_river_arcs_extra.map')
    write_river_shape_extra(my_map, f'{wdir}/total_river_arcs_extra.shp')


def write_river_shape_extra(sms_map, output_fname):
    '''make a shapefile from a map, with extra attributes'''

    my_gdf = sms_map.to_GeoDataFrame()
    # add new columns of river indices and arc indices
    my_gdf['river_idx'] = -1
    my_gdf['local_arc_idx'] = -1

    river_idx = 0
    local_arc_idx = 0
    # iterate rows in the GeoDataFrame
    for i, row in my_gdf.iterrows():
        z = np.array(row['geometry'].coords)[0, -1]
        z_info = z_decoder(z)
        nrows = z_info[0][0]
        my_gdf.at[i, 'river_idx'] = river_idx
        my_gdf.at[i, 'local_arc_idx'] = local_arc_idx

        local_arc_idx += 1
        if local_arc_idx == nrows:  # all arcs in a river has been processed
            river_idx += 1
            local_arc_idx = 0

    my_gdf.to_file(output_fname)


if __name__ == '__main__':
    test()
