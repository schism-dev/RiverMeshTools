"""
This script provides utility methods used by RiverMapper
"""

import os
import shutil
from pathlib import Path

import numpy as np
import geopandas as gpd



# The equirectangular projection, a simple map projection that maps (lat, lon) to (x, y) coordinates.
CPP_CRS = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


def silentremove(filenames):
    '''Remove files or directories without raising exceptions if they do not exist.'''
    if isinstance(filenames, str):
        filenames = [filenames]
    elif isinstance(filenames, Path):
        filenames = [str(filenames)]

    for filename in filenames:
        try:
            os.remove(filename)
        except IsADirectoryError:
            shutil.rmtree(filename)
        except FileNotFoundError:
            pass  # missing_ok


def z_encoder(int_info: np.ndarray):
    '''
    Encode information as 2-digit integers in z's decimal part,

    "int_info" is a NxM integer array,
    where N is the number of nodes along an arc,
    and M is the number of pieces of information.

    Each piece of information is encoded as a two-digit integer (0-99).
    For example, the first column can be the number of cross-channel nodes
    (number of cross-channel divisions + 1).
    A maximum of 6 integers are allowed, i.e., M <= 6

    The integer part of z records the number of pieces of information.
    '''

    if np.any(int_info > 99) or np.any(int_info < 0):
        raise ValueError('int_list must be positive and less than 100')

    # the integer part is the number of integers in the list
    n_info = int_info.shape[1]
    if n_info > 6:
        raise ValueError('int_info must not be more than 6 columns')

    # convert int_info to string
    str_info = np.apply_along_axis(''.join, axis=1, arr=np.char.zfill(int_info.astype(str), 2))
    # add prefix to each row, which is the number of integers in the list
    str_info = np.core.defchararray.add(f'{n_info}.', str_info)

    encoded_float = str_info.astype(float)

    return encoded_float


def z_decoder(z):
    '''
    Decode information encoded as 2-digit integers in the decimal parts of numbers in z_input.
    The integer part of each number in z_input specifies the number of 2-digit groups to decode.
    Returns a list of 2D integer arrays, each containing the decoded information for corresponding z.

    Parameters:
    z (float or np.ndarray): A single float or an n x 1 array of floats,
    each with its integer part between 1 and 6, inclusive.

    Returns:
    list of np.ndarray: A list of 2D integer arrays, each containing the decoded information.

    Raises:
    ValueError: If the integer part of any z in z_input is not between 1 and 6.
    TypeError: If z_input is not a float or a NumPy array of floats.
    '''

    # Convert a single float to a NumPy array
    if isinstance(z, float):
        z_array = np.array([z])
    elif isinstance(z, np.ndarray) and z.dtype == float:
        z_array = z
    else:
        raise TypeError('z_input must be a float or an n x 1 NumPy array of floats')

    num_digits = z_array.astype(int)
    if np.any((num_digits > 6) | (num_digits <= 0)):
        raise ValueError('The integer part of each z in z_array must be between 1 and 6')

    # Scale z_array to extract the relevant decimal digits and convert to strings
    z_scaled = np.round(z_array * 100**num_digits).astype(int)
    z_scaled_str = np.char.mod('%d', z_scaled)

    # Extract the decimal parts of z_input from z_scaled_str
    decimal_parts_str = np.array([z_scaled_str[i][1:] for i in range(len(z_scaled_str))])

    # Convert each 2-digit group into an integer
    decoded_arrays = []
    # loop through each row, i.e., each number
    for i, decimal_part_str in enumerate(decimal_parts_str):
        # loop through each 2-digit group in the decimal part
        decoded_arrays.append(np.array([int(decimal_part_str[j:j+2]) for j in range(0, len(decimal_part_str), 2)]))

    return decoded_arrays


def reproject_tif(tif_file, output_file, dst_crs='epsg:4326'):
    """
    Reproject a GeoTIFF file to a specified coordinate reference system (CRS).

    Parameters:
    tif_file (str): Path to the input GeoTIFF file.
    out_file (str): Path to the output reprojected GeoTIFF file.
    crs (str): The target coordinate reference system (CRS) in WKT format or EPSG code.

    Returns:
    None
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(tif_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )


def reproject_shpfile(shp_file, output_file, dst_crs='epsg:4326'):
    """
    Reproject a shapefile to a specified coordinate reference system (CRS).

    Parameters:
    shp_file (str): Path to the input shapefile.
    output_file (str): Path to the output reprojected shapefile.
    dst_crs (str): The target coordinate reference system (CRS) in WKT format or EPSG code.

    Returns:
    None
    """

    gdf = gpd.read_file(shp_file)
    gdf.to_crs(dst_crs).to_file(output_file)


if __name__ == '__main__':
    # Example usage
    # z = np.array([1.23, 2.34, 3.45])
    # decoded = z_decoder(z)
    # print(decoded)

    # Example usage of reproject_*
    reproject_shpfile(
        '/sciclone/home/feiye/Hgrid_projects/mattwig_rivmap_ontario/sodus_rivmap_1e5.shp',
        '/sciclone/home/feiye/Hgrid_projects/mattwig_rivmap_ontario/sodus_rivmap_1e5.ll.shp', dst_crs='epsg:4326')

    reproject_tif(
        '/sciclone/home/feiye/Hgrid_projects/mattwig_rivmap_ontario/sodus_rivmap.tif',
        '/sciclone/home/feiye/Hgrid_projects/mattwig_rivmap_ontario/sodus_rivmap.ll.tif', dst_crs='epsg:4326')

    print('Done')
