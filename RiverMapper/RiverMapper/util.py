"""
This script provides utility methods used by RiverMapper
"""

import os
import shutil
from pathlib import Path
import numpy as np


cpp_crs = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


def silentremove(filenames):
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

def z_encoder(int_info:np.ndarray):
    '''
    Encode information as 2-digit integers in z's decimal part
    int_info is a NxM integer array,
    where N is the number of nodes along an arc,
    and M is the number of pieces of information.
    Each piece of information is encoded as a two-digit integer (0-99).
    For example, the first column can be the number of cross-channel nodes
    (number of cross-channel divisions + 1).
    A maximum of 6 integers are allowed, i.e., M <= 6
    '''

    if np.any(int_info > 99) or np.any(int_info < 0):
        raise ValueError('int_list must be positive and less than 100')

    # the integer part is the number of integers in the list
    n_info = int_info.shape[1]
    if n_info> 6:
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
    z (float or np.ndarray): A single float or an n x 1 array of floats, each with its integer part between 1 and 6, inclusive.

    Returns:
    list of np.ndarray: A list of 2D integer arrays, each containing the decoded information.

    Raises:
    ValueError: If the integer part of any z in z_input is not between 1 and 6.
    TypeError: If z_input is not a float or a NumPy array of floats.
    '''

    # Convert a single float to a NumPy array
    if isinstance(z, float):
        z_array = np.array([z])
    elif isinstance(z_array, np.ndarray) and z_array.dtype == float:
        z_array = z
    else:
        raise TypeError('z_input must be a float or an n x 1 NumPy array of floats')


    num_digits = z_array.astype(int)
    if np.any((num_digits > 6) | (num_digits <= 0)):
        raise ValueError('The integer part of each z in z_array must be between 1 and 6')

    # Scale z_array to extract the relevant decimal digits and convert to strings
    z_scaled = np.round(z_array * 100**num_digits).astype(int)
    z_scaled_str = np.char.mod('%d', z_scaled)

    # Extract the decimal parts from the scaled strings
    decimal_parts_str = np.array([z_scaled_str[i][1:] for i in range(len(z_scaled_str))])

    # Convert each 2-digit group into an integer
    decoded_arrays = [np.array([int(decimal_parts_str[i][j:j+2]) for j in range(0, len(decimal_parts_str[i]), 2)]) for i in range(len(decimal_parts_str))]

    return decoded_arrays
