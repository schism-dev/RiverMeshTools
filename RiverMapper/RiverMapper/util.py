"""
This script provides utility methods used by RiverMapper
"""

import os
import shutil
from pathlib import Path


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