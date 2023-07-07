"""
This script provides utility methods used by RiverMapper
"""


import os
import shutil
from pathlib import Path


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