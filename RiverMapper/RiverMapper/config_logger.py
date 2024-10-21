# config_logger.py
import os
import logging
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Ensure the logs directory exists
os.makedirs("Logs", exist_ok=True)

# Create a logger
logger = logging.getLogger('my_application_logger')
logger.setLevel(logging.DEBUG)  # Set the root logger to the lowest level

# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a file handler with a file name that includes the rank and module
file_handler = logging.FileHandler(f"Logs/RiverMapper_log_rank_{rank}.log")
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter(
    fmt=f'%(asctime)s - %(levelname)s - Rank {rank} - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Set specific log levels for third-party libraries to suppress debug and info logs
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('env').setLevel(logging.WARNING)
logging.getLogger('collection').setLevel(logging.WARNING)
logging.getLogger('file').setLevel(logging.WARNING)
logging.getLogger('geodataframe').setLevel(logging.WARNING)
