"""Configure serial logging; MPI callers supply their rank explicitly."""
import os
import logging

logger = logging.getLogger('my_application_logger')
logger.setLevel(logging.DEBUG)
_handlers = []


def configure_logger(rank=0):
    """Set the log filename and message prefix for a serial or MPI rank."""
    os.makedirs("Logs", exist_ok=True)
    formatter = logging.Formatter(
        fmt=f'%(asctime)s - %(levelname)s - Rank {rank} - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(f"Logs/RiverMapper_log_rank_{rank}.log"),
    ]
    for handler in _handlers:
        logger.removeHandler(handler)
        handler.close()
    _handlers.clear()
    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    _handlers.extend(handlers)


configure_logger()

# Set specific log levels for third-party libraries to suppress debug and info logs
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('env').setLevel(logging.WARNING)
logging.getLogger('collection').setLevel(logging.WARNING)
logging.getLogger('file').setLevel(logging.WARNING)
logging.getLogger('geodataframe').setLevel(logging.WARNING)
