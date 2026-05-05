import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from from_root import from_root

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, LOG_FILE)


def configure_logger():
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    fh = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)


configure_logger()
