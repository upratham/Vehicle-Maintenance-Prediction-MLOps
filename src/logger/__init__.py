import logging
import os
import re
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Rotation backups within a single session
MAX_LOG_FILES = 3  # Total log sessions to keep on disk

# Construct log file path
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


def _prune_old_logs(log_dir: str, keep: int) -> None:
    """Delete oldest log sessions so only `keep` sessions remain.

    A "session" is one timestamped base file plus its rotation siblings
    (.log.1, .log.2, …). Sorting is by mtime of the base file.
    """
    try:
        all_files = os.listdir(log_dir)
        # Collect unique base names (strip .log.N rotation suffixes)
        bases: set[str] = set()
        for f in all_files:
            b = re.sub(r'\.log\.\d+$', '.log', f)
            if b.endswith('.log'):
                bases.add(b)

        def _mtime(b: str) -> float:
            p = os.path.join(log_dir, b)
            return os.path.getmtime(p) if os.path.exists(p) else 0.0

        stale = sorted(bases, key=_mtime, reverse=True)[keep:]
        for old_base in stale:
            for f in all_files:
                if re.sub(r'\.log\.\d+$', '.log', f) == old_base:
                    try:
                        os.remove(os.path.join(log_dir, f))
                    except OSError:
                        pass
    except Exception:
        pass


def configure_logger():
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


configure_logger()
_prune_old_logs(log_dir_path, MAX_LOG_FILES)