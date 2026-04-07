import logging
import sys
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
))

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)
        logger.setLevel(logging.INFO)
    return logger


def log_full(logger, label: str, content: str):
    """Write full content to log file only (DEBUG-level equivalent via file handler)."""
    pass  # disabled — keeping logs minimal
