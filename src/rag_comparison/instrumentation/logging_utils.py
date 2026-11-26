"""Local logging helpers for the comparison framework."""
from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime

from rag_comparison.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both the console and a local log file."""
    log_dir = settings.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rag_comparison_{datetime.now():%Y%m%d}.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=settings.LOG_FILE_MAX_BYTES,
        backupCount=settings.LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
