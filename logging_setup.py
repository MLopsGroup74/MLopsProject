"""Centralized logging configuration using Loguru.

Configures logging to output WARNING+ to stdout and all logs to a rotating file.
"""

import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="WARNING")
logger.add("project_log.log", level="DEBUG", rotation="100 MB")
