import sys
from loguru import logger

# Remove the default configuration
logger.remove()

#Only print WARNING or higher (WARNING, ERROR, CRITICAL) to terminal
logger.add(sys.stdout, level="WARNING")

#Save all logs to project_log.log file, the file will rotate after it reached 100MB
logger.add("project_log.log", level="DEBUG", rotation="100 MB")


