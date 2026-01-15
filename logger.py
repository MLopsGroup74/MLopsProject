import sys
from loguru import logger

# Remove the default configuration
logger.remove()

#Only print WARNING or higher (WARNING, ERROR, CRITICAL) to terminal
logger.add(sys.stdout, level="WARNING")

#Save all logs to project_log.log file, the file will rotate after it reached 100MB
logger.add("project_log.log", level="DEBUG", rotation="100 MB")


#What the different log files mean
logger.debug("Used for debugging your code.")          
logger.info("Informative messages from your code.")     
logger.warning("Something to be aware of.")            
logger.error("There's been a mistake.")                
logger.critical("Something is terribly wrong.")         
