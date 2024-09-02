import logging
import logging.handlers
import os
from datetime import datetime
from services.logger import error_service
import sys, traceback

def setupFileLogger():
    # Define the log message format
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    # Generate the log file name based on the current date and time
    fileName = datetime.now().strftime("%d-%m-%Y--%H-%M.log")

    # Set up the directory path for storing log files
    logs_path = os.path.join(os.getcwd()+'/test', 'logs')
    
    # Create the logs directory if it does not exist
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    # Create the full path for the log file
    path = os.path.join(logs_path, fileName)

    # Define a custom exception handler that logs uncaught exceptions
    def exc_handler(exctype, value, tb):
        # Log only if the exception type is not ResourceNotFoundError
        if exctype != error_service.ResourceNotFoundError:
            print("Unexpected Exception! Please see " + path + " for more.")
            logger.exception(''.join(traceback.format_exception(exctype, value, tb)))
    
    # Set the custom exception handler as the default handler for uncaught exceptions
    sys.excepthook = exc_handler

    # Create a file handler for logging
    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)

    # Create a logger with the name 'FileLogger'
    logger = logging.getLogger('FileLogger')
    
    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    
    # Add the file handler to the logger
    logger.addHandler(handler)
    
    # Return the configured logger
    return logger
