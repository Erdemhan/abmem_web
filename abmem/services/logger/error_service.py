from ...constants import *
import logging

# Configure the logger for error messages
logger = logging.getLogger('FileLogger')

class ResourceNotFoundError(Exception):
    """
    Custom exception class for handling resource not found errors.
    """

    def __init__(self, resourceName):
        """
        Constructor to initialize the ResourceNotFoundError.

        Args:
            resourceName (str): The name of the resource that was not found.
        """
        # Remove traceback details from the error
        self.__traceback__ = None
        
        # Set the error message using a function to format the resource name
        self.value = resourceNotFoundString(resourceName)
        
        # Print the error message to the console
        print(self.value)
        
        # Log the error using the configured logger
        logger.error(self)
 
    def __str__(self):
        """
        Override the __str__ method to return the error message.

        Returns:
            str: The error message associated with the exception.
        """
        return self.value
