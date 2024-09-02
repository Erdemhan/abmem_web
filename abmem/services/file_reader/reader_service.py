import sys
import pandas as pd
from ...models.enums import *
import yaml
from ...constants.paths import DATAPATH, SIMULATION_DATA_PATH
import logging

# TODO: Add YAML schema validator/checker to ensure consistency and validate data integrity
logger = logging.getLogger('FileLogger')

def openFile(path: str):
    """ 
    Opens and reads a YAML file from the given path.
    
    Args:
        path (str): The relative path to the YAML file.
        
    Returns:
        dict: Parsed YAML data as a dictionary.
    """
    try:
        with open(DATAPATH + path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError as error:
        # Log the error and exit if the file is not found
        print(path, "could not find in folder data")
        logger.error(error)
        sys.exit(1)

def readData(path: str, key: str) -> dict:
    """
    Reads data from a YAML file and retrieves a specific key's value.
    
    Args:
        path (str): The relative path to the YAML file.
        key (str): The key to retrieve data from the YAML file.
        
    Returns:
        dict: Data associated with the provided key.
    """
    try:
        data = openFile(path)
        return data[key]
    except KeyError as error:
        # Log the error and exit if the key is not found in the data
        print("Key: '", key, "' could not find in ", path)
        logger.error(error)
        sys.exit(1)

def readExcel(path: str, column: str) -> list:
    """
    Reads a specific column from an Excel file and returns its data as a list.
    
    Args:
        path (str): The relative path to the Excel file.
        column (str): The column name to read from the Excel file.
        
    Returns:
        list: List of data from the specified column.
    """
    try:
        df = pd.read_excel(DATAPATH + path)
        data = df[column].tolist()
        return data
    except Exception as e:
        # Log the error and exit if any exception occurs
        print(e)
        logger.error(e)
        sys.exit(1)

def readExcel(path: str, columns: list, map: list = []) -> dict:
    """
    Reads specified columns from an Excel file and returns their data as a dictionary.
    
    Args:
        path (str): The relative path to the Excel file.
        columns (list): List of column names to read from the Excel file.
        map (list, optional): List of keys to map the columns to. Defaults to using column names as keys.
        
    Returns:
        dict: Dictionary where keys are mapped to column data lists.
    """
    if len(map) == 0:
        # If no mapping is provided, use the column names as keys
        map = columns
    data = {}
    try:
        df = pd.read_excel(DATAPATH + path)
        for column, mapName in zip(columns, map):
            data[mapName] = df[column].tolist()
        return data
    except Exception as e:
        # Log the error and exit if any exception occurs
        print(e)
        logger.error(e)
        sys.exit(1)
