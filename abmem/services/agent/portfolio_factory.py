import sys
import os

# Adding the project directory to the system path
sys.path.append("D:/Projeler/abm/abmem_project/test")

from ...models.models import Portfolio, Plant, Resource, Agent
from ...models.enums import *
from ...constants import *
from ...services.logger.error_service import ResourceNotFoundError
import logging

# Setting up a logger named 'FileLogger'
logger = logging.getLogger('FileLogger')

def create(agent: Agent, plantsData: dict):
    # Create a new Portfolio instance associated with the given agent
    portfolio = Portfolio(agent=agent)
    portfolio.save()
    
    # Iterate through the plantsData dictionary to create Plant instances
    for plant in plantsData:
        createPlant(portfolio, plant)
    
    # Return the created Portfolio instance
    return portfolio


def createPlant(portfolio: Portfolio, plantData: dict):
    # Retrieve the resource associated with the plantData by filtering by name
    resource = Resource.objects.filter(name=plantData[PLANT_RESOURCE_KEY]).first()
    
    # Raise an error if the resource is not found
    if not resource:
        raise ResourceNotFoundError(plantData[PLANT_RESOURCE_KEY])
    
    # Create and save a new Plant instance with the given portfolio, resource, and capacity
    plant = Plant(portfolio=portfolio, resource=resource, capacity=plantData[PLANT_CAPACITY_KEY])
    plant.save()
    
    # Return the created Plant instance
    return plant

def createPlant(portfolio: Portfolio, resource: Resource, capacity):
    # Create and save a new Plant instance with the given portfolio, resource, and capacity
    plant = Plant(portfolio=portfolio, resource=resource, capacity=capacity)
    plant.save()
    
    # Return the created Plant instance
    return plant
