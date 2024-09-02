import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")
from ...models.models import Resource
from ...models.enums import *
from decimal import Decimal

def create(name: str, energyType: EnergyType, fuelCost: Decimal, emission = Decimal) -> (Resource, bool):
    # Check if a resource with the given name already exists in the database
    resource = Resource.objects.filter(name=name).first()

    if resource:
        # If the resource exists, set it to None
        resource = None
    else:
        # If the resource does not exist, create a new one
        if energyType == EnergyType.RENEWABLE:
            # For renewable energy types, set fuel cost and emission to 0
            fuelCost, emission = 0, 0
        
        # Create a new Resource object with the provided values
        resource = Resource(name=name, energyType=EnergyType[energyType], fuelCost=fuelCost, emission=emission)
        
        # Save the new Resource object to the database
        resource.save()
        
    # Return the created resource object (or None if it already existed)
    return resource
