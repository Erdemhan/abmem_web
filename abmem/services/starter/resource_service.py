import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")
from ...models.models import Resource
from ...services.starter import resource_factory as ResourceFactory
from ...constants import *

def createFromData(resourcesData: dict) -> [Resource]:
    # Initialize an empty list to store successfully created resources
    created = []
    
    # Iterate through each resource in the provided data dictionary
    for resource in resourcesData:
        # Attempt to create a new Resource object using the ResourceFactory
        resource = ResourceFactory.create(
            name=resource[RESOURCES_NAME_KEY],
            energyType=resource[RESOURCES_ENERGY_TYPE_KEY],
            fuelCost=resource[RESOURCES_FUELCOST_KEY],
            emission=resource[RESOURCES_EMISSION_KEY]
        )
        
        # If the resource was successfully created, add it to the created list
        if resource:
            created.append(resource)
            # Print a message indicating the successful creation of the resource
            print(resourceCreatedString(name=resource.name, id=resource.id))
    
    # Print a report summarizing the resource creation process
    print(resourcesCreationReportString(dataLenght=len(resourcesData), creationLength=len(created)))
    
    # Return the list of successfully created resources
    return created
