from ..file_reader import reader_service as ReaderService
from ..simulation import simulation_factory as SimulationFactory
from ..simulation import simulation_service as SimulationService
from ...models import Simulation, Resource
from . import resource_service as ResourceService
from ...constants import *
import django
import logging

# Set up logging for the file logger
logger = logging.getLogger('FileLogger')

def start(simulation: Simulation) -> None:
    # Check if the simulation is not a proxy
    if simulation.proxy == False:
        # Set up Django environment
        django.setup()
        import multiprocessing
        # Create a multiprocessing process to run the simulation asynchronously
        process = multiprocessing.Process(target=SimulationService.run, args=(simulation,))
        process.start()
    # Return the simulation ID
    return simulation.id

def readStarterData() -> (dict, dict):
    # Read and return simulation and resource data from predefined paths
    return (
        ReaderService.readData(path=SIMULATION_DATA_PATH, key=SIMULATION_DATA_KEY),
        ReaderService.readData(path=RESOURCES_DATA_PATH, key=RESOURCES_DATA_KEY)
    )

def createSimulation(simData: dict) -> Simulation:
    # Create a new Simulation object using the provided data
    return SimulationFactory.create(
        name=simData[SIMULATION_NAME_KEY],
        mode=simData[SIMULATION_MODE_KEY],
        periodType=simData[SIMULATION_PERIOD_TYPE_KEY],
        periodNumber=simData[SIMULATION_PERIOD_NUMBER_KEY]
    )

def initSimulation(simulation: Simulation) -> None:
    # Initialize the simulation using the SimulationService
    SimulationService.init(simulation)

def checkResources(resourceData: dict) -> [Resource]:
    # Create and return a list of resources based on the provided data
    return ResourceService.createFromData(resourceData)

# Uncomment the following line to start the simulation
# start()
