from ..file_reader import reader_service as ReaderService
from ..simulation import simulation_factory as SimulationFactory
from ..simulation import simulation_service as SimulationService
from ...models import Simulation, Resource
from . import resource_service as ResourceService
from ...constants import *
from ... import bootstrap
from django.db import connections
import logging
import multiprocessing

# Set up logging for the file logger
logger = logging.getLogger("FileLogger")


def start(simulation: Simulation) -> int:
    """Start a non-proxy simulation in a separate spawned process.

    The child process sets up Django itself and loads the simulation by id
    (see ``bootstrap.run_simulation_process``). Proxy simulations are
    templates and are never started.

    Args:
        simulation: Saved simulation to run.

    Returns:
        The id of the simulation.
    """
    if simulation.proxy == False:
        # Inherited open connections must not be shared with the child.
        connections.close_all()
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=bootstrap.run_simulation_process, args=(simulation.id,)
        )
        process.start()
    return simulation.id


def readStarterData() -> (dict, dict):
    # Read and return simulation and resource data from predefined paths
    return (
        ReaderService.readData(path=SIMULATION_DATA_PATH, key=SIMULATION_DATA_KEY),
        ReaderService.readData(path=RESOURCES_DATA_PATH, key=RESOURCES_DATA_KEY),
    )


def createSimulation(simData: dict) -> Simulation:
    # Create a new Simulation object using the provided data
    return SimulationFactory.create(
        name=simData[SIMULATION_NAME_KEY],
        mode=simData[SIMULATION_MODE_KEY],
        periodType=simData[SIMULATION_PERIOD_TYPE_KEY],
        periodNumber=simData[SIMULATION_PERIOD_NUMBER_KEY],
    )


def initSimulation(simulation: Simulation) -> None:
    # Initialize the simulation using the SimulationService
    SimulationService.init(simulation)


def checkResources(resourceData: dict) -> [Resource]:
    # Create and return a list of resources based on the provided data
    return ResourceService.createFromData(resourceData)


# Uncomment the following line to start the simulation
# start()
