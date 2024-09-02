import django
django.setup()  # Initialize Django to make sure the ORM and models are available

# Import necessary models, enums, and services
from ...models.simulation import Simulation
from ...models.enums import *
from ...services.file_reader import reader_service as ReaderService
from ...services.market import market_factory as MarketFactory
from ...services.market import market_service as MarketService
from ...services.visualization import visualization_service as VisualizationService
from ...constants import *
import timeit

def init(simulation: Simulation):
    """
    Initialize the simulation by setting its state to INITIALIZED and starting the first period.

    Args:
        simulation (Simulation): The simulation instance to be initialized.
    """
    simulation.state = SimulationState.INITIALIZED  # Set the state to INITIALIZED
    simulation.currentPeriod = 1  # Set the current period to 1 (start of the simulation)

    if simulation.mode == SimulationMode.ONLYRESULT:
        # Placeholder for handling ONLYRESULT mode in future development
        pass
    elif simulation.mode == SimulationMode.PERIODBYPERIOD:
        # Placeholder for handling PERIODBYPERIOD mode in future development
        pass

    simulation.save()  # Save the updated simulation state to the database


def readMarketData() -> dict:
    """
    Read and return market data from the specified file.

    Returns:
        dict: A dictionary containing market data read from the file.
    """
    return ReaderService.readData(path=MARKET_DATA_PATH, key=MARKET_DATA_KEY)


import time
def run(simulation: Simulation) -> bool:
    """
    Run the simulation through its defined periods, handling market operations and visualization.

    Args:
        simulation (Simulation): The simulation instance to be run.

    Returns:
        bool: Returns True if the simulation runs successfully, otherwise False.
    """
    isOk = True
    start = timeit.default_timer()  # Start the timer to measure execution time
    tOffers = []  # Initialize a list to store offers

    try:
        if simulation.currentPeriod == -1:
            simulation.currentPeriod += 1  # Start the simulation if it hasn't started yet

        if simulation.market.state == MarketState.CREATED:
            MarketService.init(simulation.market)  # Initialize the market if it's in the CREATED state
            print("Market initialized")
        
        simulation.save()  # Save the updated simulation state

        # Main loop for running through all periods in the simulation
        while simulation.currentPeriod < simulation.periodNumber:
            if simulation.currentPeriod >= 1:
                if simulation.mode == SimulationMode.ONLYRESULT:
                    # Placeholder for handling ONLYRESULT mode
                    print("Mode ONLYRESULT", simulation.mode)
                    pass
                elif simulation.mode == SimulationMode.PERIODBYPERIOD:
                    # Wait for the previous period to end before proceeding
                    while simulation.market.state == MarketState.PERIODEND:
                        simulation.market.refresh_from_db()  # Refresh market state from the database
                        time.sleep(0.5)  # Sleep briefly to avoid busy-waiting
            
            if simulation.currentPeriod > 0 and (simulation.currentPeriod % 24) == 0:
                simulation.day += 1  # Increment the day after every 24 periods

            simulation.currentPeriod += 1  # Move to the next period
            simulation.save()  # Save the updated simulation state

            print("Market run start")
            tOffers.append(MarketService.run(simulation.market))  # Run the market service for the current period

            simulation.state = SimulationState.STARTED  # Update the simulation state to STARTED
            simulation.save()  # Save the updated simulation state

        print("Simulation visualization")
        VisualizationService.visualizeSimulation(simulation.market.period_set.all())  # Visualize the entire simulation

        simulation.state = SimulationState.FINISHED  # Mark the simulation as FINISHED
        simulation.save()  # Save the final state of the simulation
        return tOffers  # Return the list of offers generated during the simulation

    except Exception as e:
        simulation.state = SimulationState.CANCELED  # If an error occurs, mark the simulation as CANCELED
        print(e)  # Print the error for debugging
        simulation.save()  # Save the simulation state as CANCELED
        return False  # Return False to indicate that the simulation failed
