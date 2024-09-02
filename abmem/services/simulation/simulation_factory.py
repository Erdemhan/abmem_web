import sys
# Add the project path to the system path to allow for module imports
sys.path.append("D:/Projeler/abm/abmem_project/test")

# Import necessary models and enums
from ...models.simulation import Simulation
from ...models.enums import SimulationMode, SimulationState, PeriodType

def create(name: str, mode: SimulationMode, periodType: PeriodType, periodNumber: int, proxy: bool) -> Simulation:
    """
    Create and initialize a new Simulation object.

    Args:
        name (str): The name of the simulation.
        mode (SimulationMode): The mode of the simulation (e.g., training, testing).
        periodType (PeriodType): The type of period for the simulation (e.g., hourly, daily).
        periodNumber (int): The total number of periods in the simulation.
        proxy (bool): Whether to use a proxy in the simulation.

    Returns:
        Simulation: The newly created and saved Simulation object.
    """
    # Create a new Simulation object with the provided parameters and initial state
    sim = Simulation(
        name=name,
        mode=mode,
        periodType=periodType,
        state=SimulationState.CREATED,  # Set the initial state to CREATED
        periodNumber=periodNumber,
        currentPeriod=-1,  # Initialize currentPeriod to -1 (no period has started yet)
        proxy=proxy,
        day=0  # Start the simulation with day 0
    )
    
    # Save the new Simulation object to the database
    sim.save()
    
    # Return the saved Simulation object
    return sim
