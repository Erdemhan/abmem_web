import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")
from ...models.simulation import Simulation
from ...models.enums import SimulationMode,SimulationState,PeriodType

def create(name: str, mode: SimulationMode, periodType: PeriodType, periodNumber: int, proxy: bool) -> Simulation:
    pass
    sim = Simulation(name = name,mode = mode, periodType = periodType,state=SimulationState.CREATED, periodNumber = periodNumber, currentPeriod = -1, proxy=proxy)
    sim.save()
    return sim