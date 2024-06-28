import django
django.setup()

from ...models.simulation import Simulation
from ...models.enums import *
from ...services.file_reader import reader_service as ReaderService
from ...services.market import market_factory as MarketFactory
from ...services.market import market_service as MarketService
from ...services.visualization import visualization_service as VisualizationService
from ...constants import *
import timeit


def init(simulation: Simulation):
    simulation.state = SimulationState.INITIALIZED
    simulation.currentPeriod = 1
    if simulation.mode == SimulationMode.ONLYRESULT:
        # Placeholder for future development
        pass
    elif simulation.mode == SimulationMode.PERIODBYPERIOD:
        # Placeholder for future development
        pass
    simulation.save()


def readMarketData() -> dict:
    return ReaderService.readData(path= MARKET_DATA_PATH, key= MARKET_DATA_KEY)

import time
def run(simulation: Simulation) -> bool:
    isOk = True
    start = timeit.default_timer()
    tOffers = []
    try:
        if simulation.currentPeriod == -1 :
            simulation.currentPeriod += 1
        if simulation.market.state == MarketState.CREATED:
            MarketService.init(simulation.market)
            print("market inited")
        simulation.save()
        while simulation.currentPeriod < simulation.periodNumber:
            if not simulation.currentPeriod < 1:
                if simulation.mode == SimulationMode.ONLYRESULT:
                    print("mode onlyresult" , simulation.mode)
                    pass
                elif simulation.mode == SimulationMode.PERIODBYPERIOD:
                    while simulation.market.state == MarketState.PERIODEND:
                        simulation.market.refresh_from_db()
                        time.sleep(0.5)
            simulation.currentPeriod += 1
            simulation.save()
            print("market run start")
            tOffers.append(MarketService.run(simulation.market))
            simulation.state = SimulationState.STARTED
            simulation.save()

        print("sim viz")
        VisualizationService.visulaizeSimulation(simulation.market.period_set.all())
        #print("T: " , timeit.default_timer() - start)
        simulation.state = SimulationState.FINISHED
        simulation.save()
        return tOffers
    except Exception as e:
        simulation.state = SimulationState.CANCELED
        simulation.save()
        return 0


