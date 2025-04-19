import django
django.setup()  # Initialize Django to make sure the ORM and models are available

# Import necessary models, enums, and services
from ...models.simulation import Simulation
from ...models.enums import *
from ...services.file_reader import reader_service as ReaderService
from ...services.market import market_service as MarketService
from ...constants import *
import timeit
import json
from collections import defaultdict
import os

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
            market_result = MarketService.run(simulation.market)
            
            # Convert numpy array to list if necessary
            if hasattr(market_result, 'tolist'):
                market_result = market_result.tolist()
            
            if isinstance(market_result, list):
                tOffers.extend(market_result)
                print(f"Added {len(market_result)} offers to tOffers")
            else:
                tOffers.append(market_result)
                print("Added single offer to tOffers")

            simulation.state = SimulationState.STARTED
            simulation.save()

        print("Simulation visualization")
        print(f"Total offers collected: {len(tOffers)}")
        print(f"First offer type: {type(tOffers[0]) if tOffers else 'No offers'}")
        
        # Sadece geçerli Offer nesnelerini filtrele
        valid_offers = [offer for offer in tOffers if hasattr(offer, 'period')]
        print(f"Valid offers found: {len(valid_offers)}")
        
        if len(valid_offers) == 0 and len(tOffers) > 0:
            print(f"Sample offer attributes: {dir(tOffers[0])}")
        
        # Organize offers by period and agent
        offers_by_period_agent = defaultdict(lambda: defaultdict(list))
        for offer in valid_offers:
            period_num = offer.period.periodNumber
            agent_name = offer.agent.name
            
            # Her periyot için MCP'yi bir kez ekleyelim
            if not offers_by_period_agent[period_num].get('market_price'):
                offers_by_period_agent[period_num]['market_price'] = str(offer.period.ptf)
            
            offers_by_period_agent[period_num][agent_name].append({
                'id': offer.id,
                'agent': offer.agent.name,
                'resource': offer.resource.name,
                'amount': offer.amount,
                'offerPrice': str(offer.offerPrice),
                'acceptance': offer.acceptance,
                'acceptancePrice': str(offer.acceptancePrice),
                'acceptanceAmount': offer.acceptanceAmount,
                'budget': str(offer.agent.budget)
            })
        
        print(f"Periods collected: {list(offers_by_period_agent.keys())}")
        
        # Convert to regular dict for JSON serialization
        output_data = {
            'simulation_id': simulation.id,
            'total_periods': simulation.periodNumber,
            'offers_by_period': {
                period: dict(agents) 
                for period, agents in offers_by_period_agent.items()
            }
        }
        
        # Create sim_data directory if it doesn't exist
        sim_data_dir = os.path.join('abm_ddpg', 'sim_data')
        os.makedirs(sim_data_dir, exist_ok=True)
        
        # Save to JSON file in sim_data directory
        output_filename = os.path.join(sim_data_dir, f'simulation_{simulation.id}_offers.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"Offers saved to {output_filename}")

        simulation.state = SimulationState.FINISHED
        simulation.save()
        return tOffers

    except Exception as e:
        simulation.state = SimulationState.CANCELED  # If an error occurs, mark the simulation as CANCELED
        print(e)  # Print the error for debugging
        simulation.save()  # Save the simulation state as CANCELED
        return False  # Return False to indicate that the simulation failed
