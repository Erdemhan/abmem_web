import sys
# Add the specified directory to the system path to allow imports from that location
sys.path.append("D:/Projeler/abm/abmem_project/test")

# Import necessary classes and enums from the models module
from ...models.market import Market
from ...models.enums import MarketStrategy, MarketState

def create(sim, strategy: MarketStrategy, lowerBound: int, upperBound: int) -> Market:
    # Retrieve the proxy from the simulation object
    proxy = sim.proxy
    
    # Create a new Market object with the specified strategy, state, and bounds
    market = Market(
        strategy= strategy, 
        state= MarketState.CREATED, 
        lowerBidBound= lowerBound, 
        upperBidBound= upperBound, 
        simulation= sim,
        proxy = proxy
    )
    
    # Save the newly created market to the database or persistent storage
    market.save()
    
    # Return the created market object
    return market
