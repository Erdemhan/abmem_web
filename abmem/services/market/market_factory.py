# Add the specified directory to the system path to allow imports from that location

# Import necessary classes and enums from the models module
from ...models.market import Market
from ...models.enums import MarketStrategy, MarketState

def create(
    sim,
    strategy: MarketStrategy,
    lowerBound: int,
    upperBound: int,
    clearing_algorithm: str = "MERIT_ORDER",
) -> Market:
    # Retrieve the proxy from the simulation object
    proxy = sim.proxy
    
    # Create a new Market object with the specified strategy, state, and bounds
    market = Market(
        strategy= strategy, 
        state= MarketState.CREATED, 
        lowerBidBound= lowerBound, 
        upperBidBound= upperBound, 
        clearing_algorithm=clearing_algorithm,
        simulation= sim,
        proxy = proxy
    )
    
    # Save the newly created market to the database or persistent storage
    market.save()
    
    # Return the created market object
    return market
