from ...models import Period

def create(market, num: int, demand: int) -> Period:
    # Create a new Period object with the provided market, period number, and demand.
    # Initialize metDemand, marketVolume, and ptf with default values of -1.
    period = Period(
        market=market, 
        periodNumber=num, 
        demand=demand, 
        metDemand=-1, 
        marketVolume=-1, 
        ptf=-1
    )
    
    # Save the newly created Period object to the database.
    period.save()
    
    # Return the saved Period object.
    return period
