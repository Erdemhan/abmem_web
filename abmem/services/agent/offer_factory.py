import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")

from ...models.agent import Agent
from ...models.market import Market
from ...models.models import Period, Resource, Offer
from ...models.enums import AgentState, AgentType
from decimal import Decimal

def create(agent: Agent, resource: Resource, amount: int, offerPrice: Decimal):
    # Retrieve the last period for the given agent's market
    period = agent.market.period_set.order_by('periodNumber').last()
    
    # Create a new Offer instance with the provided parameters and default acceptance values
    offer = Offer(
        period=period,
        agent=agent,
        resource=resource,
        amount=amount,
        offerPrice=offerPrice,
        acceptance=False,
        acceptancePrice=0,
        acceptanceAmount=0
    )
    
    # Save the newly created Offer instance to the database
    offer.save()
    
    # Return the saved Offer instance
    return offer
