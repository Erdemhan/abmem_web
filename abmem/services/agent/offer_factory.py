import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")
from ...models.agent import Agent
from ...models.market import Market
from ...models.models import Period,Resource,Offer
from ...models.enums import AgentState,AgentType
from decimal import Decimal

def create(agent: Agent, resource: Resource, amount: int, offerPrice: Decimal):
    period = agent.market.period_set.order_by('periodNumber').last()
    offer = Offer(period=period, agent=agent, resource=resource, amount=amount, offerPrice=offerPrice, acceptance=False, acceptancePrice = 0, acceptanceAmount=0)
    offer.save()
    return  offer

