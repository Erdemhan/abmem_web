from ...models.agent import Agent
from ...models.market import Market
from ...models.enums import AgentState,AgentType

def create(market: Market, budget: int,type):
    type = AgentType.get(type)
    agent = Agent(market= market, state= AgentState.CREATED, budget= budget, type= type)
    agent.save()
    return  agent