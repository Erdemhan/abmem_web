from ...models.agent import Agent
from ...models.market import Market
from ...models.enums import AgentState,AgentType

def create(market: Market,name: str, budget: int,type: AgentType, proxy: bool):
    agent = Agent(market= market,name=name, state= AgentState.CREATED, budget= budget, type= type, proxy=proxy)
    agent.save()
    return  agent