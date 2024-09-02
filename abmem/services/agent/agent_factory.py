from ...models.agent import Agent
from ...models.market import Market
from ...models.enums import AgentState,AgentType

def create(market: Market, name: str, budget: int, type: AgentType, proxy: bool):
    # Create a new Agent instance with the given parameters
    agent = Agent(market=market, name=name, state=AgentState.CREATED, budget=budget, type=type, proxy=proxy)
    
    # Save the newly created Agent instance to the database
    agent.save()
    
    # Return the saved Agent instance
    return agent
