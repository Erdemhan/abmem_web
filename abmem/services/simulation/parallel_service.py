import sys
from ...models.enums import MarketState, MarketStrategy
from ...services.market import period_factory as PeriodFactory
from ...services.visualization import visualization_service as VisualizationService
from ...models import Market, Period, Offer, Agent
from decimal import Decimal
from ...services.agent import agent_service as AgentService
from ...services.agent import agent_factory as AgentFactory
from ...services.file_reader import reader_service as ReaderService
from ...constants import *
import multiprocessing
import os
import timeit
import django

def init_worker():
    """
    Initialize Django in the child process.
    This function is called in each child process before running tasks.
    It's necessary to ensure Django is properly set up within child processes.
    """
    django.setup()

def startPool(agents: [Agent]) -> [Agent]:
    """
    Start a multiprocessing pool to execute the AgentService.run function for each agent.
    
    Args:
        agents ([Agent]): A list of Agent objects to be processed.
    
    Returns:
        [Agent]: A list of results returned by the AgentService.run function for each agent.
    """
    # Create a multiprocessing pool and map the AgentService.run function to the agents.
    with multiprocessing.Pool() as pool:
        offers = pool.map(AgentService.run, agents)
    
    # Ensure the pool has been joined and terminated properly.
    pool.join()
    pool.terminate()

    # Return the list of offers produced by the agents.
    return offers
