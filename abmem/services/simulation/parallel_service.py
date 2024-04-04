import sys
from ...models.enums import MarketState,MarketStrategy
from ...services.market import period_factory as PeriodFactory
from ...services.visualization import visualization_service as VisualizationService
from ...models import Market,Period,Offer,Agent
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
    This function will be called in each child process before running tasks.
    It ensures that Django is properly set up within child processes.
    """
    django.setup()  # Initialize Django in the child process

def startPool(agents: [Agent]) -> [Agent]:
    with multiprocessing.Pool() as pool:
        offers = pool.map(AgentService.run,agents)
    pool.join()
    pool.terminate()
    return offers



