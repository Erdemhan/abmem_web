from ...models import  Agent
from ...services.agent import agent_service as AgentService
from ...constants import *
import multiprocessing
import django

def init_worker():
    """
    Initialize Django in the child process.
    This function is called in each child process before running tasks.
    It's necessary to ensure Django is properly set up within child processes.
    """
    if not django.apps.apps.ready:  # Django'nun zaten başlatılıp başlatılmadığını kontrol et
        django.setup()

import multiprocessing

def startPool(agents, algorithms):
    """
    Start a multiprocessing pool to execute the AgentService.run function for each agent.

    Args:
        agents ([Agent]): A list of Agent objects to be processed.
        algorithms ([Algorithm]): A list of Algorithm objects with agent_id attributes.

    Returns:
        ([Algorithm], [Offer]): A tuple containing:
            - A list of updated algorithms.
            - A list of offers produced by the agents.
    """
    import multiprocessing

    # Create a dictionary to map agent IDs to their algorithms
    algorithm_map = {alg.agent_id: alg for alg in algorithms}

    # Prepare the input data for multiprocessing
    tasks = [(agent, algorithm_map.get(agent.id)) for agent in agents]

    # Create a multiprocessing pool and map the AgentService.run function to the tasks
    with multiprocessing.Pool() as pool:
        results = pool.starmap(AgentService.run, tasks)

    # Separate the algorithms and offers from the results
    updated_algorithms, offers = zip(*results)  # Unpack the tuple into two lists

    # Convert back to lists (optional, since zip returns tuples)
    updated_algorithms = list(updated_algorithms)
    offers = list(offers)

    # Return the separated lists
    return updated_algorithms, offers
