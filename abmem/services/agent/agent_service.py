import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")

import django
django.setup()

from ...models.enums import AgentState
from ...models import Agent, Offer, Portfolio
from ...services.agent import portfolio_factory as PortfolioFactory
from ...services.agent import offer_factory as OfferFactory
from ..algorithms.agent_algorithm import AgentAlgorithm
from ..algorithms.algorithm_utils import State
import random
from decimal import Decimal

PENALTY = 5000

def init(agent: Agent):
    # Initialize the agent's state without creating a portfolio
    agent.state = AgentState.INITIALIZED
    action_dim = agent.portfolio.plant_set.count()
    agent.algorithm = AgentAlgorithm(action_dim,agent.id)
    agent.save()
    return agent.algorithm

def relearn(agent: Agent, results) -> None:
    # Set the agent's state to learning and save
    agent.state = AgentState.LEARNING
    agent.save()
    # learn(self, state, action, next_state, reward, done=False):
    agent.algorithm.learn(results[0],results[1],results[2],results[3])

def predict(agent: Agent, results) -> int:
    # Set the agent's state to predicting and save
    agent.state = AgentState.PREDICTING
    agent.save()
    # Prediction module placeholder, returns a random prediction
    return random.randint(10, 1000)

def calculateOffers(agent: Agent) -> [Offer]:
    # Set the agent's state to calculating and save
    agent.state = AgentState.CALCULATING
    agent.save()
    offers = []

    period = agent.market.period_set.latest()
    played_period = agent.market.period_set.order_by('-id')[1]
    state = State(mcp=played_period.ptf,demand=period.demand)
    actions = agent.algorithm.selectAction(state)

    counter = 0
    for plant in agent.portfolio.plant_set.all():
        offer = OfferFactory.create(
            agent=agent,
            resource=plant.resource,
            amount=plant.capacity,
            offerPrice=Decimal(float(actions[counter]))
        )
        counter +=1 
        offers.append(offer)
    return offers

def calculateRandomOffers(agent: Agent) -> [Offer]:
    # Set the agent's state to calculating and save
    agent.state = AgentState.CALCULATING
    agent.save()
    offers = []
    for plant in agent.portfolio.plant_set.all():
        offer = OfferFactory.create(
            agent=agent,
            resource=plant.resource,
            amount=plant.capacity,
            offerPrice=random.randint(0,200)
        )
        offers.append(offer)
    return offers


def saveOffers(offers: [Offer]) -> None:
    # Save all the generated offers to the database
    for offer in offers:
        offer.save()

def createPortfolio(agent: Agent, plantsData: dict) -> Portfolio:
    # Create a portfolio for the agent using the provided data
    return PortfolioFactory.create(agent, plantsData)

def run(agent,algorithm) -> bool:
    agent.algorithm = algorithm
    agent.state = AgentState.RUNNING
    agent.save()

    if agent.market.period_set.count() >= 3:
        last_period = agent.market.period_set.order_by('-id')[2]
        played_period = agent.market.period_set.order_by('-id')[1]
        last_offers = agent.offer_set.filter(period=played_period)
        actions = []
        reward = 0

        for offer in last_offers:
            actions.append(offer.offerPrice)
            if offer.acceptanceAmount > 0:
                reward += ((Decimal(offer.acceptanceAmount) * offer.acceptancePrice)) - (offer.resource.fuelCost * Decimal(offer.acceptanceAmount))
            else:
                reward += -(PENALTY)

        results = [
            State(mcp=last_period.ptf, demand=last_period.demand),
            actions,
            State(mcp=played_period.ptf, demand=played_period.demand),
            reward
        ]

        relearn(agent, results)
        offers = calculateOffers(agent)
    else:
        offers = calculateRandomOffers(agent)

    saveOffers(offers)
    agent.state = AgentState.WAITING
    agent.save()

    return agent.algorithm,offers

