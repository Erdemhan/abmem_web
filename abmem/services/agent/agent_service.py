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
import torch
from decimal import Decimal
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(agent: Agent):
    # Initialize the agent's state without creating a portfolio
    agent.state = AgentState.INITIALIZED
    action_dim = agent.portfolio.plant_set.count()
    algorithm = AgentAlgorithm(action_dim, agent.id)


    agent.algorithm = algorithm
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
    new_offers = []

    period = agent.market.period_set.latest()
    offers = agent.offer_set.all()
    played_period = agent.market.period_set.order_by('-id')[1]
    state = State(mcp=played_period.ptf,demand=period.demand)


    accepted_offers = sum(1 for offer in offers if offer.acceptance)
    acceptance_rate = accepted_offers / len(offers) if len(offers) > 0 else 0
    counter = 0
    actions = agent.algorithm.selectAction(state)
    for plant in agent.portfolio.plant_set.all():
        offerPrice = actions[counter]

        if isinstance(offerPrice, list):  # <--- Yeni kontrol
            offerPrice = offerPrice[0]

        if agent.algorithm.failStack > 4:
            if random.random() < 0.9:
                agent.algorithm.failStack = int(agent.algorithm.failStack / 2)
                offerPrice = Decimal(random.randrange(int(played_period.ptf - 10), int(played_period.ptf + 10)))
            else:
                offerPrice = Decimal(float(offerPrice))
        else:
            offerPrice = Decimal(float(offerPrice))

        offer = OfferFactory.create(
            agent=agent,
            resource=plant.resource,
            amount=plant.capacity,
            offerPrice=offerPrice
        )
        counter += 1
        new_offers.append(offer)

    return new_offers

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

def to_safe_history(history):
    return [
        [int(x.item()) if hasattr(x, "item") else int(x) for x in list(row)]
        for row in history
    ]


def run(agent,algorithm) -> bool:
    agent.algorithm = algorithm
    h = agent.algorithm.hidden[0]
    c = agent.algorithm.hidden[1]
    if isinstance(h, np.ndarray):
        h = torch.tensor(h, dtype=torch.float32).to(device)
        c = torch.tensor(c, dtype=torch.float32).to(device)
    if h.dim() == 2:
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
    agent.algorithm.hidden = (h, c)


    agent.state = AgentState.RUNNING
    agent.save()

    if agent.market.period_set.count() >=  algorithm.seq_len:
        
        if agent.market.period_set.order_by('-id')[2:3].exists():
            last_period = agent.market.period_set.order_by('-id')[2]
        if agent.market.period_set.order_by('-id')[1:2].exists():
            played_period = agent.market.period_set.order_by('-id')[1]
        if agent.market.period_set.order_by('-id')[23:24].exists():
            last24_period = agent.market.period_set.order_by('-id')[23]
            last24_period_ptf = last24_period.ptf
        else:
            last24_period_ptf= 0
        if agent.market.period_set.order_by('-id')[168:169].exists():
            last168_period = agent.market.period_set.order_by('-id')[167]
            last168_period_ptf = last168_period.ptf
        else:
            last168_period_ptf = 0
        if agent.market.period_set.order_by('-id')[24:25].exists():
            last24_period_old = agent.market.period_set.order_by('-id')[24]
            last24_period_old_ptf = last24_period_old.ptf
        else:
            last24_period_old_ptf = 0
        if agent.market.period_set.order_by('-id')[168:169].exists():
            last168_period_old = agent.market.period_set.order_by('-id')[168]
            last168_period_old_ptf = last168_period_old.ptf
        else:
            last168_period_old_ptf = 0

    
        last_offers = agent.offer_set.filter(period=played_period)
        actions = []
        reward = 0

        for offer in last_offers:
            actions.append(offer.offerPrice)
            if offer.acceptanceAmount > 0:
                reward += ((Decimal(offer.acceptanceAmount) * offer.acceptancePrice)) - ((offer.resource.staticCost() * offer.amount) + (offer.resource.variableCost() * offer.acceptanceAmount))
            else:
                reward += -(offer.resource.staticCost() * offer.amount)


        results = [
            State(mcp=last_period.ptf, demand=last_period.demand,mcp24=last24_period_old_ptf,mcp168= last168_period_old_ptf),
            actions,
            State(mcp=played_period.ptf, demand=played_period.demand,mcp24=last24_period_ptf,mcp168=last168_period_ptf),
            reward
        ]

        relearn(agent, results)
        offers = calculateOffers(agent)
    else:
        offers = calculateRandomOffers(agent)

    saveOffers(offers)
    agent.state = AgentState.WAITING
    agent.save()

    agent.algorithm.hidden = (
        agent.algorithm.hidden[0].detach().cpu().numpy(),
        agent.algorithm.hidden[1].detach().cpu().numpy()
    )
    return agent.algorithm,offers





