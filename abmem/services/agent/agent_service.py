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
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def init(agent: Agent):
    # Initialize the agent's state without creating a portfolio
    agent.state = AgentState.INITIALIZED
    action_dim = agent.portfolio.plant_set.count()
    algorithm = AgentAlgorithm(action_dim, agent.name)
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
    played_period = agent.market.period_set.order_by('-id')[1]
    state = State(mcp=played_period.ptf,demand=period.demand)

    counter = 0
    actions = agent.algorithm.selectAction(state)
    for plant in agent.portfolio.plant_set.all():
        offerPrice = actions[counter]

        if isinstance(offerPrice, list):  # <--- Yeni kontrol
            offerPrice = offerPrice[0]

        if agent.algorithm.failStack > 6:
            if random.random() < 0.7:
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

def compute_composite_reward(last_offers, market_price, max_capacity, price_cap=200):
    offer = last_offers[0]
    offer_price = float(offer.offerPrice)
    offer_amount = float(offer.amount)
    acceptance_amount = float(offer.acceptanceAmount)
    acceptance_price = float(offer.acceptancePrice)
    resource = offer.resource

    static_cost = float(resource.staticCost()) * offer_amount
    variable_cost = float(resource.variableCost()) * acceptance_amount

    if acceptance_amount > 0:
        budget_growth = (acceptance_amount * acceptance_price) - (static_cost + variable_cost)
    else:
        budget_growth = -static_cost

    max_possible_growth = price_cap * max_capacity
    normalized_growth = budget_growth / max_possible_growth  # ŞİMDİ -1 ila +1 arasında

    # MCP similarity (her zaman 0–1)
    offer_mcp_similarity = 1 - abs(float(offer_price)- float(market_price)) / float(price_cap)
    offer_mcp_similarity = max(min(offer_mcp_similarity, 1), 0)

    reward = 0.5 * normalized_growth + 0.5 * offer_mcp_similarity
    print((normalized_growth, offer_mcp_similarity))
    return reward




def run(agent) -> bool:

    agent.state = AgentState.RUNNING
    agent.save()

    if agent.market.period_set.count() >= 3:
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
        reward = compute_composite_reward(
            last_offers,
            played_period.ptf,
            offer.amount,
            price_cap=200
        )


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

    return agent,offers





