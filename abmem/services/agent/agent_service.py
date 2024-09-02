import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")

import django
django.setup()

from ...models.enums import AgentState
from ...models import Agent, Offer, Portfolio, Resource
from ...services.file_reader import reader_service as ReaderService
from ...services.agent import portfolio_factory as PortfolioFactory
from ...services.agent import offer_factory as OfferFactory
import random

def init(agent: Agent, portfolioData: dict):
    # Initialize the agent's state and create a portfolio using the provided data
    agent.state = AgentState.INITIALIZED
    createPortfolio(agent, portfolioData)
    agent.save()

def init(agent: Agent):
    # Initialize the agent's state without creating a portfolio
    agent.state = AgentState.INITIALIZED
    agent.save()

def relearn(agent: Agent, results) -> None:
    # Set the agent's state to learning and save
    agent.state = AgentState.LEARNING
    agent.save()
    # Learning module placeholder
    pass

def predict(agent: Agent, results) -> int:
    # Set the agent's state to predicting and save
    agent.state = AgentState.PREDICTING
    agent.save()
    # Prediction module placeholder, returns a random prediction
    return random.randint(10, 1000)

def calculateOffers(agent: Agent, prediction: int) -> [Offer]:
    # Set the agent's state to calculating and save
    agent.state = AgentState.CALCULATING
    agent.save()
    offers = []
    # Calculate random offers for each plant in the agent's portfolio
    for plant in agent.portfolio.plant_set.all():
        lowerBound = plant.resource.fuelCost
        offer = OfferFactory.create(
            agent=agent,
            resource=plant.resource,
            amount=plant.capacity,
            offerPrice=random.randint(lowerBound, agent.market.upperBidBound)
        )
        offers.append(offer)
    # Offer module placeholder
    return offers

def saveOffers(offers: [Offer]) -> None:
    # Save all the generated offers to the database
    for offer in offers:
        offer.save()

def createPortfolio(agent: Agent, plantsData: dict) -> Portfolio:
    # Create a portfolio for the agent using the provided data
    return PortfolioFactory.create(agent, plantsData)

def run(agent: Agent) -> bool:
    # Main function to run the agent's operations
    print(agent)
    if agent.state == AgentState.CREATED:
        # Initialize the agent if it is in the created state
        init(agent)
    print(agent.id, " entered to agent run. Budget: ", agent.budget)
    
    # Set the agent's state to running and save
    agent.state = AgentState.RUNNING
    agent.save()
    
    # Perform learning and prediction, then calculate and save offers
    relearn(agent, results=0)
    prediction = predict(agent, results=0)
    offers = calculateOffers(agent, prediction)
    saveOffers(offers)
    
    # Set the agent's state to waiting and save
    agent.state = AgentState.WAITING
    agent.save()
    
    # Return the generated offers
    return offers
