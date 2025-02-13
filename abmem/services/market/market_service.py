import sys
# Add the specified directory to the system path to allow imports from that location
sys.path.append("D:/Projeler/abm/abmem_project/test")

# Import necessary modules and services from the project
from ...models.enums import MarketState, MarketStrategy
from ...services.market import period_factory as PeriodFactory
from ...services.visualization import visualization_service as VisualizationService
from ...models import Market, Period, Offer, Agent
from decimal import Decimal
from ...services.algorithms import MPIP
from ...services.agent import agent_factory as AgentFactory, agent_service as AgentService
from ...services.simulation import parallel_service as ParallelService
from ...services.file_reader import reader_service as ReaderService
from ...constants import *
import numpy as np
import decimal
import timeit
from django.db.models import Q

# Initialize a global variable for market data
marketData = []

algorithms = []

def init(market: Market) -> None:
    global marketData
    # Read market data from an Excel file and map the columns
    marketData = ReaderService.readExcel(
        path='marketData.xlsx',
        columns=['Submitted Bid Order Volume(MWh)', 'Daily exchange rates(USD)', 
                 'Natural Gas Price (USD/1000Sm3)', 'İstanbul average temperature'],
        map=['demand', 'der', 'ngp', 'ist']
    )
    for agent in market.agent_set.all():
        algorithm = AgentService.init(agent)
        algorithms.append(algorithm)

    # Set the market state to INITIALIZED and save
    market.state = MarketState.INITIALIZED
    market.save()

def estimatePTF(market: Market):
    day = market.simulation.day
    currentPeriod = market.simulation.currentPeriod

    # Determine if the current day is a holiday (if day % 6 == 0)
    holiday = 0
    if day % 6 == 0:
        holiday = 1

    # Fetch periods for specific previous intervals (1, 24, 168, and 672 periods ago)
    periods = market.period_set.filter(
        Q(periodNumber=currentPeriod - 1) |
        Q(periodNumber=currentPeriod - 24) |
        Q(periodNumber=currentPeriod - 168) |
        Q(periodNumber=currentPeriod - 672)
    )

    # Initialize variables for previous market clearing prices and total generation
    mcp24 = 1
    mcp168 = 1
    mcp672 = 1
    tg = 1

    # Extract relevant data from previous periods
    for period in periods:
        if period.periodNumber == currentPeriod - 1:
            tg = float(period.marketVolume)
        if period.periodNumber == currentPeriod - 24:
            mcp24 = float(period.ptf)
        elif period.periodNumber == currentPeriod - 168:
            mcp168 = float(period.ptf)
        elif period.periodNumber == currentPeriod - 672:
            mcp672 = float(period.ptf)

    # Extract market data for the current period
    der = marketData['der'][currentPeriod] # Daily Exchange Rate
    ngp = marketData['ngp'][currentPeriod] # Natural Gas Price
    ist = marketData['ist'][currentPeriod] # İstanbul Weather

    # Initialize resource capacities
    ng = 1
    lig = 1
    rhyd = 1
    icoal = 1
    sol = 1
    asph = 1
    bcoal = 1
    imex = 1

    # Sum up the capacities for each resource type based on the agents' portfolios
    agents = market.agent_set.all()
    for agent in agents:
        for plant in agent.portfolio.plant_set.all():
            if plant.resource.name == "naturalgas":
                ng += plant.capacity
            elif plant.resource.name == "lignite":
                lig += plant.capacity
            elif plant.resource.name == "hydro":
                rhyd += plant.capacity
            elif plant.resource.name == "importcoal":
                icoal += plant.capacity
            elif plant.resource.name == "solar":
                sol += plant.capacity
            elif plant.resource.name == "asphaltitecoal":
                asph += plant.capacity
            elif plant.resource.name == "blackcoal":
                bcoal += plant.capacity
            elif plant.resource.name == "imex":
                imex += plant.capacity

    # Calculate the PTF (Market Clearing Price) using the MPIP algorithm
    ptf = MPIP.calculate(
        holy=holiday, mcp24=mcp24, mcp168=mcp168, mcp672=mcp672, der=der,
        tg=tg, ng=ng, lig=lig, rhyd=rhyd, icoal=icoal, sol=sol, asph=asph, bcoal=bcoal, imex=imex,
        ngp=ngp, ist=ist
    )
    
    print(ptf)
    return ptf

def startPool(market: Market,algorithms) -> None:
    # Set market state to WAITINGAGENTS and save
    market.state = MarketState.WAITINGAGENTS
    market.save()
    return ParallelService.startPool(market.agent_set.all(),algorithms)

from collections import defaultdict

def marketClearing(market: Market, offers: [Offer], demand: int):
    # Set market state to CALCULATING and save
    market.state = MarketState.CALCULATING
    market.save()
    
    metDemand = demand
    ptf = 0
    
    # Group offers by their prices
    offer_groups = defaultdict(list)
    volume = 0
    for offer in offers:
        offer_groups[offer.offerPrice].append(offer)
        volume += offer.amount
    
    # Sort the offer groups by price in ascending order
    sorted_groups = sorted(offer_groups.items(), key=lambda x: x[0])

    # Match offers to demand based on price
    for _, group in sorted_groups:
        if demand > 0:
            total_amount = sum(offer.amount for offer in group)
            # If the group's total amount meets the demand
            if total_amount <= demand:
                for offer in group:
                    offer.acceptanceAmount = offer.amount
                    offer.acceptance = True
                    offer.acceptancePrice = offer.offerPrice
                    demand -= offer.acceptanceAmount
                    ptf = offer.acceptancePrice
            else:
                # If the group's total amount exceeds demand, perform recursive calculation
                ptf = priceGroupCalculation(group, demand)
                break
        else:
            break
    
    return offers, metDemand - demand, ptf, volume

def priceGroupCalculation(group: [Offer], demand: int):
    # Recursive function to calculate the PTF for the group
    ptf = 0
    if len(group) <= 0 or demand <= 0:
        return 0
    gDemand = demand / len(group)
    for offer in group:
        if offer.amount <= gDemand:
            offer.acceptanceAmount = offer.amount
            offer.acceptance = True
            offer.acceptancePrice = offer.offerPrice
            demand -= offer.acceptanceAmount
            group.remove(offer)
            ptf = priceGroupCalculation(group=group, demand=demand)
            break
        else:
            offer.acceptanceAmount = gDemand
            demand -= offer.acceptanceAmount
            offer.acceptance = True
            offer.acceptancePrice = offer.offerPrice
            ptf = offer.offerPrice
    return ptf


def updatePeriod(period: Period) -> Period:
    # Save the period data and return the updated period
    period.save()
    return period

def saveOffers(market: Market, offers: [Offer]) -> None:
    # Set market state to BROADCASTING and save offers
    market.state = MarketState.BROADCASTING
    for offer in offers:
        offer.save()
        agent = offer.agent
        agent.budget += budgetCalculation(offer)
        agent.save()

def budgetCalculation(offer: Offer):
    # Calculate and return the budget based on the offer's acceptance amount and price
    return decimal.Decimal((decimal.Decimal(offer.acceptanceAmount) * offer.acceptancePrice)) - (offer.resource.fuelCost * decimal.Decimal(offer.acceptanceAmount))

def createPeriod(market: Market) -> Period:
    # Create and return a new period for the market
    return PeriodFactory.create(market=market, num=market.simulation.currentPeriod, demand=getDemand(market.simulation.currentPeriod))

def showPeriodDetails(period: Period) -> None:
    # Display details of the given period
    print("PTF: ", period.ptf)
    offers = period.offer_set.all()
    for offer in offers:
        print("Agent: ", offer.agent.id, "Resource: ", offer.resource.name, offer.amount, "MW/h        ",
              offer.offerPrice, "$      ", offer.acceptance, " ", offer.acceptancePrice, "$   ", offer.acceptanceAmount, "/", offer.amount, "MW/h")
    VisualizationService.visualizePeriod(period)

def payasptf(offers: [Offer], ptf: int):
    # Adjust acceptance prices of accepted offers to the PTF
    for offer in offers:
        if offer.acceptance:
            offer.acceptancePrice = ptf
    return offers


import time
def run(market: Market) -> bool:
    start = timeit.default_timer()
    
    if market.state == MarketState.CREATED:
        print("market inited in market service")
        init(market)
    global algorithms
    # Create a new period, estimate the PTF, and start the agent pool
    period = createPeriod(market)
    #period.estimatedPtf = estimatePTF(market)
    algorithms,offers = startPool(market,algorithms)
    offers = np.concatenate(offers)

    # Perform market clearing and adjust offers according to the market strategy
    offers, metDemand, ptf, volume = marketClearing(market, offers, period.demand)
    if market.strategy == MarketStrategy.PAYASPTF:
        payasptf(offers, ptf)

    # Update the period and save the offers
    period.metDemand = metDemand
    period.ptf = ptf  # MCP (PTF) değerini period'a kaydediyoruz
    period.marketVolume = volume
    period = updatePeriod(period=period)
    saveOffers(market, offers)
    print("funcs called and period updated")

    # Offer nesnelerine period bilgisini ekle
    for offer in offers:
        offer.period = period
        offer.save()

    # Mark the market state as PERIODEND and save
    market.state = MarketState.PERIODEND
    market.save()
    print("period details will be shown")
    showPeriodDetails(period)
    print(timeit.default_timer() - start)
    
    return offers

def getDemand(currentPeriod: int) -> int:
    # Retrieve and return the demand for the current period from the market data
    return marketData['demand'][currentPeriod]
