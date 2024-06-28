from django.shortcuts import render, redirect
from .forms import ResourceForm, PlantForm, AgentForm
from .models.models import Resource, Portfolio, Plant, Period
from .models import Simulation, Agent, enums
from django.http.response import HttpResponse
from .services.starter import starter_service as StarterService
from .services.simulation import simulation_factory as SimulationFactory
from .services.agent import agent_factory as AgentFactory
from .services.market import market_factory as MarketFactory
from django.http import HttpResponseNotFound, HttpResponseBadRequest
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from .models import Agent, Portfolio, Plant
from .forms import AgentForm,  PlantForm
import copy
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Simulation, Agent, Portfolio, Plant
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render
from django.http import JsonResponse
from .models import Agent
import re



# SYSTEM
def start(request):
    offers = StarterService.start()
    return HttpResponse(offers)

def run_simulation(request, simulation_id):
    simulation = get_object_or_404(Simulation, id=simulation_id)
    try:
        if simulation.proxy or simulation.state != enums.SimulationState.CREATED :
            newSimulation = copy.deepcopy(simulation)
            newSimulation.id = None
            newSimulation.state = enums.SimulationState.CREATED
            newSimulation.currentPeriod = -1
            newSimulation.proxy = False
            newSimulation.save()
            newMarket = copy.deepcopy(simulation.market)
            newMarket.id = None
            newMarket.state = enums.MarketState.CREATED
            newMarket.simulation = newSimulation
            newMarket.proxy = False
            newMarket.save()

            for agent in list(simulation.market.agent_set.all()):
                newAgent = copy.deepcopy(agent)
                newAgent.id = None
                newAgent.market = newMarket
                newAgent.proxy = False
                newAgent.save()
                newPortfolio = copy.deepcopy(agent.portfolio)
                newPortfolio.id = None
                newPortfolio.proxy = False
                newPortfolio.agent = newAgent
                newPortfolio.save()
                for plant in list(agent.portfolio.plant_set.all()):
                    newPlant = copy.deepcopy(plant)
                    newPlant.id = None
                    newPlant.portfolio = newPortfolio
                    newPlant.proxy = False
                    newPlant.save()
            simulation_id = StarterService.start(newSimulation)
            return redirect('watch_simulation', simulation_id=simulation_id)
    except Exception as e:
        print(str(e))
        if newSimulation.id != None:
            newSimulation.delete()
        if newMarket.id != None:
            newMarket.delete()
        return HttpResponseBadRequest()
    
    return redirect('simulation_list')

def dashboard(request):
    num_resources = Resource.objects.count()
    num_agents = Agent.objects.count()
    num_simulations = Simulation.objects.filter(proxy=False).count()
    num_proxy_simulations = Simulation.objects.filter(proxy=True).count()
    latest_simulations = Simulation.objects.filter(proxy=False).order_by('-created_at')[:5]
    context = {
        'num_resources': num_resources,
        'num_agents': num_agents,
        'num_simulations': num_simulations,
        'num_proxy_simulations': num_proxy_simulations,
        'latest_simulations': latest_simulations
    }
    return render(request, 'pages/dashboard.html', context)

def blank(request):
    resources = Resource.objects.all()
    return render(request, 'pages/blank.html', {'resources': resources})

def watch_simulation(request, simulation_id):
    simulation = get_object_or_404(Simulation, id=simulation_id)
    context = {
        'simulation_id': simulation.id
    }
    return render(request, 'pages/watch_simulation.html', context)

# Simülasyon durumu kontrol view fonksiyonu
def get_simulation_status(request):
    simulation_id = request.GET.get('simulation_id')
    simulation = get_object_or_404(Simulation, id=simulation_id)
    status = simulation.state.name
    mode = simulation.mode.name
    market = simulation.market
    marketStatus = market.state.name
    currentPeriod = simulation.currentPeriod
    agents = list(market.agent_set.all().order_by('-budget'))
    agent_list = []
    for agent in agents:
        agent_list.append({
            'name':agent.name,
            'state':agent.state.name,
            'budget': agent.budget
        })

    periods_list = []
    if simulation.state == enums.SimulationState.FINISHED:
        periods = list(market.period_set.all())
        for period in periods:
            offersList = []
            if period != None:
                offers = list(period.offer_set.all().order_by('-agent_id'))
                for offer in offers:
                    offersList.append(offer.to_dict())
                if period.ptf != -1:
                    offersList.append(create_mcp(period.ptf))
                periods_list.append({
                     'id': period.periodNumber,
                    'offers':offersList
                })         
    else:
        period = market.period_set.filter(periodNumber = currentPeriod).first()
        offersList = []
        if period != None:
            offers = list(period.offer_set.all().order_by('-agent_id'))
            for offer in offers:
                offersList.append(offer.to_dict())
            if period.ptf != -1:
                offersList.append(create_mcp(period.ptf))
            periods_list.append({
                    'id': period.periodNumber,
                    'offers':offersList
                })


    return JsonResponse({'status': status,
                        'mode':mode,
                         'total': simulation.periodNumber,
                         'period': currentPeriod,
                         'marketStatus': marketStatus,
                         'agents': agent_list,
                         'periods': periods_list
                         })

def nextPeriod(request):
    simulation_id = request.GET.get('simulation_id')
    simulation = get_object_or_404(Simulation, id=simulation_id)
    simulation.market.state = enums.MarketState.INITIALIZED
    fb = simulation.market.save()
    return JsonResponse({'status': 'OK'
                         })


def create_mcp(mcp):
    return {
    'id': -1,
    'agent': 'MCP',
    'resource': mcp,
    'amount': mcp,
    'offerPrice': mcp,
    'acceptance': mcp,
    'acceptancePrice': mcp,
    'acceptanceAmount': mcp,
    }

#RESOURCE
def resource_list(request):
    resources = Resource.objects.all()
    return render(request, 'pages/resource-list.html', {'resources': resources})

'''def resource_detail(request, resource_id):
    resource = Resource.objects.filter(id=resource_id).first()
    if resource:
        resourceDTO={'energyType':resource.energyType, 
                'name': resource.name, 
                'fuelCost': resource.fuelCost, 
                'emission': resource.emission}
        return render(request, 'pages/resource-view.html', {'resource': resourceDTO})
    else:
        return HttpResponseNotFound("KAYNAK BULUNAMADI")'''

def resource_create(request):
    if request.method == 'POST':
        form = ResourceForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('resource_list')
    else:
        form = ResourceForm()
    return redirect(resource_list)

def resource_delete(request, resource_id):
    resource = Resource.objects.get(id=resource_id)
    if request.method == 'GET' and resource:
        resource.delete()
        return redirect('resource_list')
    else:
        return HttpResponseNotFound("KAYNAK BULUNAMADI")

def resource_update(request, resource_id):
    resource = Resource.objects.get(id=resource_id)
    if request.method == 'POST':
        form = ResourceForm(request.POST, instance=resource)
        if form.is_valid():
            form.save()
            return redirect('resource_list')
    else:
        return HttpResponseNotFound("KAYNAK BULUNAMADI")



#SIMULATION
def simulation_list_proxy(request):
    simulations = Simulation.objects.filter(proxy=True).order_by('-updated_at')
    return render(request, 'pages/simulation-list-proxy.html', {'simulations': simulations})

def simulation_list(request):
    simulations = Simulation.objects.filter(proxy=False).order_by('-updated_at')
    return render(request, 'pages/simulation-list.html', {'simulations': simulations})

@csrf_exempt
def create_simulation(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        simulation_name = data.get('simulation_name')
        simulationMode= int(data['simulationMode'])
        periodNumber = int(data['periodNumber'])
        periodType = int(data['periodType'])

        marketStrategy = int(data['strategy'])
        lowerBound = int(data['lowerBound'])
        upperBound = int(data['upperBound'])
        
        simulation = SimulationFactory.create(name=simulation_name, mode=enums.SimulationMode(simulationMode), 
                                               periodType=enums.PeriodType(periodType), periodNumber= periodNumber, proxy=True)

        market = MarketFactory.create(
            sim=simulation,
            lowerBound=lowerBound,
            upperBound=upperBound,
            strategy= enums.MarketStrategy(marketStrategy)
        )


        agents_data = data.get('agents', [])

        for agent_data in agents_data:
            agent = AgentFactory.create(
                name=agent_data['name'],
                type=enums.AgentType(int(agent_data['type'])),
                budget=agent_data['budget'],
                market=market,
                proxy=market.proxy
            )

            portfolio = Portfolio.objects.create(agent=agent,proxy=agent.proxy)

            for plant_data in agent_data['plants']:
                resource = Resource.objects.get(id=int(plant_data['resource']))
                Plant.objects.create(
                    portfolio=portfolio,
                    resource=resource,
                    capacity=plant_data['capacity'],
                    proxy = portfolio.proxy
                )
        return JsonResponse({'status': 'success'}, status=200)
    else:
        resources = Resource.objects.all()
        return render(request, 'pages/simulation-create.html', {'resources': resources})

def delete_simulation(request, simulation_id):
    simulation = get_object_or_404(Simulation, id=simulation_id)
    if simulation.proxy:
        simulation.delete()
        return redirect('simulation_list_proxy')
    else:
        simulation.delete()
        return redirect('simulation_list')
    

@csrf_exempt
def update_simulation(request, simulation_id):
    if request.method == 'GET':
        resources = Resource.objects.all()
        simulation = get_object_or_404(Simulation, id=simulation_id)
        agents = list(simulation.market.agent_set.all())
        return render(request, 'pages/simulation-update.html',{'simulation':simulation,
                                                            'agents': agents,
                                                            'resources': resources})
    
    if request.method == 'POST':
        data = json.loads(request.body)

        simulation_name = data.get('simulation_name')
        simulationMode= int(data['simulationMode'])
        periodNumber = int(data['periodNumber'])
        periodType = int(data['periodType'])

        marketStrategy = int(data['strategy'])
        lowerBound = int(data['lowerBound'])
        upperBound = int(data['upperBound'])

        simulation = get_object_or_404(Simulation, id=simulation_id)

        if simulation.proxy:
            simulation.name = simulation_name
            simulation.mode = enums.SimulationMode(simulationMode)
            simulation.periodNumber = periodNumber
            simulation.periodType = enums.PeriodType(periodType)
            simulation.state = enums.SimulationState.CREATED
            simulation.currentPeriod = -1
            simulation.proxy = True
            simulation.save()

            market = simulation.market
            market.lowerBidBound = lowerBound
            market.upperBidBound = upperBound
            market.strategy = enums.MarketStrategy(marketStrategy)
            market.save()


            market.agent_set.all().delete()

            agents_data = data.get('agents', [])

            for agent_data in agents_data:
                agent = AgentFactory.create(
                    name=agent_data['name'],
                    type=enums.AgentType(int(agent_data['type'])),
                    budget=agent_data['budget'],
                    market=market,
                    proxy=True
                )

                portfolio = Portfolio.objects.create(agent=agent,proxy=True)

                for plant_data in agent_data['plants']:
                    resource = Resource.objects.get(id=int(plant_data['resource']))
                    Plant.objects.create(
                        portfolio=portfolio,
                        resource=resource,
                        capacity=plant_data['capacity'],
                        proxy=True
                    )

            return JsonResponse({'status': 'success'}, status=200)
        else:
            return HttpResponseBadRequest
            
    return render(request, 'create_simulation.html')

def simulation_view(request, sim_id):
    simulations = Simulation.objects.order_by('-created_at')
    return render(request, 'pages/blank.html', {'simulations': simulations})



#AGENT
def get_agents(request):
    agents = Agent.objects.all().values('name', 'type', 'budget')
    return JsonResponse(list(agents), safe=False)

def create_agent_and_related(request):
    if request.method == 'POST':
        agent_form = AgentForm(request.POST)
        resources = request.POST.getlist('resources[]')
        capacities = request.POST.getlist('capacities[]')
        plant_forms = [PlantForm({'resource': resource, 'capacity': capacity}) for resource, capacity in zip(resources, capacities)]

        if agent_form.is_valid() and all([form.is_valid() for form in plant_forms]):
            agent = agent_form.save()
            portfolio = Portfolio.objects.create(agent=agent,proxy = agent.proxy)
            
            for form in plant_forms:
                capacity = form.cleaned_data['capacity']
                Plant.objects.create(portfolio=portfolio, resource=form.cleaned_data['resource'], capacity=capacity, proxy=portfolio.proxy)
            
            return redirect('dashboard')
 
def update_agent(request, agent_id):
    agent = get_object_or_404(Agent, id=agent_id)
    portfolio = agent.portfolio
    if agent.proxy:
        if request.method == 'POST':
            agent_form = AgentForm(request.POST, instance=agent)

            resources = request.POST.getlist('resources[]')
            capacities = request.POST.getlist('capacities[]')
            
            plant_forms = [PlantForm({'resource': resource, 'capacity': capacity}) for resource, capacity in zip(resources, capacities)]

            if agent_form.is_valid() and all(plant_form.is_valid() for plant_form in plant_forms):
                agent_form.save()

                # Sil mevcut Plant'leri
                Plant.objects.filter(portfolio=portfolio).delete()

                # Yeni Plant'leri ekle
                for plant_form in plant_forms:
                    plant = plant_form.save(commit=False)
                    plant.portfolio = portfolio
                    plant.save()

                return redirect('agent_list')
            else:
                return HttpResponseBadRequest()
        else:
                return HttpResponseBadRequest()

def delete_agent(request, agent_id):
    agent = get_object_or_404(Agent, id=agent_id)

    if request.method == 'POST':
        agent.delete()
        return redirect('agent_list')  # agent_list sayfasına yönlendirme
    else:
        return HttpResponseBadRequest()

def agent_list(request):
    if request.method == 'GET':
        resources = Resource.objects.all()
        agents = Agent.objects.filter(proxy= True)
        return render(request, 'pages/agent-list.html',{'resources': resources,
                                                            'agents': agents})

'''
def get_agents(request):
    agents = Agent.objects.all()
    agents_data = []
    for agent in agents:
        agents_data.append({
            'id': agent.id,
            'name': agent.name,
            'type': agent.type,
            'budget': agent.budget
        })
    return JsonResponse({'agents': agents_data})

'''


#HELPERS
def increment_suffix(name):
    # Regex pattern to find an underscore followed by digits at the end of the string
    pattern = r'_(\d+)$'
    
    # Search for the pattern in the given string
    match = re.search(pattern, name)
    
    if match:
        # If a match is found, increment the number by 1
        number = int(match.group(1)) + 1
        # Replace the old number with the new number
        new_name = re.sub(pattern, f'_{number}', name)
    else:
        # If no match is found, append _2 to the string
        new_name = f'{name}_2'
    
    return new_name


      