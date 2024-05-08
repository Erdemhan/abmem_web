from django.shortcuts import render, redirect
from .forms import ResourceForm, PortfolioForm, PlantForm, PeriodForm, OfferForm
from .models.models import Resource, Portfolio, Plant, Period, Offer
from .models import Simulation, Agent, enums
from django.http.response import HttpResponse
from .services.starter import starter_service as StarterService
from django.http import HttpResponseNotFound


def start(request):
    offers = StarterService.start()
    return HttpResponse(offers)


def dashboard(request):
    num_resources = Resource.objects.count()
    num_agents = Agent.objects.count()
    num_simulations = Simulation.objects.count()
    latest_simulations = Simulation.objects.order_by('-created_at')[:5]
    context = {
        'num_resources': num_resources,
        'num_agents': num_agents,
        'num_simulations': num_simulations,
        'latest_simulations': latest_simulations
    }
    return render(request, 'pages/dashboard.html', context)


def blank(request):
    return render(request, 'pages/blank.html')



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


def simulation_list(request):
    simulations = Simulation.objects.order_by('-created_at')
    return render(request, 'pages/simulation-list.html', {'simulations': simulations})

def simulation_create(request):
    simulations = Simulation.objects.order_by('-created_at')
    return render(request, 'pages/simulation-create.html', {'simulations': simulations})

def simulation_view(request, sim_id):
    simulations = Simulation.objects.order_by('-created_at')
    return render(request, 'pages/blank.html', {'simulations': simulations})





#RESOURCE



#PORTFOLIO

def portfolio_list(request):
    portfolios = Portfolio.objects.all()
    return render(request, 'portfolio_list.html', {'portfolios': portfolios})

def portfolio_detail(request, pk):
    portfolio = Portfolio.objects.get(pk=pk)
    return render(request, 'portfolio_detail.html', {'portfolio': portfolio})

def portfolio_create(request):
    if request.method == 'POST':
        form = PortfolioForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('portfolio_list')
    else:
        form = PortfolioForm()
    return render(request, 'portfolio_form.html', {'form': form})

def portfolio_update(request, pk):
    portfolio = Portfolio.objects.get(pk=pk)
    if request.method == 'POST':
        form = PortfolioForm(request.POST, instance=portfolio)
        if form.is_valid():
            form.save()
            return redirect('portfolio_list')
    else:
        form = PortfolioForm(instance=portfolio)
    return render(request, 'portfolio_form.html', {'form': form})

def portfolio_delete(request, pk):
    portfolio = Portfolio.objects.get(pk=pk)
    if request.method == 'POST':
        portfolio.delete()
        return redirect('portfolio_list')
    return render(request, 'portfolio_confirm_delete.html', {'portfolio': portfolio})


#PLANT

def plant_list(request):
    plants = Plant.objects.all()
    return render(request, 'plant_list.html', {'plants': plants})

def plant_detail(request, pk):
    plant = Plant.objects.get(pk=pk)
    return render(request, 'plant_detail.html', {'plant': plant})

def plant_create(request):
    if request.method == 'POST':
        form = PlantForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('plant_list')
    else:
        form = PlantForm()
    return render(request, 'plant_form.html', {'form': form})

def plant_update(request, pk):
    plant = Plant.objects.get(pk=pk)
    if request.method == 'POST':
        form = PlantForm(request.POST, instance=plant)
        if form.is_valid():
            form.save()
            return redirect('plant_list')
    else:
        form = PlantForm(instance=plant)
    return render(request, 'plant_form.html', {'form': form})

def plant_delete(request, pk):
    plant = Plant.objects.get(pk=pk)
    if request.method == 'POST':
        plant.delete()
        return redirect('plant_list')
    return render(request, 'plant_confirm_delete.html', {'plant': plant})


#PERIOD

def period_list(request):
    periods = Period.objects.all()
    return render(request, 'period_list.html', {'periods': periods})

def period_detail(request, pk):
    period = Period.objects.get(pk=pk)
    return render(request, 'period_detail.html', {'period': period})

def period_create(request):
    if request.method == 'POST':
        form = PeriodForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('period_list')
    else:
        form = PeriodForm()
    return render(request, 'period_form.html', {'form': form})

def period_update(request, pk):
    period = Period.objects.get(pk=pk)
    if request.method == 'POST':
        form = PeriodForm(request.POST, instance=period)
        if form.is_valid():
            form.save()
            return redirect('period_list')
    else:
        form = PeriodForm(instance=period)
    return render(request, 'period_form.html', {'form': form})

def period_delete(request, pk):
    period = Period.objects.get(pk=pk)
    if request.method == 'POST':
        period.delete()
        return redirect('period_list')
    return render(request, 'period_confirm_delete.html', {'period': period})


#OFFER

def offer_list(request):
    offers = Offer.objects.all()
    return render(request, 'offer_list.html', {'offers': offers})

def offer_detail(request, pk):
    offer = Offer.objects.get(pk=pk)
    return render(request, 'offer_detail.html', {'offer': offer})

def offer_create(request):
    if request.method == 'POST':
        form = OfferForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('offer_list')
    else:
        form = OfferForm()
    return render(request, 'offer_form.html', {'form': form})

def offer_update(request, pk):
    offer = Offer.objects.get(pk=pk)
    if request.method == 'POST':
        form = OfferForm(request.POST, instance=offer)
        if form.is_valid():
            form.save()
            return redirect('offer_list')
    else:
        form = OfferForm(instance=offer)
    return render(request, 'offer_form.html', {'form': form})

def offer_delete(request, pk):
    offer = Offer.objects.get(pk=pk)
    if request.method == 'POST':
        offer.delete()
        return redirect('offer_list')
    return render(request, 'offer_confirm_delete.html', {'offer': offer})
