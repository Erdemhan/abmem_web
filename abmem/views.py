from django.shortcuts import render, redirect
from .forms import ResourceForm, PortfolioForm, PlantForm, PeriodForm, OfferForm
from .models.models import Resource, Portfolio, Plant, Period, Offer
from django.http.response import HttpResponse
from .services.starter import starter_service as StarterService


def index(request):
    StarterService.start()
    return HttpResponse("Started")

#RESOURCE

def resource_list(request):
    resources = Resource.objects.all()
    return render(request, 'resource_list.html', {'resources': resources})

def resource_detail(request, pk):
    resource = Resource.objects.get(pk=pk)
    return render(request, 'resource_detail.html', {'resource': resource})

def resource_create(request):
    if request.method == 'POST':
        form = ResourceForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('resource_list')
    else:
        form = ResourceForm()
    return render(request, 'resource_form.html', {'form': form})

def resource_update(request, pk):
    resource = Resource.objects.get(pk=pk)
    if request.method == 'POST':
        form = ResourceForm(request.POST, instance=resource)
        if form.is_valid():
            form.save()
            return redirect('resource_list')
    else:
        form = ResourceForm(instance=resource)
    return render(request, 'resource_form.html', {'form': form})

def resource_delete(request, pk):
    resource = Resource.objects.get(pk=pk)
    if request.method == 'POST':
        resource.delete()
        return redirect('resource_list')
    return render(request, 'resource_confirm_delete.html', {'resource': resource})


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
