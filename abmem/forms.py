from django import forms
from .models.models import Resource, Portfolio, Plant, Period, Offer

class ResourceForm(forms.ModelForm):
    class Meta:
        model = Resource
        fields = ['energyType', 'name', 'fuelCost', 'emission']

class PortfolioForm(forms.ModelForm):
    class Meta:
        model = Portfolio
        fields = ['agent']

class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = ['portfolio', 'resource', 'capacity']

class PeriodForm(forms.ModelForm):
    class Meta:
        model = Period
        fields = ['market', 'periodNumber', 'demand', 'metDemand', 'marketVolume', 'ptf']

class OfferForm(forms.ModelForm):
    class Meta:
        model = Offer
        fields = ['period', 'agent', 'resource', 'amount', 'offerPrice', 'acceptance', 'acceptancePrice', 'acceptanceAmount']
