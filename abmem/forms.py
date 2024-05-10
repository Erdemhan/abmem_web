from django import forms
from .models.models import Resource, Portfolio, Plant, Period, Offer
from .models.agent import AgentProxy
class ResourceForm(forms.ModelForm):
    class Meta:
        model = Resource
        fields = ['name', 'energyType',  'emission', 'fuelCost']


class AgentForm(forms.ModelForm):
    class Meta:
        model = AgentProxy
        fields = ['budget', 'type']  # Agent modelindeki tüm alanları içerir

class PortfolioForm(forms.ModelForm):
    class Meta:
        model = Portfolio
        fields = []  # Herhangi bir alan alınmayacak

class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = ['resource', 'capacity']  # Buradaki fields Plant modelinde bulunan alanları temsil etmeli
class PeriodForm(forms.ModelForm):
    class Meta:
        model = Period
        fields = ['market', 'periodNumber', 'demand', 'metDemand', 'marketVolume', 'ptf']

class OfferForm(forms.ModelForm):
    class Meta:
        model = Offer
        fields = ['period', 'agent', 'resource', 'amount', 'offerPrice', 'acceptance', 'acceptancePrice', 'acceptanceAmount']
