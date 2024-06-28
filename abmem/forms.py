from django import forms
from .models.models import Resource, Portfolio, Plant, Period, Offer, Agent
from django.forms import inlineformset_factory

class ResourceForm(forms.ModelForm):
    class Meta:
        model = Resource
        fields = ['name', 'energyType',  'emission', 'fuelCost']


class AgentForm(forms.ModelForm):
    class Meta:
        model = Agent
        fields = ['budget', 'type','name','proxy']  # Agent modelindeki tüm alanları içerir

class PortfolioForm(forms.ModelForm):
    class Meta:
        model = Portfolio
        fields = []  # Herhangi bir alan alınmayacak

class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = ['resource', 'capacity']  # Buradaki fields Plant modelinde bulunan alanları temsil etmeli

# PlantFormSet, Portfolio ile Plant arasındaki ilişkiyi yönetmek için kullanılır
PlantFormSet = inlineformset_factory(Portfolio, Plant, form=PlantForm, extra=1, can_delete=True)

class PeriodForm(forms.ModelForm):
    class Meta:
        model = Period
        fields = ['market', 'periodNumber', 'demand', 'metDemand', 'marketVolume', 'ptf']

class OfferForm(forms.ModelForm):
    class Meta:
        model = Offer
        fields = ['period', 'agent', 'resource', 'amount', 'offerPrice', 'acceptance', 'acceptancePrice', 'acceptanceAmount']
