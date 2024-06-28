# db/models.py
from django.db import models
from django_enumfield import enum
from .enums import EnergyType
from .market import Market
from .agent import Agent
from .base import Base



# -------------------- MODELS ---------------------

# RESOURCE
class Resource(Base):
    energyType = enum.EnumField(EnergyType, null=False)
    name = models.CharField(max_length=20, null=False, unique=True)
    fuelCost = models.DecimalField(decimal_places=1, max_digits=5, null=False, default=0)
    emission = models.DecimalField(decimal_places=1, max_digits=5, null=False, default=0)


# PORTFOLIO
class Portfolio(Base):
    agent = models.OneToOneField(Agent, on_delete=models.CASCADE, null=False)


# PLANT
class Plant(Base):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, null=False)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, null=False)
    capacity = models.IntegerField(default=0, null=False)


# PERIOD
class Period(Base):
    market = models.ForeignKey(Market, on_delete=models.CASCADE, null=False)
    periodNumber = models.IntegerField(null=False)
    demand = models.IntegerField()
    metDemand = models.IntegerField()
    marketVolume = models.IntegerField()
    ptf = models.IntegerField()




from django.core.serializers.json import DjangoJSONEncoder
import json


# OFFER
class Offer(Base):
    period = models.ForeignKey(Period, on_delete=models.CASCADE, null=False)
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, null=False)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, null=False)
    amount = models.IntegerField(null=False)
    offerPrice = models.DecimalField(decimal_places=2, max_digits=7, null=False)
    acceptance = models.BooleanField(null=True)
    acceptancePrice = models.DecimalField(decimal_places=2, max_digits=7, null=True)
    acceptanceAmount = models.IntegerField(null=True)

    def to_dict(self):
        return {
        'id': self.id,
        'agent': self.agent.name,
        'resource': self.resource.name,
        'amount': self.amount,
        'offerPrice': str(self.offerPrice),
        'acceptance': self.acceptance,
        'acceptancePrice': str(self.acceptancePrice) if self.acceptancePrice is not None else None,
        'acceptanceAmount': self.acceptanceAmount,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), cls=DjangoJSONEncoder)
