from django_enumfield import enum
from typing import Any
from .enums import *
from .simulation import Simulation
from .base import Base
from django.db import models


# MARKET
class Market(Base):
    strategy = enum.EnumField(MarketStrategy,null=False)
    state = enum.EnumField(MarketState,null=False,default=MarketState.CREATED)
    lowerBidBound = models.IntegerField()
    upperBidBound = models.IntegerField()
    # Key of a registered clearing algorithm
    # (services/market/clearing_algorithms/registry.py). The settlement rule
    # stays in ``strategy``.
    clearing_algorithm = models.CharField(max_length=64, default="MERIT_ORDER")
    simulation = models.OneToOneField(Simulation,on_delete=models.CASCADE,null=False)
    

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.data = None
        


    