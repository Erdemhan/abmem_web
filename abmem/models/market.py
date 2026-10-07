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
    # Per-scenario overrides of the clearing algorithm parameters, applied
    # over ClearingDefault.params (PLAN 2.12).
    clearing_params = models.JSONField(default=dict, blank=True)
    # Parameters the clearing algorithm of a run actually used, written once
    # when the run starts. NULL on scenarios and on older runs.
    effective_clearing_params = models.JSONField(null=True, blank=True, default=None)
    simulation = models.OneToOneField(Simulation,on_delete=models.CASCADE,null=False)
    

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.data = None
        


    