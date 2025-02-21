# db/models.py
import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")
from django.db import models
from django_enumfield import enum
from .enums import SimulationMode,SimulationState,PeriodType
from .base import Base
from typing import Any


# SIMULATION
class Simulation(Base):
    name = models.CharField(max_length=20,null=False)
    mode = enum.EnumField(SimulationMode,null=False)
    state = enum.EnumField(SimulationState,null=False,default=SimulationState.CREATED)
    periodType = enum.EnumField(PeriodType,null=False)
    periodNumber = models.IntegerField(null=False)
    currentPeriod = models.IntegerField(null=False)
    day = models.IntegerField(null=False)



    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.data = None


