import sys
sys.path.append("D:/Projeler/abm/abmem_project/test")

from typing import Any, Iterable
from django_enumfield import enum
from .enums import AgentState , AgentType
from .base import Base
from .market import Market
from django.db import models


# AGENT
class Agent(Base):
    name = models.CharField(max_length=20,null=False)
    market = models.ForeignKey(Market,on_delete=models.CASCADE,null=True)
    state = enum.EnumField(AgentState,null=True,default=AgentState.CREATED)
    budget = models.DecimalField(decimal_places=2,max_digits=12,null=False,default=0)
    type = enum.EnumField(AgentType,null=True,default=AgentType.FOSSIL)
      

   
    

