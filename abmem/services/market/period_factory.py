
from ...models import Period

def create(market, num: int, demand: int) -> Period:
    period = Period(market= market, periodNumber= num, demand= demand, metDemand = -1,marketVolume = -1, ptf = -1)
    period.save()
    return period