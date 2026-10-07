from django.contrib import admin

from .models import AgentStepLog, AlgorithmDefault, HPOStudy, HPOTrial, PolicyCheckpoint
from .models.models import Offer, Period, Plant, Portfolio, Resource

# Register models with the admin site.
admin.site.register(Resource)
admin.site.register(Portfolio)
admin.site.register(Plant)
admin.site.register(Period)
admin.site.register(Offer)
admin.site.register(AgentStepLog)
admin.site.register(HPOStudy)
admin.site.register(HPOTrial)
admin.site.register(AlgorithmDefault)
admin.site.register(PolicyCheckpoint)
