from django.contrib import admin
from .models.models import Resource, Portfolio, Plant, Period, Offer

# Modelleri yönetim arayüzüne kaydet
admin.site.register(Resource)
admin.site.register(Portfolio)
admin.site.register(Plant)
admin.site.register(Period)
admin.site.register(Offer)
