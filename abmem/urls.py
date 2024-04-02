from django.urls import path
from . import views

urlpatterns = [
    path('resource/', views.resource_create, name='resource')
]

