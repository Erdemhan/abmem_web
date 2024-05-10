from django.urls import path
from . import views

urlpatterns = [
    path('resource/create', views.resource_create, name='resource_create'),
    path('resource/list', views.resource_list, name='resource_list'),
    path('resource/<int:resource_id>', views.resource_update, name='resource_update'),
    path('resource/delete/<int:resource_id>', views.resource_delete, name='resource_delete'),
    path('simulation/create', views.simulation_create, name='simulation_create'),
    path('simulation/list', views.simulation_list, name='simulation_list'),
    path('simulation/<int:sim_id>', views.simulation_view, name='simulation_view'),
    path('agent/create', views.create_agent_and_related, name='agent_create'),
    path('blank/', views.blank, name='blank'),
    path('start', views.start, name='start'),
    path('', views.dashboard, name='dashboard'),
]

