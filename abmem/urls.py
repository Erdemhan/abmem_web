from django.urls import path
from . import views
from abmem.consumers import SimulationConsumer


urlpatterns = [
    path('resource/create', views.resource_create, name='resource_create'),
    path('resource/list', views.resource_list, name='resource_list'),
    path('resource/<int:resource_id>', views.resource_update, name='resource_update'),
    path('resource/delete/<int:resource_id>', views.resource_delete, name='resource_delete'),
    path('simulation/list-proxy', views.simulation_list_proxy, name='simulation_list_proxy'),
    path('simulation/delete/<int:simulation_id>', views.delete_simulation, name='simulation_delete'),
    path('simulation/list', views.simulation_list, name='simulation_list'),
    path('simulation/<int:sim_id>', views.simulation_view, name='simulation_view'),
    path('get_agents/', views.get_agents, name='get_agents'),
    path('agent/create', views.create_agent_and_related, name='agent_create'),
    path('agent/list', views.agent_list, name='agent_list'),
    path('agent/<int:agent_id>/', views.update_agent, name='agent_update'),
    path('agent/delete/<int:agent_id>/', views.delete_agent, name='agent_delete'),
    path('blank/', views.blank, name='blank'),
    path('start', views.start, name='start'),
    path('', views.dashboard, name='dashboard'),
    path('simulation/create/', views.create_simulation, name='create_simulation'),
    path('simulation/run/<int:simulation_id>/', views.run_simulation, name='run_simulation'),
    path('simulation/update/<int:simulation_id>/', views.update_simulation, name='update_simulation'),
    path('simulation/watch/<str:simulation_id>/', views.watch_simulation, name='watch_simulation'),
    path('simulation/status/', views.get_simulation_status, name='get_simulation_status'),
    path('simulation/next/', views.nextPeriod, name='nextPeriod'),
    path('ws/simulation/<str:simulation_id>/', SimulationConsumer.as_asgi()),
]

