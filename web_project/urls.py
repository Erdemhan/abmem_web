from django.urls import path,include
from django.contrib import admin




urlpatterns = [
    path('dev/', include( 'abmem.urls')),
    path('admin/', admin.site.urls)
]
