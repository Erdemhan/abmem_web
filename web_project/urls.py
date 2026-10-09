from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="/dev/", permanent=False)),
    path("dev/", include("abmem.urls")),
    path("admin/", admin.site.urls),
]
