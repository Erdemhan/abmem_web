"""URL patterns of the hyperparameter search UI (ARCHITECTURE.md 8, prefix /dev/hpo/)."""

from django.urls import URLPattern, path

from . import views_hpo

urlpatterns: list[URLPattern] = [
    path("", views_hpo.study_list, name="hpo_list"),
    path("create/", views_hpo.study_create, name="hpo_create"),
    path("param-space/", views_hpo.param_space, name="hpo_param_space"),
    path("<int:study_id>/", views_hpo.study_watch, name="hpo_watch"),
    path("<int:study_id>/status/", views_hpo.study_status, name="hpo_status"),
    path("<int:study_id>/cancel/", views_hpo.study_cancel, name="hpo_cancel"),
    path("<int:study_id>/apply/", views_hpo.study_apply, name="hpo_apply"),
]
