from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("prune/", views.prune_model_view, name="prune"),
    path("job_status/<int:job_id>/", views.job_status, name="job_status"),
    path("result/", views.result_view, name="result"),
]
