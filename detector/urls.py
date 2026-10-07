from __future__ import annotations

from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.home, name="home"),
    path("sample/<slug:slug>/", views.predict_sample, name="predict_sample"),
    path("diagnosis/<uuid:pk>/", views.prediction_detail, name="prediction_detail"),
    path("diagnosis/<uuid:pk>/delete/", views.prediction_delete, name="prediction_delete"),
    path("history/", views.history, name="history"),
    path("diseases/", views.diseases, name="diseases"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("model/", views.about, name="about"),
    # accounts
    path("accounts/login/", views.SignInView.as_view(), name="login"),
    path("accounts/logout/", views.signout, name="logout"),
    path("accounts/signup/", views.signup, name="signup"),
    # JSON API
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/stats/", views.api_stats, name="api_stats"),
    path("api/health/", views.api_health, name="api_health"),
]
