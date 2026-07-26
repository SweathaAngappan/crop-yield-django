from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_view, name='predict'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('model/', views.model_view, name='model'),
]