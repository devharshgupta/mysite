from django.urls import path

from . import views

urlpatterns = [
    path('ml', views.price, name='price'),
    path('', views.index, name='index'),

]
