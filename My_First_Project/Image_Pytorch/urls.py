from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('Pytorch/', Sub, name='Sub'),
    path('', handle, name='handle'),
]
