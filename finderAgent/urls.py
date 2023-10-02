from django.urls import path

from .views import *

urlpatterns = [
    path('findSims/', getNumberOfSimilars),
    path('addDoc/', addDoc),
]
