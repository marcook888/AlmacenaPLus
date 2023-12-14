from django.urls import path
from .views import home, datos, exit, register


urlpatterns = [
    path('', home, name='home'),
    path('datos/', datos, name='datos'), 
    path('logout/', exit, name='exit'),
    path('register/', register, name='register'),
]