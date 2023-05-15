from django.urls import path
from . import views
urlpatterns=[
    path('', views.home),
    path('registrarProducto/', views.registrarProducto),
    path('edicionProducto/<codigo>', views.edicionProducto),
    path('editarProducto/', views.editarProducto),
    path('eliminacionProducto/<codigo>', views.eliminacionProducto)
    
]
 