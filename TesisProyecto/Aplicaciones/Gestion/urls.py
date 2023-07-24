from django.urls import path
from . import views
urlpatterns=[
    path('', views.home),
    path('registrarProducto/', views.registrarProducto),
    path('edicionProducto/<codigo>', views.edicionProducto),
    path('editarProducto/', views.editarProducto),
    path('eliminacionProducto/<codigo>', views.eliminacionProducto),
    path('dashboard/', views.dashboard),
    path('compras/', views.compras),
    path('subir_csv/', views.subir_csv, name='subir_csv')
    
]
 