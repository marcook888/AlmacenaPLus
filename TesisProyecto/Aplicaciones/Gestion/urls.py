from django.urls import path
from . import views
from .views import home, exit, login,dashboard
urlpatterns=[
    path('', home, name='home'),
    path('registrarProducto/', views.registrarProducto),
    path('edicionProducto/<codigo>', views.edicionProducto),
    path('editarProducto/', views.editarProducto),
    path('eliminacionProducto/<codigo>', views.eliminacionProducto),
    path('dashboard/', dashboard ,name='dashboard'),
    path('compras/', views.compras),
    path('subir_csv/', views.subir_csv, name='subir_csv'),
    path('logout/', exit, name='exit'),
    path('login/', login, name='login' )

    
]
 