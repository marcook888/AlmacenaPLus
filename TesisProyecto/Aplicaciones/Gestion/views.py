from django.shortcuts import render
from .models import Producto

# Create your views here.

def home(request):
    ProductoListados = Producto.objects.all() 
    return render(request, "gestionGestion.html", {"Producto": ProductoListados})