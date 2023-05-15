from django.shortcuts import render, redirect
from .models import Producto

# Create your views here.

def home(request):
    ProductoListados = Producto.objects.all() 
    return render(request, "gestionGestion.html", {"Producto": ProductoListados})

def registrarProducto(request):
    codigo=request.POST['txtCodigo']
    nombre=request.POST['txtNombre']
    precio=request.POST['numPrecio']

    producto=Producto.objects.create(codigo=codigo, nombre=nombre, precio=precio)
    return redirect('/')