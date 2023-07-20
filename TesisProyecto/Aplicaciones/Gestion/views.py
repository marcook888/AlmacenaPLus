from django.shortcuts import render, redirect
from .models import Producto
from django.contrib import messages

# Create your views here.

def home(request):
    ProductoListados = Producto.objects.all() 
    messages.success(request, '!Productos Listados!')
    return render(request, "gestionGestion.html", {"Producto": ProductoListados})

def registrarProducto(request):
    codigo=request.POST['txtCodigo']
    nombre=request.POST['txtNombre']
    precio=request.POST['numPrecio']

    producto=Producto.objects.create(codigo=codigo, nombre=nombre, precio=precio)
    messages.success(request, '!Producto Registrado!')
    return redirect('/')

def edicionProducto (request, codigo):
    producto=Producto.objects.get(codigo=codigo)
    return render (request, "edicionProducto.html", {"producto":producto})

def editarProducto(request):
    codigo=request.POST['txtCodigo']
    nombre=request.POST['txtNombre']
    precio=request.POST['numPrecio']

    producto=Producto.objects.get(codigo=codigo)
    producto.nombre=nombre
    producto.precio=precio
    producto.save()
    messages.success(request, '!Producto Actualizado!')

    return redirect('/')

def eliminacionProducto(request, codigo):
    producto=Producto.objects.get(codigo=codigo)
    producto.delete()
    messages.success(request, '!Producto Eliminado!')
    return redirect('/')

def dashboard(request):
    return render(request, 'dashboard.html')