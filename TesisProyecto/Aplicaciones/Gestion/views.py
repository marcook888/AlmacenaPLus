from django.shortcuts import render, redirect
from .models import Producto
from django.contrib import messages
from django.core.files.storage import FileSystemStorage

import psycopg2
import pandas as pd

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

def compras(request):
    return render(request, 'compras.html')


def archivos(request):
    if connection() != None:
        cur2 = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur2.execute("SELECT * FROM archivos WHERE usuario = '"+request.user.username+"' ORDER BY fecha DESC")
        datos = cur2.fetchall()
    
    context = {"datos": datos, "segment": 'archivos'}
    return context


def abrir_archivo(request, id):
    if connection() != None:
        cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
        cur.execute("SELECT archivo FROM archivos al WHERE idarch='"+str(id)+"'")
        datos = cur.fetchall()
        nom_archivo = "".join(datos[0])

        if '.csv' in nom_archivo:
            df = pd.read_csv('data/'+request.user.username+'--'+nom_archivo, sep=';')
        elif '.xls' in nom_archivo or '.xlsx' in nom_archivo:
            df = pd.read_excel('data/'+request.user.username+'--'+nom_archivo)
        elif '.txt' in nom_archivo:
            df = pd.read_csv('data/'+request.user.username+'--'+nom_archivo, delimiter='\t')
        datos = df.values.tolist()
        columns = df.columns

    context = {"datos": datos, "nom_arch": nom_archivo, "columns": columns}
    return render(request, '../templates/home/abrir_archivo.html', context=context)


def subir_csv(request):
    datos=""
    if request.method == 'POST' and request.FILES['myFile']:
        usuario = request.POST['usuario']
        myFile = request.FILES['myFile']

        fs = FileSystemStorage()
        if not fs.exists(usuario + '--' + myFile.name):
            filename = fs.save(usuario + '--' + myFile.name, myFile)
            uploaded_file_url = fs.url(filename)
            if connection() != None:
                cur = connection().cursor(cursor_factory= psycopg2.extras.DictCursor)
                cur.execute("""
                    INSERT INTO archivos (usuario, archivo, fecha)
                    VALUES ('"""+usuario+"""', '"""+filename.split(usuario + '--')[-1]+"""', current_timestamp)
                """)

            if '.csv' in filename:
                df = pd.read_csv(filename, sep=';')
            elif '.xls' in filename or '.xlsx' in filename:
                df = pd.read_excel(filename)
            elif '.txt' in filename:
                df = pd.read_csv(filename, delimiter='\t')
            datos = df.head(100).values.tolist()
            columns = df.columns
        else:
            filename = fs.delete(usuario + '--' + myFile.name)
            filename = fs.save(usuario + '--' + myFile.name, myFile)
            if '.csv' in myFile.name:
                df = pd.read_csv(usuario+'--'+myFile.name, sep=';')
            elif '.xls' in myFile.name or '.xlsx' in myFile.name:
                df = pd.read_excel(usuario+'--'+myFile.name)
            elif '.txt' in myFile.name:
                df = pd.read_csv(usuario+'--'+myFile.name, delimiter='\t')
            datos = df.head(100).values.tolist()
            columns = df.columns
            for col in columns:
                print(df[col].dtype)

    context = {"datos": datos, "nom_arch": myFile.name+" (Primeras 100 filas...)", "columns": columns}
    return render(request, '../templates/dashboard.html', context=context)



def connection():
    hostname = 'localhost'
    database = 'postgres'
    username = 'postgres'
    pwd = '12345'
    port_id = 5432

    conn = None

    try:
        conn = psycopg2.connect (
            host = hostname,
            database = database,
            user = username,
            password = pwd,
            port = port_id
        )
        conn.set_session(autocommit=True)
        
    except Exception as error:
        print('Mensaje de error: ', error)
    
    return conn