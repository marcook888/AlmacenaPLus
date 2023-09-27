#app.py
from flask import Flask, request, session, redirect, url_for, render_template, flash
import psycopg2 #pip install psycopg2 
import pandas as pd
import psycopg2.extras
import re 
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



app = Flask(__name__)
app.secret_key = 'cairocoders-ednalan'
 
DB_HOST = "localhost"
DB_NAME = "sampledb"
DB_USER = "postgres"
DB_PASS = "12345"
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
 
@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
    
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
 
@app.route('/login/', methods=['GET', 'POST'])
def login():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        print(password)
 
        # Check if account exists using MySQL
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        # Fetch one record and return result
        account = cursor.fetchone()
 
        if account:
            password_rs = account['password']
            print(password_rs)
            # If account exists in users table in out database
            if check_password_hash(password_rs, password):
                # Create session data, we can access this data in other routes
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                # Redirect to home page
                return redirect(url_for('home'))
            else:
                # Account doesnt exist or username/password incorrect
                flash('Incorrect username/password')
        else:
            # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password')
 
    return render_template('login.html')
  
@app.route('/register/', methods=['GET', 'POST'])
def register():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
 
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        nit = request.form['nit']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
    
        _hashed_password = generate_password_hash(password)
 
        #Check if account exists using MySQL
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        print(account)
        # If account exists show error and validation checks
        if account:
            flash('Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers!')
        elif not username or not password or not email:
            flash('Please fill out the form!')
        else:
            # Account doesnt exists and the form data is valid, now insert new account into users table
            cursor.execute("INSERT INTO users (nit, username, password, email) VALUES (%s,%s,%s,%s)", (nit, username, _hashed_password, email))
            conn.commit()
            flash('You have successfully registered!')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('Please fill out the form!')
    # Show registration form with message (if any)
    return render_template('register.html')
   
   
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
  
@app.route('/profile')
def profile(): 
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM users WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'txt'}


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and allowed_file(uploaded_file.filename):
            try:
                df = pd.read_csv(uploaded_file)
                conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
                cursor = conn.cursor()

                for index, row in df.iterrows():
                    query = "INSERT INTO productos (codigo, nombre, precio, cantidad_vendida, cantidad_stock, categoria, fecha_venta) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    values = (
                        row['codigo'],
                        row['nombre'],
                        row['precio'],
                        row['cantidad_vendida'],
                        row['cantidad_stock'],
                        row['categoria'],
                        row['fecha_venta']
                    )
                    cursor.execute(query, values)

                conn.commit()
                conn.close()
                flash('Archivo subido y datos guardados en la base de datos.')
            except Exception as e:
                flash(f'Error al procesar el archivo: {e}')
        else:
            flash('Formato de archivo no permitido.')

    return render_template('home.html')




############################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'txt'}

@app.route('/Indicadores')
def indicadores():
    # Consulta SQL para obtener datos para los gráficos
    query = "SELECT nombre, precio, cantidad_vendida, categoria, fecha_venta FROM productos"
    
    # Leer datos en un DataFrame de pandas
    df = pd.read_sql(query, conn)
    
    # Gráfico de barras para la cantidad vendida por producto
    plt.figure(figsize=(12, 6))
    plt.bar(df['nombre'], df['cantidad_vendida'])
    plt.xlabel('Producto')
    plt.ylabel('Cantidad Vendida')
    plt.title('Cantidad Vendida por Producto')
    plt.xticks(rotation=90)
    
    # Guardar el gráfico en un archivo temporal
    img1 = BytesIO()
    plt.savefig(img1, format="png")
    img1.seek(0)
    grafico1 = base64.b64encode(img1.read()).decode("utf-8")
    
    # Limpiar la figura actual para el siguiente gráfico
    plt.clf()
    
    # Gráfico de barras para el precio por producto
    plt.figure(figsize=(12, 6))
    plt.bar(df['nombre'], df['precio'])
    plt.xlabel('Producto')
    plt.ylabel('Precio')
    plt.title('Precio por Producto')
    plt.xticks(rotation=90)
    
    # Guardar el gráfico en un archivo temporal
    img2 = BytesIO()
    plt.savefig(img2, format="png")
    img2.seek(0)
    grafico2 = base64.b64encode(img2.read()).decode("utf-8")
    
    # Limpiar la figura actual para el siguiente gráfico
    plt.clf()
    
    # Gráfico de barras apiladas para la cantidad vendida por categoría
    categoria_cantidad = df.groupby('categoria')['cantidad_vendida'].sum()
    categoria_cantidad.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad Vendida')
    plt.title('Cantidad Vendida por Categoría')
    
    # Guardar el gráfico en un archivo temporal
    img3 = BytesIO()
    plt.savefig(img3, format="png")
    img3.seek(0)
    grafico3 = base64.b64encode(img3.read()).decode("utf-8")
    
    # Limpiar la figura actual para el siguiente gráfico
    plt.clf()
    
    # Gráfico de pastel para la distribución de productos por categoría
    categoria_cantidad = df['categoria'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(categoria_cantidad, labels=categoria_cantidad.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribución de Productos por Categoría')

    # Guardar el gráfico en un archivo temporal
    img4 = BytesIO()
    plt.savefig(img4, format="png")
    img4.seek(0)
    grafico4 = base64.b64encode(img4.read()).decode("utf-8")

    # Limpiar la figura actual para el siguiente gráfico
    plt.clf()
    
    # Gráfico de dispersión para la relación entre precio y cantidad vendida
    plt.figure(figsize=(8, 6))
    plt.scatter(df['precio'], df['cantidad_vendida'], alpha=0.5)
    plt.xlabel('Precio')
    plt.ylabel('Cantidad Vendida')
    plt.title('Relación entre Precio y Cantidad Vendida')
    
    # Guardar el gráfico en un archivo temporal
    img5 = BytesIO()
    plt.savefig(img5, format="png")
    img5.seek(0)
    grafico5 = base64.b64encode(img5.read()).decode("utf-8")
    

    
    # Renderizar la plantilla HTML con los gráficos
    return render_template('indicadores.html', grafico1=grafico1, grafico2=grafico2, grafico3=grafico3, grafico4=grafico4, grafico5=grafico5)



############################################

@app.route('/clustering')
def clustering():
    # Consulta SQL para obtener los datos necesarios para el clustering
    query = "SELECT nombre, precio, cantidad_vendida, cantidad_stock FROM productos"
    
    # Leer datos en un DataFrame de pandas
    df = pd.read_sql(query, conn)
    
    # Selecciona las columnas relevantes para el clustering
    X = df[['precio', 'cantidad_vendida', 'cantidad_stock']]
    
    # Normalizar los datos para que todas las columnas tengan la misma escala
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Realizar el clustering (por ejemplo, con K-Means)
    kmeans = KMeans(n_clusters=3, random_state=42)  # Número de clústeres a determinar
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Reducir la dimensionalidad para visualización (puedes ajustar esto según tus necesidades)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    # Mapeo de nombres de productos a nombres de clusters
    cluster_names = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Crear el gráfico de clustering
    plt.figure(figsize=(10, 6))
    
    # Colorear los puntos según los nombres de los clusters
    scatter = plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis')
    
    # Dibujar círculos alrededor de los puntos de cada cluster
    for cluster_id, cluster_name in cluster_names.items():
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_center = (cluster_data['pca1'].mean(), cluster_data['pca2'].mean())
        max_distance = max(cluster_data.apply(lambda row: ((row['pca1'] - cluster_center[0])**2 + (row['pca2'] - cluster_center[1])**2)**0.5, axis=1))
        
        # Colorear el círculo con el mismo color que el cluster
        circle = plt.Circle(cluster_center, max_distance + 0.1, fill=False, color=scatter.to_rgba(cluster_id), linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)
        
        # Etiquetar el centroide del cluster con el nombre del cluster
        plt.annotate(cluster_name, (cluster_center[0], cluster_center[1]), color=scatter.to_rgba(cluster_id), weight='bold',
                     fontsize=12, ha='center', va='center', backgroundcolor='white', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    plt.xlabel('Precio')
    plt.ylabel('Cantidad Vendida')
    plt.title('Clustering de Productos')
    
    # Guardar el gráfico en un archivo temporal
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    grafico = base64.b64encode(img.read()).decode("utf-8")

    # Limpiar la figura actual para el siguiente gráfico
    plt.clf()
    
    # Renderizar la plantilla HTML con el gráfico de clustering
    return render_template('clustering.html', grafico=grafico)

if __name__ == "__main__":
    app.run(debug=True)