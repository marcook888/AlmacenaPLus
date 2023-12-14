#app.py
from flask import Flask, request, session, redirect, url_for, render_template, flash
import psycopg2 
import pandas as pd
import psycopg2.extras
import re 
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import base64
from io import BytesIO
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.spatial import distance
from scipy.cluster import hierarchy



 
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
                    # Verificar si el registro ya existe en la base de datos
                    query_check = "SELECT * FROM productos WHERE codigo = %s"
                    cursor.execute(query_check, (row['codigo'],))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        # Si existe, actualizar los valores
                        query_update = "UPDATE productos SET nombre=%s, precio=%s, cantidad_vendida=%s, cantidad_stock=%s, categoria=%s, fecha_venta=%s WHERE codigo=%s"
                        values_update = (
                            row['nombre'],
                            row['precio'],
                            row['cantidad_vendida'],
                            row['cantidad_stock'],
                            row['categoria'],
                            row['fecha_venta'],
                            row['codigo']
                        )
                        cursor.execute(query_update, values_update)
                    else:
                        # Si no existe, realizar una inserción
                        query_insert = "INSERT INTO productos (codigo, nombre, precio, cantidad_vendida, cantidad_stock, categoria, fecha_venta) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                        values_insert = (
                            row['codigo'],
                            row['nombre'],
                            row['precio'],
                            row['cantidad_vendida'],
                            row['cantidad_stock'],
                            row['categoria'],
                            row['fecha_venta']
                        )
                        cursor.execute(query_insert, values_insert)

                conn.commit()
                conn.close()
                flash('Archivo subido y datos guardados o actualizados en la base de datos.')

                # Después de cargar los datos, redirigir a la página de dashboard
                return redirect(url_for('dashboard'))
            except Exception as e:
                flash(f'Error al procesar el archivo: {e}')
        else:
            flash('Formato de archivo no permitido.')

    return render_template('dashboard.html')


@app.route('/dashboard')
def dashboard():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Verificar si el usuario está autenticado
    if 'loggedin' in session:
        # Consulta para obtener todos los productos de la base de datos
        cursor.execute('SELECT * FROM productos')
        productos = cursor.fetchall()

        ###############################################KNN##################################################
        query = "SELECT nombre, precio, cantidad_vendida, cantidad_stock, categoria, fecha_venta FROM productos"
        df = pd.read_sql(query, conn)

        # Seleccionar características y etiqueta
        X = df[["precio", "cantidad_stock", "cantidad_vendida"]]  # características
        y = df["categoria"]  # etiqueta

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear instancia del modelo
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

        # Entrenar el modelo
        knn.fit(X_train, y_train)

        # Evaluar el modelo
        score = knn.score(X_test, y_test)  # precisión del modelo
  

        ####################################GRAFICA KNN###############################################################################################
        X = df[["precio", "cantidad_stock", "cantidad_vendida"]]
        y_pred = knn.predict(X)
        df["ventas_predichas"] = y_pred

        # Ordenar los productos por ventas predichas en orden ascendente (los menos favorables primero)
        df = df.sort_values(by="ventas_predichas", ascending=True)

        # Crear un gráfico de dispersión con colores por categoría y leyenda
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(df["precio"], df["cantidad_stock"], c=df["categoria"].factorize()[0], cmap="tab20")
        plt.xlabel("Precio")
        plt.ylabel("Cantidad en Stock")
        plt.title("Productos Menos Favorables por Categoría")

        # Agregar una leyenda que mapee los números de categoría a sus nombres
        handles, labels = scatter.legend_elements()
        categories = df["categoria"].unique()
        category_names = [f"{cat_num}: {cat_name}" for cat_num, cat_name in enumerate(categories)]
        plt.legend(handles, category_names, title="Categoría")

        # Guardar el gráfico en un archivo temporal
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        scatter_plot = base64.b64encode(img.read()).decode("utf-8")

        # Limpiar la figura actual para el siguiente gráfico
        plt.clf()
        print(f"La precisión del modelo es: {score}")
        print(f"Las etiquetas predichas son: {y_pred}")

        ##########################################RandomForestClassifier################################
       # Consulta SQL para extraer los datos
        query = "SELECT nombre, precio, cantidad_vendida, cantidad_stock, categoria, fecha_venta FROM productos"
        df_rf = pd.read_sql(query, conn)

        X = df_rf[['precio', 'cantidad_vendida', 'cantidad_stock']]
        y = df_rf['categoria']

        RF = RandomForestClassifier(n_estimators=100, max_depth=5)
        RF.fit(X, y)
        df_rf['categoria_predicha'] = RF.predict(X)

        df_rf['fecha_venta'] = pd.to_datetime(df_rf['fecha_venta'])

        df_rf['fecha_venta'] = pd.to_datetime(df_rf['fecha_venta'])

        # Crear una nueva columna para el mes
        df_rf['mes'] = df_rf['fecha_venta'].dt.strftime('%Y-%m')

        # Agrupar por mes y categoría predicha y contar la cantidad de productos en cada categoría
        sales_by_month = df_rf.groupby(['mes', 'categoria_predicha']).size().unstack(fill_value=0)

        # Crear un gráfico de líneas para cada categoría
        for category in sales_by_month.columns:
            plt.plot(sales_by_month.index, sales_by_month[category], label=category)

        # Configurar el gráfico
        plt.xlabel('Mes')
        plt.ylabel('Cantidad de Productos Vendidos')
        plt.title('Temporada de Ventas por Mes')
        plt.legend(loc='upper right')


        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graficoLinea = base64.b64encode(img.read()).decode("utf-8")

        #################################################DecisionTreeClassifier##################################################################
        # Preprocesamiento de datos: selecciona características y etiquetas
        query = "SELECT nombre, precio, cantidad_vendida, cantidad_stock, categoria, fecha_venta FROM productos"
        df_DTC = pd.read_sql(query, conn)
        df_DTC['ganancia'] = df_DTC['precio'] * df_DTC['cantidad_vendida']
        rentabilidad_promedio = df_DTC.groupby('categoria')['ganancia'].mean().reset_index()
       
        X = rentabilidad_promedio['ganancia'].values.reshape(-1, 1)
        y = rentabilidad_promedio['categoria']
        # Crear e entrenar el modelo DecisionTreeClassifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(X, y)
        categoria_mas_rentable = dt_classifier.predict([[rentabilidad_promedio['ganancia'].max()]])

        ########################################grafico#########################################################
        # Calcular la ganancia promedio por categoría
        ganancia_promedio_por_categoria = df_DTC.groupby('categoria')['ganancia'].mean()

        # Obtener la categoría más rentable según el modelo
        categoria_mas_rentable_predicha = categoria_mas_rentable[0]

        # Obtener la ganancia promedio de la categoría más rentable predicha por el modelo
        ganancia_promedio_categoria_mas_rentable = ganancia_promedio_por_categoria[categoria_mas_rentable_predicha]

        # Crear un DataFrame con la ganancia promedio de todas las categorías
        df_rentabilidad = pd.DataFrame({'Categoría': ganancia_promedio_por_categoria.index, 'Ganancia Promedio': ganancia_promedio_por_categoria})

        # Configurar el tamaño del gráfico
        plt.figure(figsize=(10, 6))

        # Crear el gráfico de barras
        plt.bar(df_rentabilidad['Categoría'], df_rentabilidad['Ganancia Promedio'])

        # Destacar la categoría más rentable predicha por el modelo
        plt.bar(categoria_mas_rentable_predicha, ganancia_promedio_categoria_mas_rentable, color='red', label='Categoría Más Rentable')

        # Etiquetas y título
        plt.xlabel('Categoría')
        plt.ylabel('Ganancia Promedio')
        plt.title('Ganancia Promedio por Categoría')

        # Mostrar el gráfico
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        grafico_rentable = base64.b64encode(img.read()).decode("utf-8")

        ###############################################MARCO###############################################################################

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
        


        ######################################### knn gerarquico ###########################################################

            # Query the database to retrieve the data you need for KNN hierarchical
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT nombre, precio, cantidad_vendida, cantidad_stock FROM productos")
        data = cursor.fetchall()

        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(data, columns=['nombre', 'precio', 'cantidad_vendida', 'cantidad_stock'])

        # Calculate the pairwise distance matrix
        pairwise_distances = distance.pdist(df[['precio', 'cantidad_vendida', 'cantidad_stock']])

        # Calculate the hierarchical clustering
        linkage = hierarchy.linkage(pairwise_distances, method='average')

        # Create a dendrogram
        plt.figure(figsize=(10, len(df) * 0.5))  # Ajusta el tamaño vertical de acuerdo a la cantidad de productos
        dendrogram = hierarchy.dendrogram(linkage, labels=df['nombre'].values, orientation='left', leaf_font_size=8)  # Reduce el tamaño de la fuente

        # Save the dendrogram as an image
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')  # Ajusta el tamaño de la figura para que quepa el contenido
        img.seek(0)
        dendrogram_image = base64.b64encode(img.read()).decode('utf-8')
        plt.clf()


        ######################################### clustering ###########################################################

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

        return render_template('dashboard.html', scatter_plot=scatter_plot, productos=productos, graficoLinea=graficoLinea, grafico_rentable=grafico_rentable,
                               grafico1=grafico1, grafico2=grafico2, grafico3=grafico3, grafico4=grafico4, grafico5=grafico5, dendrogram_image=dendrogram_image, grafico=grafico)

    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)

  
