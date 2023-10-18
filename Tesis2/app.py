#app.py
from flask import Flask, request, session, redirect, url_for, render_template, flash
import psycopg2 #pip install psycopg2 
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
import io
from sklearn.metrics import roc_curve, roc_auc_score, classification_report



 
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
        data = cursor.fetchall()

        cursor2 = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor2.execute('SELECT categoria, SUM(cantidad_vendida) as total_ventas FROM productos GROUP BY categoria')
        results = cursor2.fetchall()
        # Preprocesar los datos (codificar la columna "categoria")
        label_encoder = LabelEncoder()
        data['categoria'] = label_encoder.fit_transform(data['categoria'])

        feature=data[['precio', 'cantidad_stock', 'categoria']]
        # Dividir los datos en características (X) y etiquetas (y)
        X = feature
        y = data['cantidad_vendida']  # Puedes cambiar la columna de destino según la clasificación deseada

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        sc.fit(X_train)

        # Entrenar diferentes modelos
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)

        rf_model = RandomForestClassifier(n_estimators=50,max_depth=10, criterion='gini')
        rf_model.fit(X_train, y_train)
       
        multiclass_model = DecisionTreeClassifier()
        multiclass_model.fit(X_train, y_train)
    
        # Calcular la precisión de los modelos
        knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        multiclass_accuracy = accuracy_score(y_test, multiclass_model.predict(X_test))

        # Seleccionar el modelo con la mejor precisión para cada clasificación
        best_model = None
        if knn_accuracy >= rf_accuracy and knn_accuracy >= multiclass_accuracy:
            best_model = knn_model
        elif rf_accuracy >= knn_accuracy and rf_accuracy >= multiclass_accuracy:
            best_model = rf_model
        else:
            best_model = multiclass_model
        # Realizar predicciones con el mejor modelo
        predictions = best_model.predict(X_test)
     # Calcular las probabilidades de predicción del mejor modelo
        probs = best_model.predict_proba(X_test)
        # Calcular la Curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
        # Calcular el Área bajo la Curva ROC (AUC)
        roc_auc = roc_auc_score(y_test, probs[:, 1])

###################### gráfico de la Curva ROC
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")

        # Guardar la imagen de la Curva ROC
        plt.savefig('static/curva_roc.png')

        # Obtener los Classification Reports
        knn_report = classification_report(y_test, knn_model.predict(X_test))
        rf_report = classification_report(y_test, rf_model.predict(X_test))
        multiclass_report = classification_report(y_test, multiclass_model.predict(X_test))

        #########################grafico de barras######################
        categories = [row['categoria'] for row in results]
        total_sales = [row['total_ventas'] for row in results]

        plt.bar(categories, total_sales)
        plt.xlabel('Categoría')
        plt.ylabel('Ventas Totales')
        plt.title('Ventas por Categoría')
        
        # Guardar el gráfico en un objeto BytesIO
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        
        # Codificar la imagen en base64
        barras = base64.b64encode(image_stream.read()).decode("utf-8")

        #################gráfico de líneas de tendencia de ventas mensuales######################
        dates = [row['fecha_venta'] for row in data]
        total_sales = [row['total_ventas'] for row in data]

        plt.plot(dates, total_sales, marker='o')
        plt.xlabel('Fecha de Venta')
        plt.ylabel('Ventas Mensuales')
        plt.title('Tendencia de Ventas Mensuales')

        # Guardar el gráfico en un objeto BytesIO
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        
        # Codificar la imagen en base64
        tendencia = base64.b64encode(image_stream.read()).decode("utf-8")

        ###########################gráfico de dispersión de precio vs. cantidad vendida###########################################
        prices = [row['precio'] for row in results]
        quantities = [row['cantidad_vendida'] for row in results]

        plt.scatter(prices, quantities)
        plt.xlabel('Precio')
        plt.ylabel('Cantidad Vendida')
        plt.title('Precio vs. Cantidad Vendida')

        # Guardar el gráfico en un objeto BytesIO
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        
        # Codificar la imagen en base64
        PrecioCantidad = base64.b64encode(image_stream.read()).decode("utf-8")


        return render_template('dashboard.html', productos=productos,knn_report=knn_report, rf_report=rf_report, 
                               multiclass_report=multiclass_report,barras=barras,tendencia=tendencia,PrecioCantidad=PrecioCantidad)
    
    # Si el usuario no está autenticado, redirigir a la página de inicio de sesión
    return redirect(url_for('login'))

"""
    #Graficas
@app.route('/barras')
# Función para generar el gráfico de barras de ventas por categoría
def generate_category_sales_bar_chart():
    cursor2 = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor2.execute('SELECT categoria, SUM(cantidad_vendida) as total_ventas FROM productos GROUP BY categoria')
    results = cursor2.fetchall()

    categories = [row['categoria'] for row in results]
    total_sales = [row['total_ventas'] for row in results]

    plt.bar(categories, total_sales)
    plt.xlabel('Categoría')
    plt.ylabel('Ventas Totales')
    plt.title('Ventas por Categoría')
    
    # Guardar el gráfico en un objeto BytesIO
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    
    # Codificar la imagen en base64
    barras = base64.b64encode(image_stream.read()).decode("utf-8")
    
    return render_template('barras.html', barras=barras)

@app.route('/tendencia')
# Función para generar el gráfico de líneas de tendencia de ventas mensuales
def generate_monthly_sales_line_chart():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT fecha_venta, SUM(cantidad_vendida) as total_ventas FROM productos GROUP BY fecha_venta')
    results = cursor.fetchall()

    dates = [row['fecha_venta'] for row in results]
    total_sales = [row['total_ventas'] for row in results]

    plt.plot(dates, total_sales, marker='o')
    plt.xlabel('Fecha de Venta')
    plt.ylabel('Ventas Mensuales')
    plt.title('Tendencia de Ventas Mensuales')

    # Guardar el gráfico en un objeto BytesIO
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    
    # Codificar la imagen en base64
    tendencia = base64.b64encode(image_stream.read()).decode("utf-8")
    
    return render_template('tendencia.html', tendencia=tendencia)


@app.route('/praciocantidad')
# Función para generar el gráfico de dispersión de precio vs. cantidad vendida
def generate_price_vs_quantity_scatter_plot():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT precio, cantidad_vendida FROM productos')
    results = cursor.fetchall()

    prices = [row['precio'] for row in results]
    quantities = [row['cantidad_vendida'] for row in results]

    plt.scatter(prices, quantities)
    plt.xlabel('Precio')
    plt.ylabel('Cantidad Vendida')
    plt.title('Precio vs. Cantidad Vendida')

    # Guardar el gráfico en un objeto BytesIO
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    
    # Codificar la imagen en base64
    PrecioCantidad = base64.b64encode(image_stream.read()).decode("utf-8")

    return render_template('precioCantidad.html', PrecioCantidad=PrecioCantidad)

# Ruta para mostrar los gráficos en la página de dashboard.html
@app.route('/charts')
def show_charts():
    category_sales_chart = generate_category_sales_bar_chart()
    monthly_sales_chart = generate_monthly_sales_line_chart()
    price_quantity_chart = generate_price_vs_quantity_scatter_plot()

    return render_template('dashboard.html', category_sales_chart=category_sales_chart,
                           monthly_sales_chart=monthly_sales_chart, price_quantity_chart=price_quantity_chart)

"""
 
if __name__ == "__main__":
    app.run(debug=True)