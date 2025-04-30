from flask import Flask, request, render_template, redirect, session, url_for
import os
import re
import mysql.connector
import pandas as pd
from mysql.connector import errorcode
from flask_session import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = '@dkjgfjgfhkj jxbjljv kjxgvljklkj'

# Configurations and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, 'static', 'files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

config = {
    'host': 'krogerdatabase.mysql.database.azure.com',
    'user': 'ccfinal',
    'password': 'Ccfinal123',
    'database': 'cckroger',
    'port': 3306,
    'ssl_ca': os.path.join(dir_path, 'ssl', 'BaltimoreCyberTrustRoot.crt.pem'),
    'connect_timeout': 50000
}

# Database helper functions
def get_db_connection():
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        return None, str(err)

def execute_query(query, params=None):
    conn, error = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor
    return None, error

def fetch_data_from_db(query, params=None):
    conn, error = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    return None, error

def clean_data(df, columns_to_strip):
    df.columns = df.columns.str.strip()
    return df.dropna(subset=columns_to_strip)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username, password = request.form['username'], request.form['password']
        query = 'SELECT * FROM users WHERE username = %s AND password = %s'
        user = fetch_data_from_db(query, (username, password))
        if user:
            session['loggedin'] = True
            session['username'] = username
            return redirect(url_for('profile', username=username))
        msg = 'Incorrect username/password!'
    return render_template("homepage.html", msg=msg)


@app.route('/profile/<string:username>', methods=['GET', 'POST'])
def profile(username):
    if 'loggedin' in session:
        return render_template('profile.html', username=username)
    return redirect(url_for('homepage'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = {
            'households': request.files['households'],
            'transactions': request.files['transactions'],
            'products': request.files['products']
        }
        msg = handle_file_upload(files)
        return render_template("upload.html", msg=msg)
    return render_template("upload.html")


def handle_file_upload(files):
    conn, error = get_db_connection()
    if not conn:
        return f"Database connection error: {error}"

    try:
        for key, file in files.items():
            if file.filename == '':
                return f"No file selected for {key}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            if key == 'households':
                upload_household_data(file_path, conn)
            elif key == 'transactions':
                upload_transaction_data(file_path, conn)
            elif key == 'products':
                upload_product_data(file_path, conn)

        return 'Data uploaded successfully!'
    except Exception as e:
        return f"Error: {e}"


def upload_household_data(file_path, conn):
    col_names = ['HSHD_NUM', 'L', 'AGE_RANGE', 'MARITAL', 'INCOME_RANGE', 'HOMEOWNER', 'HSHD_COMPOSITION', 'HH_SIZE', 'CHILDREN']
    data = pd.read_csv(file_path, names=col_names, header=0)
    query = """
    INSERT INTO households (HSHD_NUM, L, AGE_RANGE, MARITAL, INCOME_RANGE, HOMEOWNER, HSHD_COMPOSITION, HH_SIZE, CHILDREN)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for _, row in data.iterrows():
        cur = conn.cursor()
        cur.execute(query, tuple(row))
    conn.commit()


def upload_transaction_data(file_path, conn):
    col_names = ['BASKET_NUM', 'HSHD_NUM', 'PURCHASE_', 'PRODUCT_NUM', 'SPEND', 'UNITS', 'STORE_R', 'WEEK_NUM', 'YEAR']
    data = pd.read_csv(file_path, names=col_names, header=0)
    query = """
    INSERT INTO transactions (BASKET_NUM, PURCHASE_, SPEND, UNITS, STORE_R, WEEK_NUM, YEAR, HSHD_NUM, PRODUCT_NUM)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for _, row in data.iterrows():
        cur = conn.cursor()
        cur.execute(query, tuple(row))
    conn.commit()


def upload_product_data(file_path, conn):
    col_names = ['PRODUCT_NUM', 'DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']
    data = pd.read_csv(file_path, names=col_names, header=0)
    query = """
    INSERT INTO products (PRODUCT_NUM, DEPARTMENT, COMMODITY, BRAND_TY, NATURAL_ORGANIC_FLAG)
    VALUES (%s, %s, %s, %s, %s)
    """
    for _, row in data.iterrows():
        cur = conn.cursor()
        cur.execute(query, tuple(row))
    conn.commit()


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Fetch and clean data
        transactions = fetch_data_from_db("SELECT * FROM transactions")
        products = fetch_data_from_db("SELECT * FROM products")
        if not transactions or not products:
            return "<h4>No data found!</h4>"

        # Clean Data
        transactions = clean_data(transactions, ['SPEND'])
        products = clean_data(products, ['PRODUCT_NUM'])

        # Merge tables
        merged = pd.merge(transactions, products, on='PRODUCT_NUM', how='left')

        if merged.empty:
            return "<h4>⚠️ Merged data is empty!</h4>"

        # Further processing and prediction logic...
    except Exception as e:
        return f"<h4>⚠️ Error: {e}</h4>"

