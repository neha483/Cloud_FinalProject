from flask import Flask, request, render_template, redirect, session, url_for
import os
import mysql.connector
import pandas as pd
from flask_session import Session
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
import lightgbm as lgb
import traceback
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

def get_db_connection():
    try:
        conn = mysql.connector.connect(**config)
        return conn, None
    except mysql.connector.Error as err:
        return None, str(err)

def fetch_data(query, params=None):
    conn, err = get_db_connection()
    if err or conn is None:
        print(f"DB Connection Error: {err}")
        return None
    try:
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        print(f"SQL Read Error: {e}")
        return None
    finally:
        conn.close()

@app.route('/', methods=['GET', 'POST'])
def homepage():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        df = fetch_data("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        if df is not None and not df.empty:
            session['loggedin'] = True
            session['username'] = username
            return redirect(url_for('profile', username=username))
        msg = 'Incorrect username or password.'
    return render_template("homepage.html", msg=msg)

@app.route('/profile/<string:username>')
def profile(username):
    if 'loggedin' in session:
        return render_template("profile.html", username=username)
    return redirect(url_for('homepage'))

@app.route('/predict_clv')
def predict_clv():
    try:
        df = fetch_data("SELECT * FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM")
        if df is None or df.empty:
            return "<h4>No data available for prediction.</h4>"

        if 'PURCHASE_' not in df.columns or 'SPEND' not in df.columns:
            return "<h4>Required columns are missing from the data.</h4>"

        df['SPEND'] = pd.to_numeric(df['SPEND'], errors='coerce')
        df = df.dropna(subset=['SPEND'])

        rfm = df.groupby('HSHD_NUM').agg({
            'PURCHASE_': 'max',
            'BASKET_NUM': 'count',
            'SPEND': 'sum'
        }).rename(columns={
            'PURCHASE_': 'Recency',
            'BASKET_NUM': 'Frequency',
            'SPEND': 'Monetary'
        }).reset_index()

        household_info = df[['HSHD_NUM', 'AGE_RANGE', 'INCOME_RANGE', 'HH_SIZE']].drop_duplicates()
        merged = pd.merge(rfm, household_info, on='HSHD_NUM', how='left')

        merged = merged.dropna()
        merged = pd.get_dummies(merged, columns=['AGE_RANGE', 'INCOME_RANGE'])

        X = merged.drop(columns=['Monetary', 'HSHD_NUM'])
        y = merged['Monetary']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        return f"<h4>CLV Prediction complete. Mean Absolute Error: {mae:.2f}</h4>"
    except Exception as e:
        traceback.print_exc()
        return f"<h4>Error during prediction: {e}</h4>"

if __name__ == '__main__':
    app.run(debug=True)
