import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request, render_template
import pickle
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress the version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

application = Flask(__name__)
app = application

model = pickle.load(open('model/linear_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    
## Index page - where our app is start with
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predictdata():
    name = request.form.get('name')
    return render_template('predict.html', myName=name)

@app.route('/predictdata/predict_result', methods=['GET', 'POST'])
def predict_result():
    if request.method == "POST":
        temprature = float(request.form.get('temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        region = float(request.form.get('region'))

        new_scaled_data = scaler.transform([[temprature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, region]])
        result = model.predict(new_scaled_data)
        return render_template('predict.html', model_prediction= result[0])
    else:
        return render_template('predict.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')