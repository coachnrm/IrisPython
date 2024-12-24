from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    result = ''
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        septal_length = float(request.form['septal_length'])
        septal_width = float(request.form['septal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        loadded_model = joblib.load('iris-model.sav')
        result = loadded_model.predict([[septal_length, septal_width, petal_length, petal_width]])[0]
        return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()
