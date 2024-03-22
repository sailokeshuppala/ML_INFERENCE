from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/Logistic Regression.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = model.predict([message])[0]
        return render_template('result.html', prediction=prediction)

@app.route('/prediction')
def prediction():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)