import numpy as np
from flask import Flask, request, render_template
import pickle
import os
from pathlib import Path

app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), '../static'),
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'))

# prediction function 
def ValuePredictor(to_predict_list): 
    model_path = os.path.join(os.path.dirname(__file__), '../model.pkl')
    to_predict = np.array(to_predict_list).reshape(1, 7)
    loaded_model = pickle.load(open(model_path, "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]     
    
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
    if int(result) == 1:
        prediction = 'Given transaction is fraudulent'
    else:
        prediction = 'Given transaction is NOT fraudulent'            
    return render_template("result.html", prediction=prediction) 

# Vercel serverless handler
from werkzeug.serving import WSGIRequestHandler

def handler(request):
    return app(request.environ, request.start_response)
