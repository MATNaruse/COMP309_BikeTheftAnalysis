# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
===========================
301 063 251 : Arthur Batista
300 549 638 : Matthew Naruse
301 041 132 : Trent B Minia
300 982 276 : Simon Ducuara
300 944 562 : Zeedan Ahmed

5)	Deploying the model
    a)	Using flask framework arrange to turn your selected machine-learning 
        model into an API.
        
    b)	Using pickle module arrange for Serialization & Deserialization of 
        your model.
        
    c)	Build a client to test your model API service. Use the test data, 
        which was not previously used to train the module. You can use 
        simple Jinja HTML templates with or without Java script, REACT 
        or any other technology but at minimum use POSTMAN Client API.
"""

import traceback, joblib, os
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if dTree:
        try:
            json_ = request.json

            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)

            prediction = list(dTree.predict(query))
            count_0 = prediction.count(0)
            count_1 = prediction.count(1)
            
            out_json = {'predictions': str(prediction), 
                            'count_0': count_0, 
                            'count_1': count_1}
            print(out_json)
            return jsonify(out_json)

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    
    print("Loading Model...")
    dTree = joblib.load(os.path.join(Path(__file__).parents[0],'model_lr.pkl'))
    print('...Model loaded!')
    
    print("Loading Model Columns...")
    model_columns = joblib.load(os.path.join(Path(__file__).parents[0],'model_columns.pkl'))
    print('....Model columns loaded!')
    
    
    app.run(port=12345, debug=True)
