import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
with open('model.pkl', 'rb') as mf:
    model = pickle.load(mf)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/displayPrediction",methods=['POST'])
def displayPrediction():
    features_list = [int(x) for x in request.form.values()]
    final_features = [np.array(features_list)]
    modelPrediction = model.predict(final_features)

    outputValue = round(modelPrediction[0],2)
    return render_template('index.html', predicted_text='Predicted Employee Salary: Rs {} /-'.format(outputValue))

if __name__=="__main__":
    app.run(debug=True, threaded=True)