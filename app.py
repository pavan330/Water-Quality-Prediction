from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
       Ph = float(request.form['Ph'])
       Cw = float(request.form['Cw'])
       Ppm = float(request.form['Ppm'])
       Chloramines = float(request.form['Chloramines'])
       Sulfates = float(request.form['Sulfates'])
       Electricity = float(request.form['Electricity'])
       Carbon = float(request.form['Carbon'])
       Trihalomethanes = float(request.form['Trihalomethanes'])
       Light = float(request.form['Light'])
        

       values = np.array([[Ph, Cw,  Ppm,Chloramines,Sulfates,Electricity,Carbon,Trihalomethanes,Light]])
       prediction = model.predict(values)

       return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)