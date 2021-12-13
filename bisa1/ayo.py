from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
app = Flask(__name__)
hk = pickle.load(open('model.pkl','rb'))
hk1 = pickle.load(open('model_1.pkl','rb'))



@app.route("/",methods=['GET','POST'])
def pro():
    if request.method == 'POST':
        _card_1 = request.form['card_1']
        _card_2 = request.form['card_2']
        _card_deler = request.form['card_deler_2']
        _round = hk1.predict([[_card_1,_card_2,_card_deler]])
        _prediction = hk.predict([[_card_1,_card_2,_round,_card_deler]])
        return render_template('index1.html',prediction=round(_prediction[0]))

    return render_template('index1.html')


if __name__ == "__main__":
    app.run()