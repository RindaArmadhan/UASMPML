from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load model and encoders
model = joblib.load('model.sav')
le_category = joblib.load('le_category.sav')
le_item = joblib.load('le_item.sav')
le_profitability = joblib.load('le_profitability.sav')

# Load data for form options
data = pd.read_csv('restaurant_menu_optimization_data.csv')
categories = sorted(data['MenuCategory'].unique())
items = sorted(data['MenuItem'].unique())

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    result = model.predict(to_predict)
    return le_profitability.inverse_transform(result)[0]

@app.route('/')
def home():
    return render_template("home.html", categories=categories, items=items)

@app.route('/', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        category = to_predict_list['category']
        item = to_predict_list['item']
        price = float(to_predict_list['price'])

        category_encoded = le_category.transform([category])[0]
        item_encoded = le_item.transform([item])[0]

        to_predict_list = [category_encoded, item_encoded, price]
        result = ValuePredictor(to_predict_list)

        return render_template("home.html", result=result, categories=categories, items=items)

if __name__ == '__main__':
    app.run(debug=True)