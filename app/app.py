from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)

#Hope page
@app.route('/')
def home():
    return render_template('index.html')

# After user hits the submit button.
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    income = request.form['income']                 # Number Field
    house_age = request.form['age']                 # Number Field
    rooms = request.form['rooms']                   # Number Field        
    bedrooms = request.form['bedrooms']             # Number Field
    population = request.form['population']         # Number Field
    
    # Create a numpy array containing our values that will be inputted into the model
    input_array = np.array([[income, house_age, rooms, bedrooms, population]])
    
    #Load Ridge Regression Model from the .joblib file
    model = load('RidgeRegressor.joblib')

    # Grab our prediction and format in the form $9,999,999.00
    pred = '${:,.2f}'.format(model.predict(input_array)[0])

    # Finally return the output to the user
    return render_template('index.html', data=pred)
    
    