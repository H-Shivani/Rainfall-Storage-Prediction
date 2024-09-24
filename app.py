import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


import pickle
# Frontend using Streamlit
st.title('Rainfall Storage Prediction')
le = None

#column transform file
with open('encoder.pkl','rb') as file:
    le = pickle.load(file)

#loading the saved model
with open('model.pkl','rb') as file:
    model = pickle.load(file)

#transforming , standardizing the values
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
# User input for state, day, month, and year
state = st.selectbox('Select State', ['Kerala', 'Tamil Nadu', 'Assam', 'Uttarakhand', 'Tripura', 'Telangana', 'Delhi', 'Mizoram', 'Sikkim'])  # Add your states here
day = st.number_input('Enter Day (1-31)', min_value=1, max_value=31, value=30)
month = st.number_input('Enter Month (1-12)', min_value=1, max_value=12, value=4)
year = st.number_input('Enter Year', min_value=2000, max_value=2100, value=2025)

# User input for model selection
# model_choice = st.selectbox('Select Model', ['SVR', 'Decision Tree', 'Linear Regression'])

# Function to predict rainfall storage
def predict_future_rainfall(state, day, month, year):
    state_encoded = le.transform([state])[0]
    input_data = np.array([[state_encoded, day, month, year]])
    input_data_scaled = scaler.transform(input_data)
    pred = model.predict(input_data_scaled)
    return round(pred[0],3)

# Initialize models and encoders (use pre-trained models)
# For this example, we're assuming these are pre-fitted.
# Replace with actual loading or fitting logic.
# Example: dt_model = load_model('decision_tree.pkl')

# When the user clicks the predict button
if st.button('Predict Rainfall Storage'):
    predicted_rfs = predict_future_rainfall(state, day, month, year)
    result = f"Predicted rainfall storage in {state} on {day}-{month}-{year} is: {predicted_rfs} mm"
    st.success(result)



    