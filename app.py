import streamlit as st
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('data.csv')
st.title("Car Price Predictor")

MODELS = []


coumpany = st.selectbox("Select a coumpany", df['company'].unique())

for i in sorted(df['name'].unique()):
    if coumpany in i:
        MODELS.append(i)

model_vc = st.selectbox("Select a model", MODELS)

year = st.number_input("Enter a year", min_value=1900, max_value=2019, value=2019)

fule = st.selectbox("Select a fule", ['Petrol', 'Diesel', 'LPG'])

kmps = st.number_input("Enter a kmps Driven", min_value=0, max_value=1000000000, value=500)

predict = st.button("Predict")

if predict:
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([model_vc,coumpany,year,kmps,fule]).reshape(1, 5)))

    st.text(prediction[0].round())

