import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("adult_model.pkl")  # new joblib file

model = load_model()

st.title("Adult Income Prediction App")

# Input fields
age = st.number_input("Age", 18, 100, 25)
workclass = st.text_input("Workclass")
fnlwgt = st.number_input("Fnlwgt", 0, 1000000, 50000)
education_num = st.number_input("Education Number", 1, 16, 10)
marital_status = st.text_input("Marital Status")
occupation = st.text_input("Occupation")
relationship = st.text_input("Relationship")
race = st.text_input("Race")
sex = st.text_input("Sex")
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.number_input("Hours per week", 1, 100, 40)
country = st.text_input("Country")

if st.button("Predict"):
    input_data = pd.DataFrame([[age, workclass, fnlwgt, education_num, marital_status,
                                occupation, relationship, race, sex, capital_gain,
                                capital_loss, hours_per_week, country]],
                              columns=['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                                       'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                       'capital-loss', 'hours-per-week', 'country'])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
