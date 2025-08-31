import streamlit as st
import pandas as pd
import joblib

# Load the trained model
def load_model():
    model = joblib.load("adult_model.pkl")  # Ensure this file is in repo root
    return model

model = load_model()

# Streamlit App UI
st.title("Adult Income Prediction App")

st.write("Fill in the details to predict whether salary >50K or <=50K")

# User Inputs
age = st.number_input("Age", min_value=17, max_value=90, value=25)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
fnlwgt = st.number_input("Fnlwgt", min_value=10000, max_value=1000000, value=200000)
education_num = st.number_input("Education-num", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Marital-status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital-gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital-loss", min_value=0, max_value=5000, value=0)
hours_per_week = st.number_input("Hours-per-week", min_value=1, max_value=100, value=40)
native_country = st.selectbox("Country", ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "England", "China", "Cuba", "Jamaica", "South", "Other"])

# Create dataframe for prediction
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Predicted Salary: >50K")
    else:
        st.success("Predicted Salary: <=50K")
