import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("adult_model.pkl")  # Ensure this file is in repo root
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please make sure 'adult_model.pkl' is in the same folder as app.py")
        return None

model = load_model()

# -------------------------------
# Streamlit App
# -------------------------------
st.title("💼 Adult Income Prediction App")
st.write("Predict whether a person earns **>50K/year** based on demographic and work attributes.")

# User input fields
age = st.number_input("Age", 18, 100, 30)
workclass = st.selectbox("Workclass", 
    ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
     "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education_num = st.slider("Education (Years)", 1, 16, 10)
marital_status = st.selectbox("Marital Status", 
    ["Married-civ-spouse", "Divorced", "Never-married", "Separated", 
     "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.selectbox("Occupation", 
    ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", 
     "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", 
     "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = st.selectbox("Relationship", 
    ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.radio("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 50000, 0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", 
    ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", 
     "England", "China", "Japan", "Other"])

# Prepare input for model
input_data = pd.DataFrame({
    "age": [age],
    "workclass": [workclass],
    "education_num": [education_num],
    "marital_status": [marital_status],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "sex": [sex],
    "capital_gain": [capital_gain],
    "capital_loss": [capital_loss],
    "hours_per_week": [hours_per_week],
    "native_country": [native_country]
})

# Predict button
if st.button("🔍 Predict"):
    if model:
        prediction = model.predict(input_data)[0]
        if prediction == ">50K":
            st.success("💰 The model predicts this person earns **>50K/year**")
        else:
            st.info("👤 The model predicts this person earns **<=50K/year**")
