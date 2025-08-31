import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("adult_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Adult Income Prediction", page_icon="💰", layout="wide")

st.title("💰 Adult Income Prediction App")
st.write("Predict whether a person earns more than 50K based on census data.")

# Input fields
age = st.number_input("Age", min_value=17, max_value=100, value=25)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                       'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                                       'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th',
                                       'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th',
                                       'Preschool'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 
                                                 'Separated', 'Widowed', 'Married-spouse-absent',
                                                 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                                         'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                         'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                         'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                         'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                                             'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.radio("Sex", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=10000, value=0)
hours_per_week = st.slider("Hours per week", 1, 100, 40)
native_country = st.selectbox("Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                                          'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                                          'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras',
                                          'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
                                          'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
                                          'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary',
                                          'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
                                          'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

# Convert inputs into dataframe for prediction
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education': [education],
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

if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 This person is likely to earn **more than 50K**.")
    else:
        st.warning("⚠️ This person is likely to earn **less than or equal to 50K**.")
