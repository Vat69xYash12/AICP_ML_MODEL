import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="IncomeScope", page_icon="💰", layout="wide")

# -------------------------------
# Load Pickle Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("adult_model.pkl")
  # trained RF model

model = load_model()

# -------------------------------
# Sidebar with Context & Image
# -------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
st.sidebar.title("📊 IncomeScope")
st.sidebar.markdown(
    """
    ### Understanding Socio-Economic Disparities
    In the US, income levels are strongly linked with **education, gender, race, and occupation**.  
    Such disparities highlight systemic inequalities that affect **access to opportunities, healthcare, housing, and overall quality of life**.  
    
    ---
    ### Why This Matters?
    🔹 **Policy Makers** → identify vulnerable groups for better welfare programs.  
    🔹 **Businesses** → design fair hiring practices and wage policies.  
    🔹 **NGOs** → target initiatives to uplift disadvantaged communities.  
    
    ---
    💡 *By predicting income levels from census data, we can uncover patterns that drive inequality — and use those insights to create social impact.*
    """)

# -------------------------------
# Main Title
# -------------------------------
st.title("💰 IncomeScope: Predicting Socio-Economic Outcomes")
st.write("Predict whether a person earns **≤50K** or **>50K** annually.")
st.markdown("---")

# -------------------------------
# Input Form
# -------------------------------
with st.form("income_form"):
    st.subheader("Enter Person Details")

    age = st.number_input("Age", min_value=17, max_value=100, value=30)
    fnlwgt = st.number_input("Final Weight (Fnlwgt)", min_value=10000, max_value=1000000, value=200000)
    education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

    workclass = st.selectbox("Workclass", ["Private", "Other"])
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Divorced", "Never-married", 
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ])
    relationship = st.selectbox("Relationship", [
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ])
    race = st.selectbox("Race", ["White", "Other"])
    sex = st.radio("Sex", ["Male", "Female"])
    country = st.selectbox("Country", ["United-States", "Other"])

    submit = st.form_submit_button("🔍 Predict Income")

# -------------------------------
# Preprocess & Predict
# -------------------------------
if submit:
    # Apply same preprocessing rules as training
    workclass_val = 1 if workclass == "Private" else 0
    race_val = 1 if race == "White" else 0
    country_val = 1 if country == "United-States" else 0
    hours_val = 1 if hours > 40 else 0
    sex_val = 1 if sex == "Male" else 0

    # LabelEncoder assumption (alphabetical encoding)
    marital_status_val = pd.Series([marital_status]).astype("category").cat.codes[0]
    occupation_val = pd.Series([occupation]).astype("category").cat.codes[0]
    relationship_val = pd.Series([relationship]).astype("category").cat.codes[0]

    # Build input dataframe with exact feature names
        # Construct input dataframe
    input_data = pd.DataFrame([{
        "age": int(age),
        "workclass": workclass_val,
        "fnlwgt": int(fnlwgt),
        "education-num": int(education_num),
        "marital-status": marital_status_val,
        "occupation": occupation_val,
        "relationship": relationship_val,
        "race": race_val,
        "sex": sex_val,
        "capital-gain": int(capital_gain),
        "capital-loss": int(capital_loss),
        "hours-per-week": hours_val,
        "country": country_val
    }])

    # ✅ Ensure columns are in the exact order used during training
    expected_order = [
        "age", "workclass", "fnlwgt", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week", "country"
    ]
    input_data = input_data[expected_order]

    st.write("### Encoded Input Data")
    st.dataframe(input_data)

    # Prediction
    results = model.predict(input_data)
    salary_prediction = '💵 Income > 50K' if results[0] == 1 else '💼 Income ≤ 50K'

    st.markdown("---")
    st.success(f"### Prediction Result: {salary_prediction}")


