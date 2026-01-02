import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor 🌟",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    .stSidebar {
        background-color: #1c1f26;
        color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("👔 Employee Attrition Predictor")
st.markdown("Predict whether an employee will **LEAVE** or **STAY** using Random Forest")

# -------------------------------
# SAMPLE DATA
# -------------------------------
df = pd.DataFrame({
    "Age": [25, 35, 45, 28],
    "MonthlyIncome": [20000, 40000, 60000, 30000],
    "YearsAtCompany": [1, 8, 15, 3],
    "JobSatisfaction": [3, 4, 2, 3],
    "WorkLifeBalance": [2, 3, 4, 2],
    "OverTime": ["Yes", "No", "No", "Yes"],
    "NumCompaniesWorked": [1, 2, 3, 1],
    "TrainingTimes": [2, 3, 4, 1],
    "Attrition": ["Yes", "No", "No", "Yes"]
})

# -------------------------------
# PREPROCESS
# -------------------------------
le = LabelEncoder()
df["OverTime"] = le.fit_transform(df["OverTime"])
df["Attrition"] = le.fit_transform(df["Attrition"])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("📝 Employee Details")
age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 10000, 100000, 30000)
years = st.sidebar.slider("Years at Company", 0, 40, 5)
job_sat = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
work_life = st.sidebar.selectbox("Work Life Balance", [1, 2, 3, 4])
overtime = st.sidebar.selectbox("Over Time", ["No", "Yes"])
companies = st.sidebar.slider("Companies Worked", 0, 10, 2)
training = st.sidebar.slider("Training Times / Year", 0, 6, 2)

overtime_val = 1 if overtime == "Yes" else 0

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🔍 Predict Attrition"):
    prediction = model.predict([[age, income, years, job_sat,
                                 work_life, overtime_val,
                                 companies, training]])[0]

    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.markdown(
            f"<div style='background-color:#ff4b5c;padding:20px;border-radius:10px;'>"
            f"❌ <b>Employee is likely to LEAVE</b>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background-color:#28c76f;padding:20px;border-radius:10px;'>"
            f"✅ <b>Employee is likely to STAY</b>"
            f"</div>", unsafe_allow_html=True)

# -------------------------------
# DISPLAY SAMPLE DATA
# -------------------------------
with st.expander("📊 Sample Employee Data"):
    st.dataframe(df)
