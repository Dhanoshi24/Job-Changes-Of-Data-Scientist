import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Job Change Prediction",
    page_icon="💼",
    layout="wide",
)

# Custom CSS for beauty
st.markdown("""
    <style>
    .stApp {
        background: #E4D4C8;
    }
    .title {
        font-size: 44px;
        font-weight: bold;
        text-align: center;
        color: #533440;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
    }
    label, .stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label, .stRadio label, .stCheckbox label {
        color: #533440;
        font-weight: bold;
    }
    .stSubheader {
        color: #533440;
        font-weight: bold;
    }
    .card {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(164, 119, 134, 0.3);
        margin: auto;
        max-width: 1000px;
    }
    div.stButton > button {
        background-color: #A47786;
        color: #fff;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: background 0.2s;
        box-shadow: 0 1px 3px rgba(164, 119, 134, 0.3);
    }
    div.stButton > button:hover {
        background-color: #9DB6CC;
        color: #fff;
    }
    .success, .error {
        font-size: 22px;
        font-weight: bold;
        border-radius: 8px;
        padding: 16px 0;
        text-align: center;
        margin-top: 16px;
    }
    .success {
        color: #533440;
        background: #9DB6CC;
        border: 1.5px solid #9DB6CC;
    }
    .error {
        color: #E4D4C8;
        background: #A47786;
        border: 1.5px solid #A47786;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Load model
# --------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
job_model = pickle.load(open(f'{working_dir}/saved_models/dsJobPrediction_model.sav', 'rb'))
scaler = pickle.load(open(f'{working_dir}/saved_models/scaler.pkl', 'rb'))

accuracy_path = f'{working_dir}/saved_models/job_model_accuracy.pkl'
try:
    with open(accuracy_path, 'rb') as f:
        model_accuracy = pickle.load(f)
except FileNotFoundError:
    model_accuracy = 88.5

# --------------------------
# App Title
# --------------------------
st.markdown('<div class="title">💼 Job Change Prediction System</div>', unsafe_allow_html=True)

# --------------------------
# Candidate Form
# --------------------------
with st.container():
    st.subheader("📋 Candidate Information")

    col1, col2, col3 = st.columns(3)

    gender_options = ["Select gender", "Male", "Female", "Other", "Unknown"]
    experience_options = ["Select experience", "Has relevent experience", "No relevent experience"]
    education_options = ["Select education level", "Primary School", "High School", "Graduate", "Masters", "Phd", "Unknown"]
    university_options = ["Select enrollment", "no_enrollment", "Full time course", "Part time course", "Unknown"]
    discipline_options = ["Select discipline", "STEM", "Business Degree", "Arts", "Humanities", "Other", "Unknown"]
    company_size_options = ["Select company size", "<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"]
    company_type_options = ["Select company type", "Pvt Ltd", "Startup", "Public Sector", "NGO", "Other", "Unknown"]
    last_job_options = ["Select last new job", "never", "1", "2", "3", "4", ">4", "Unknown"]

    with col1:
        city = st.text_input("City (e.g., city_103)", placeholder="city_103")
        gender = st.selectbox("Gender", gender_options)
        relevent_experience = st.selectbox("Relevant Experience", experience_options)
        education_level = st.selectbox("Education Level", education_options)

    with col2:
        city_development_index = st.text_input("City Development Index", placeholder="Value between 0 and 1")
        enrolled_university = st.selectbox("Enrolled University", university_options)
        major_discipline = st.selectbox("Major Discipline", discipline_options)
        experience = st.text_input("Experience (e.g., 2, >20, <1, Unknown)")

    with col3:
        company_size = st.selectbox("Company Size", company_size_options)
        company_type = st.selectbox("Company Type", company_type_options)
        last_new_job = st.selectbox("Last New Job", last_job_options)
        training_hours = st.text_input("Training Hours", placeholder="e.g., 40")

    predict_btn = st.button("🔮 Predict Job Change")

# --------------------------
# Validation and Prediction
# --------------------------
if predict_btn:
    missing_fields = []

    # Validate text fields
    if not city.strip():
        missing_fields.append("City")
    if not city_development_index.strip():
        missing_fields.append("City Development Index")
    if not experience.strip():
        missing_fields.append("Experience")
    if not training_hours.strip():
        missing_fields.append("Training Hours")

    # Validate dropdown selections
    dropdowns = {
        "Gender": gender,
        "Relevant Experience": relevent_experience,
        "Education Level": education_level,
        "University": enrolled_university,
        "Major Discipline": major_discipline,
        "Company Size": company_size,
        "Company Type": company_type,
        "Last New Job": last_new_job,
    }

    for label, value in dropdowns.items():
        if value.startswith("Select"):
            missing_fields.append(label)

    # Show error if missing
    if missing_fields:
        st.error(f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}")
    else:
        try:
            city_development_index_val = float(city_development_index)
            training_hours_val = float(training_hours)
        except ValueError:
            st.error("⚠️ City Development Index and Training Hours must be numeric.")
            st.stop()

        input_data = pd.DataFrame([{
            "city": city,
            "city_development_index": city_development_index_val,
            "gender": gender,
            "relevent_experience": relevent_experience,
            "enrolled_university": enrolled_university,
            "education_level": education_level,
            "major_discipline": major_discipline,
            "experience": experience,
            "company_size": company_size,
            "company_type": company_type,
            "last_new_job": last_new_job,
            "training_hours": training_hours_val
        }])

        num_cols = ["city_development_index", "training_hours"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = job_model.predict(input_data)
        prediction_proba = job_model.predict_proba(input_data)
        confidence = max(prediction_proba[0]) * 100

        if prediction[0] == 1:
            st.markdown('<div class="success">✅ The candidate is <b>likely</b> to change jobs.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="success">Model Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="success">Model Accuracy: {model_accuracy:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error">❌ The candidate is <b>unlikely</b> to change jobs.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="error">Model Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="error">Model Accuracy: {model_accuracy:.1f}%</div>', unsafe_allow_html=True)
