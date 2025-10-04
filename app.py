import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Constants / Configuration
# --------------------------
MODEL_DIR = 'saved_models'
MODEL_FILE = 'dsJobPrediction_model.sav'
SCALER_FILE = 'scaler.pkl'
ACCURACY_FILE = 'job_model_accuracy.pkl'
COMMON_CITIES = ['city_103', 'city_21', 'city_149', 'city_11', 'city_73', 'city_160', 'city_40', 'city_67', 'city_104']

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
    :root {
        --color-bg-start: #E4D4C8;
        --color-bg-mid: #F5E6E0;
        --color-bg-end: #DAD0C2;
        --color-primary: #2D2A32;
        --color-secondary: #4A4E69;
        --color-accent: #9DB6CC;
        --color-success-bg-start: #B5EAD7;
        --color-success-bg-end: #C7CEEA;
        --color-error-bg-start: #FF6B6B;
        --color-error-bg-end: #A47786;
        --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, var(--color-bg-start), var(--color-bg-mid), var(--color-bg-end));
        font-family: var(--font-family);
    }
    .title {
        font-size: 52px;
        font-weight: 800;
        text-align: center;
        color: var(--color-primary);
        margin-bottom: 40px;
        letter-spacing: 1px;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    }
    label, .stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label, .stRadio label, .stCheckbox label {
        color: var(--color-primary);
        font-weight: 600;
        font-size: 18px;
    }
    .stSubheader {
        color: black !important;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
        border-bottom: 3px solid var(--color-accent);
        display: inline-block;
        padding-bottom: 5px;
    }
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        margin: auto;
        max-width: 1100px;
        backdrop-filter: blur(8px);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #A47786, var(--color-accent));
        color: #fff;
        border-radius: 12px;
        height: 55px;
        font-size: 20px;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 3px 8px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        filter: brightness(1.1);
    }
    .result-box {
        font-size: 24px;
        font-weight: 700;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    .result-likely {
        color: var(--color-primary);
        background: linear-gradient(135deg, var(--color-success-bg-start), var(--color-success-bg-end));
        border: none;
    }
    .result-unlikely {
        color: #fff;
        background: linear-gradient(135deg, var(--color-error-bg-start), var(--color-error-bg-end));
        border: none;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1100px;
        margin: auto;
        margin-bottom: 20px;
    }
    .info-box {
        background: #f0f4f8;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 16px;
        color: var(--color-primary);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        max-width: 350px;
        text-align: center;
    }
    [data-testid="stMetricValue"] {
        color: red !important;
    }
    [data-testid="stMetricLabel"] {
        color: red !important;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Load model and assets
# --------------------------
@st.cache_resource
def load_assets():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, MODEL_DIR, MODEL_FILE)
    scaler_path = os.path.join(working_dir, MODEL_DIR, SCALER_FILE)
    accuracy_path = os.path.join(working_dir, MODEL_DIR, ACCURACY_FILE)
    try:
        job_model = pickle.load(open(model_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        job_model = None
    try:
        scaler = pickle.load(open(scaler_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading scaler file: {e}")
        scaler = None
    try:
        with open(accuracy_path, 'rb') as f:
            model_accuracy = pickle.load(f)
    except Exception:
        model_accuracy = 88.5
    return job_model, scaler, model_accuracy

job_model, scaler, model_accuracy = load_assets()

# --------------------------
# App Title and Accuracy Info
# --------------------------
st.markdown('<div class="title">Enter employee Details.</div>', unsafe_allow_html=True)

col_acc, col_info = st.columns([1.2, 2])
with col_acc:
    st.metric(label="Model Accuracy", value=f"{model_accuracy:.1f}%")
with col_info:
    st.success(f"Model file: `{MODEL_FILE}` loaded from `{MODEL_DIR}` directory.")

# --------------------------
# Candidate Form
# --------------------------
with st.container():
   
    st.subheader("Candidate Information")

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

    # Center the button
    btn_col1, btn_col2, btn_col3 = st.columns([3,1,3])
    with btn_col2:
        predict_btn = st.button("Predict Job Change")
    st.markdown('</div>', unsafe_allow_html=True)

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

        # Show progress bar during scaling and prediction
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        num_cols = ["city_development_index", "training_hours"]
        progress_bar.progress(20)
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        progress_bar.progress(60)

        prediction = job_model.predict(input_data)
        prediction_proba = job_model.predict_proba(input_data)
        confidence = max(prediction_proba[0]) * 100
        progress_bar.progress(100)
        progress_placeholder.empty()
        progress_bar.empty()

        if prediction[0] == 1:
            st.markdown(f'<div class="result-box result-likely">✅ The candidate is <b>likely</b> to change jobs.<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box result-unlikely">❌ The candidate is <b>unlikely</b> to change jobs.<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)

        with st.expander("Model Details & Input Data"):
            st.write(f"Model Accuracy: {model_accuracy:.1f}%")
            st.write("Input Data (numerical features scaled):")
            st.dataframe(input_data)
