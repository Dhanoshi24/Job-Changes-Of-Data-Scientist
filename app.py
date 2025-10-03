import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="Job Change Prediction",
                   layout="wide",
                   page_icon="💼")

# --------------------------
# Get working directory
# --------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved job prediction model
job_model = pickle.load(open(f'{working_dir}/saved_models/dsJobPrediction_model.sav', 'rb'))

# --------------------------
# Sidebar navigation
# --------------------------
with st.sidebar:
    selected = option_menu('Job Prediction System',
                           ['Job Change Prediction'],
                           menu_icon='briefcase-fill',
                           icons=['person-badge'],
                           default_index=0)

# --------------------------
# Job Prediction Page
# --------------------------
if selected == 'Job Change Prediction':

    st.title('Job Change Prediction using ML 💼')

    st.markdown("Fill in the candidate details below:")

    # Collect candidate info
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.text_input("City (e.g., city_103)")

    with col2:
        city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])

    with col1:
        relevent_experience = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])

    with col2:
        enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course", "Unknown"])

    with col3:
        education_level = st.selectbox("Education Level", ["Primary School", "High School", "Graduate", "Masters", "Phd", "Unknown"])

    with col1:
        major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "Other", "Unknown"])

    with col2:
        experience = st.text_input("Experience (e.g., 2, >20, <1, Unknown)")

    with col3:
        company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"])

    with col1:
        company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other", "Unknown"])

    with col2:
        last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4", "Unknown"])

    with col3:
        training_hours = st.number_input("Training Hours", min_value=0, step=1)

    # --------------------------
    # Prediction
    # --------------------------
    job_prediction = ''
    if st.button("Predict Job Change"):

        # Create DataFrame for input
        input_data = pd.DataFrame([{
            "city": city,
            "city_development_index": city_development_index,
            "gender": gender,
            "relevent_experience": relevent_experience,
            "enrolled_university": enrolled_university,
            "education_level": education_level,
            "major_discipline": major_discipline,
            "experience": experience,
            "company_size": company_size,
            "company_type": company_type,
            "last_new_job": last_new_job,
            "training_hours": training_hours
        }])

        # Predict
        prediction = job_model.predict(input_data)

        if prediction[0] == 1:
            job_prediction = "✅ The candidate is **likely to change jobs**."
        else:
            job_prediction = "❌ The candidate is **unlikely to change jobs**."

    st.success(job_prediction)