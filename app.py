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
    /* Background */
    .stApp {
        background: linear-gradient(to right, #e6f2ff, #f9f9f9);
    }
    /* Title */
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #003366;
        margin-bottom: 30px;
    }
    /* Card */
    .card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin: auto;
        max-width: 1000px;
    }
    /* Button */
    div.stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #0055a5;
        color: #fff;
    }
    /* Result text */
    .success {
        font-size: 22px;
        font-weight: bold;
        color: green;
        text-align: center;
    }
    .error {
        font-size: 22px;
        font-weight: bold;
        color: red;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# --------------------------
# Load model
# --------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
job_model = pickle.load(open(f'{working_dir}/saved_models/dsJobPrediction_model.sav', 'rb'))

# --------------------------
# App Title
# --------------------------
st.markdown('<div class="title">💼 Job Change Prediction System</div>', unsafe_allow_html=True)


# --------------------------
# Candidate Form inside Card
# --------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 Candidate Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.text_input("City (e.g., city_103)")
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
        relevent_experience = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])
        education_level = st.selectbox("Education Level", ["Primary School", "High School", "Graduate", "Masters", "Phd", "Unknown"])

    with col2:
        city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)
        enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course", "Unknown"])
        major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "Other", "Unknown"])
        experience = st.text_input("Experience (e.g., 2, >20, <1, Unknown)")

    with col3:
        company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"])
        company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other", "Unknown"])
        last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4", "Unknown"])
        training_hours = st.number_input("Training Hours", min_value=0, step=1)

    # Prediction button
    predict_btn = st.button("🔮 Predict Job Change")

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------
# Prediction Result
# --------------------------
if predict_btn:
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

    prediction = job_model.predict(input_data)

    if prediction[0] == 1:
        st.markdown('<div class="success">✅ The candidate is <b>likely</b> to change jobs.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error">❌ The candidate is <b>unlikely</b> to change jobs.</div>', unsafe_allow_html=True)

    # Optional: feature importance preview (static top features)
    st.subheader("📊 Key Factors Driving Predictions (Global Importance)")
    features = ["city", "company_size", "experience", "city_development_index", "last_new_job", "training_hours"]
    importance = [12.7, 12.6, 10.5, 10.0, 9.4, 8.9]  # Replace with your actual importances

    fig, ax = plt.subplots()
    ax.barh(features[::-1], importance[::-1], color="skyblue")
    ax.set_xlabel("Importance %")
    ax.set_title("Top Features")
    st.pyplot(fig)