import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model (adjust path if necessary)
model = joblib.load('best_forest_model.joblib')  # Load your desired model

# Load the dataset (if needed for reference)
data = pd.read_csv('SalaryData.csv')

# Function to make prediction
def predict_salary(age, gender, education_level, job_title, years_of_experience):
    # Data preprocessing (adjust based on your model's expectations)
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'EducationLevel': [education_level],
        'JobTitle': [job_title],
        'YearsOfExperience': [years_of_experience]
    })
    
    # Prediction using the loaded model
    salary_prediction = model.predict(input_data)
    return salary_prediction[0]

# Streamlit UI
st.title("Salary Prediction App")

# Input fields for user to provide data
age = st.slider("Age", 18, 65, 25)  # Slider for age selection
gender = st.selectbox("Gender", ["Male", "Female", "Other"])  # Dropdown for gender selection
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])  # Dropdown for education level
job_title = st.text_input("Job Title")  # Text input for job title
years_of_experience = st.slider("Years of Experience", 0, 40, 5)  # Slider for experience selection

# Prediction button
if st.button("Predict Salary"):
    salary = predict_salary(age, gender, education_level, job_title, years_of_experience)
    st.write(f"Predicted Salary: ${salary:,.2f}")

pip install streamlit joblib pandas scikit-learn
