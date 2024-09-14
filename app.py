import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
try:
    model = joblib.load('best_forest_model.joblib')  # Adjust the path if necessary
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the dataset (if needed for reference)
try:
    data = pd.read_csv('SalaryData.csv')
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading the dataset: {e}")

# Function to make prediction
def predict_salary(age, gender, education_level, job_title, years_of_experience):
    # Adjust the column names to match what was used in training the model
    input_data = pd.DataFrame({
        'age': [age],  # Ensure lowercase or match exact feature name
        'gender': [gender],  # Ensure it matches what was used in the training
        'education': [education_level],  # Adjust to match training data
        'job_title': [job_title],  # Adjust to match the model's training data
        'experience': [years_of_experience]  # Make sure this matches the model
    })
    
    try:
        # Prediction using the loaded model
        salary_prediction = model.predict(input_data)
        return salary_prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit UI
st.title("Salary Prediction App")

# Input fields for user to provide data
age = st.slider("Age", 18, 65, 25)  # Slider for age selection
gender = st.selectbox("Gender", ["Male", "Female"])  # Dropdown for gender selection (removed "Other")
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])  # Adjusted education levels
job_title = st.text_input("Job Title")  # Text input for job title
years_of_experience = st.slider("Years of Experience", 0, 40, 5)  # Slider for experience selection

# Prediction button
if st.button("Predict Salary"):
    salary = predict_salary(age, gender, education_level, job_title, years_of_experience)
    if salary:
        st.write(f"Predicted Salary: ${salary:,.2f}")

