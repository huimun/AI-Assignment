import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# List of unique job titles
job_titles = [
    'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director', 'Marketing Analyst',
    'Product Manager', 'Sales Manager', 'Marketing Coordinator', 'Senior Scientist', 'Software Developer',
    # Add the rest of the job titles
]

# Load different machine learning models
model_options = {
    'Random Forest': 'best_forest_model.joblib',
    'K-Nearest Neighbors (KNN)': 'knn.joblib',
    'Decision Tree': 'decision_tree_model.joblib'
}

# User selects the machine learning model
selected_model = st.selectbox("Choose the Machine Learning Model", list(model_options.keys()))

# Try to load the selected model
try:
    model = joblib.load(model_options[selected_model])  # Load the corresponding model file
    st.success(f"{selected_model} model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the {selected_model} model: {e}")

# Load the dataset (if needed for reference)
try:
    data = pd.read_csv('SalaryData.csv')
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading the dataset: {e}")

# Encoding for categorical features (ensure it matches the model's encoding)
gender_map = {'Male': 0, 'Female': 1}
education_level_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}

# Label encoding for job titles
label_encoder = LabelEncoder()
label_encoder.fit(job_titles)

# Function to make prediction
def predict_salary(age, gender, education_level, job_title, years_of_experience):
    try:
        # Convert categorical variables into numeric values
        gender_encoded = gender_map[gender]
        education_level_encoded = education_level_map[education_level]
        job_title_encoded = label_encoder.transform([job_title])[0]  # Encode the selected job title
        
        # Input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Education Level': [education_level_encoded],
            'Job Title': [job_title_encoded],
            'Years of Experience': [years_of_experience]
        })

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
gender = st.selectbox("Gender", ["Male", "Female"])  # Dropdown for gender selection
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])  # Adjusted education levels
job_title = st.selectbox("Job Title", job_titles)  # Dropdown for job title selection
years_of_experience = st.slider("Years of Experience", 0, 40, 5)  # Slider for experience selection

# Prediction button
if st.button("Predict Salary"):
    salary = predict_salary(age, gender, education_level, job_title, years_of_experience)
    if salary:
        st.write(f"Predicted Salary: ${salary:,.2f}")
