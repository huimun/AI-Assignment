import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# List of unique job titles
job_titles = [
    'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director', 'Marketing Analyst',
    'Product Manager', 'Sales Manager', 'Marketing Coordinator', 'Senior Scientist', 'Software Developer', 
    'HR Manager', 'Financial Analyst', 'Project Manager', 'Customer Service Rep', 'Operations Manager',
    'Marketing Manager', 'Senior Engineer', 'Data Entry Clerk', 'Sales Director', 'Business Analyst',
    'VP of Operations', 'IT Support', 'Recruiter', 'Financial Manager', 'Social Media Specialist', 
    'Software Manager', 'Junior Developer', 'Senior Consultant', 'Product Designer', 'CEO', 'Accountant',
    'Data Scientist', 'Marketing Specialist', 'Technical Writer', 'HR Generalist', 'Project Engineer',
    'Customer Success Rep', 'Sales Executive', 'UX Designer', 'Operations Director', 'Network Engineer',
    'Administrative Assistant', 'Strategy Consultant', 'Copywriter', 'Account Manager', 'Director of Marketing',
    'Help Desk Analyst', 'Customer Service Manager', 'Business Intelligence Analyst', 'Event Coordinator',
    'VP of Finance', 'Graphic Designer', 'UX Researcher', 'Senior Data Scientist', 'Junior Accountant', 
    'Digital Marketing Manager', 'IT Manager', 'Customer Service Representative', 'Business Development Manager',
    'Senior Financial Analyst', 'Web Developer', 'Research Director', 'Technical Support Specialist',
    'Creative Director', 'Senior Software Engineer', 'Human Resources Director', 'Content Marketing Manager',
    'Technical Recruiter', 'Sales Representative', 'Chief Technology Officer', 'Junior Designer', 
    'Financial Advisor', 'Junior Account Manager', 'Senior Project Manager', 'Principal Scientist', 
    'Supply Chain Manager', 'Senior Marketing Manager', 'Training Specialist', 'Research Scientist', 
    'Junior Software Developer', 'Public Relations Manager', 'Operations Analyst', 'Product Marketing Manager',
    'Junior Web Developer', 'Senior HR Manager', 'Junior Operations Analyst', 'Customer Success Manager',
    'Senior Graphic Designer', 'Software Project Manager', 'Supply Chain Analyst', 'Senior Business Analyst',
    'Junior Marketing Analyst', 'Office Manager', 'Principal Engineer', 'Junior HR Generalist', 
    'Senior Product Manager', 'Junior Recruiter', 'Senior Business Development Manager', 'Senior Product Designer',
    'Junior Customer Support Specialist', 'Senior Marketing Analyst', 'Senior IT Support Specialist',
    'Director of Product Management', 'Junior Copywriter', 'Senior Human Resources Manager', 'Junior HR Coordinator',
    'Senior Researcher', 'Director of Finance', 'Senior Quality Assurance Analyst', 'Senior Project Coordinator',
    'Director of Business Development', 'Senior Account Executive', 'Director of Human Resources',
    'Senior Financial Manager', 'Senior Data Engineer', 'Senior Operations Analyst', 'Junior Business Operations Analyst',
    'Senior IT Consultant', 'Director of Engineering', 'Senior Marketing Director', 'Senior Product Development Manager',
    'Senior Operations Coordinator', 'Director of Human Capital', 'Junior Operations Manager', 'Senior Software Architect',
    'Senior Financial Advisor'
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

# Initialize the scalers
age_scaler = MinMaxScaler()
experience_scaler = MinMaxScaler()

# Fit the scalers based on the original data
age_scaler.fit(data[['Age']])
experience_scaler.fit(data[['Years of Experience']])

# Function to make prediction
def predict_salary(age, gender, education_level, job_title, years_of_experience):
    try:
        # Convert categorical variables into numeric values
        gender_encoded = gender_map[gender]
        education_level_encoded = education_level_map[education_level]
        job_title_encoded = label_encoder.transform([job_title])[0]  # Encode the selected job title

        # Scale the numerical features (age and years of experience)
        age_scaled = age_scaler.transform([[age]])[0][0]  # Scale age
        experience_scaled = experience_scaler.transform([[years_of_experience]])[0][0]  # Scale experience

        # Input data
        input_data = pd.DataFrame({
            'Age': [age_scaled],  # Scaled Age
            'Gender': [gender_encoded],
            'Education Level': [education_level_encoded],
            'Job Title': [job_title_encoded],
            'Years of Experience': [experience_scaled]  # Scaled Years of Experience
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
st.write(f"Selected Age: {age}")  # Ensure age is being captured

gender = st.selectbox("Gender", ["Male", "Female"])  # Dropdown for gender selection
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])  # Dropdown for education level
job_title = st.selectbox("Job Title", job_titles)  # Dropdown for job title selection
years_of_experience = st.slider("Years of Experience", 0, 40, 5)  # Slider for years of experience

# Prediction button
if st.button("Predict Salary"):
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Education Level: {education_level}")
    st.write(f"Job Title: {job_title}")
    st.write(f"Years of Experience: {years_of_experience}")
    salary = predict_salary(age, gender, education_level, job_title, years_of_experience)
    if salary:
        st.write(f"Predicted Salary: ${salary:,.2f}")
