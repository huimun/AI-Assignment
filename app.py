import streamlit as st
import pandas as pd
import joblib

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

# Encoding for categorical features (ensure it matches the model's encoding)
gender_map = {'Male': 0, 'Female': 1}
education_level_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}

# Function to make prediction
def predict_salary(age, gender, education_level, job_title, years_of_experience):
    try:
        # Convert categorical variables into numeric values
        gender_encoded = gender_map[gender]
        education_level_encoded = education_level_map[education_level]
        
        # Input data (ensure job title is handled if needed)
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Education Level': [education_level_encoded],
            'Job Title': [job_title],  # Keep this as the raw string if your model handles text encoding
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
