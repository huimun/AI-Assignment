{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83245555-6de5-4faa-9828-c0ff86286ba2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load the pre-trained model (adjust path if necessary)\n",
    "model = joblib.load('best_forest_model.joblib')  # Load your desired model\n",
    "\n",
    "# Load the dataset (if needed for reference)\n",
    "data = pd.read_csv('SalaryData.csv')\n",
    "\n",
    "# Function to make prediction\n",
    "def predict_salary(age, gender, education_level, job_title, years_of_experience):\n",
    "    # Data preprocessing (adjust based on your model's expectations)\n",
    "    input_data = pd.DataFrame({\n",
    "        'Age': [age],\n",
    "        'Gender': [gender],\n",
    "        'EducationLevel': [education_level],\n",
    "        'JobTitle': [job_title],\n",
    "        'YearsOfExperience': [years_of_experience]\n",
    "    })\n",
    "    \n",
    "    # Prediction using the loaded model\n",
    "    salary_prediction = model.predict(input_data)\n",
    "    return salary_prediction[0]\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Salary Prediction App\")\n",
    "\n",
    "# Input fields for user to provide data\n",
    "age = st.slider(\"Age\", 18, 65, 25)  # Slider for age selection\n",
    "gender = st.selectbox(\"Gender\", [\"Male\", \"Female\", \"Other\"])  # Dropdown for gender selection\n",
    "education_level = st.selectbox(\"Education Level\", [\"High School\", \"Bachelor's\", \"Master's\", \"PhD\"])  # Dropdown for education level\n",
    "job_title = st.text_input(\"Job Title\")  # Text input for job title\n",
    "years_of_experience = st.slider(\"Years of Experience\", 0, 40, 5)  # Slider for experience selection\n",
    "\n",
    "# Prediction button\n",
    "if st.button(\"Predict Salary\"):\n",
    "    salary = predict_salary(age, gender, education_level, job_title, years_of_experience)\n",
    "    st.write(f\"Predicted Salary: ${salary:,.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
