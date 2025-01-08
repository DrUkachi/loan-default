import streamlit as st
import pandas as pd
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/predict"

# Helper function to make predictions via FastAPI
def get_prediction(data):
    response = requests.post(BACKEND_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error: Unable to connect to backend!")
        return None

# Title and description
st.title("Loan Default Prediction App")
st.write("Predict loan defaults by entering data manually or uploading a CSV file.")

# Input method
input_method = st.radio("Choose input method:", ["Manual Input", "Batch Upload (CSV)"])

if input_method == "Manual Input":
    # Collect user inputs for each column
    st.subheader("Manual Input Form")

    checking_balance = st.number_input("Checking Balance", value=0.0)
    months_loan_duration = st.number_input("Loan Duration (Months)", min_value=1, value=72)
    credit_history = st.selectbox("Credit History", ["critical", "repaid", "delayed", "fully repaid", "fully repaid this bank"])
    purpose = st.selectbox("Purpose", ['radio/tv', 'education', 'furniture', 'car (new)', 'car (used)',
                                       'business', 'domestic appliances', 'repairs', 'others',
                                       'retraining'])
    
    amount = st.number_input("Loan Amount", min_value=0.0, value=1000.0)

    savings_balance = st.number_input("Savings Balance", value=0.0)

    employment_length = st.selectbox("Employment Length", ['13 years', '2 years', '5 years', '3 years', '11 years', '4 years',
                                                            '1 years', '6 months', '5 months', '16 years', '17 years',
                                                            '3 months', '9 years', '4 months', '10 years', '10 months',
                                                            '1 months', '7 months', '19 years', '7 years', '14 years',
                                                            '18 years', '0 months', '15 years', '9 months', '6 years',
                                                            '8 years', '12 years', '11 months', '2 months', '8 months'])
    
    installment_rate = st.number_input("Installment Rate", min_value=1, max_value=4, value=1)

    personal_status = st.selectbox("Personal Status", ["single", "married", "divorced"])

    other_debtors = st.selectbox("Other Debtors", ["none", "guarantor", "co-applicant"])
    residence_history = st.selectbox("Residence History", ['6 years', '5 months', '4 years', '13 years', '8 years',
                                                            '12 years', '3 months', '24 years', '10 months', '0 months',
                                                            '10 years', '2 years', '19 years', '7 years', '3 years',
                                                            '8 months', '7 months', '14 years', '1 years', '16 years',
                                                            '6 months', '20 years', '11 months', '21 years', '5 years',
                                                            '9 months', '2 months', '15 years', '11 years', '18 years',
                                                            '22 years', '23 years', '1 months', '9 years', '4 months',
                                                            '17 years'])
    
    property = st.selectbox("Property", ["real estate", "building society savings", "unknown/none", "other"])
    age = st.number_input("Age", min_value=0, value=25)
    installment_plan = st.selectbox("Installment Plan", ["none", "bank", "store"])
    housing = st.selectbox("Housing", ["own", "rent", "for free"])
    existing_credits = st.number_input("Existing Credits", min_value=0, value=1)
    dependents = st.number_input("Dependents", min_value=0, value=0)
    has_telephone = st.selectbox("Has Telephone", ["0", "1"])
    foreign_worker = st.selectbox("Foreign Worker", ["yes", "no"])
    job = st.selectbox("Job", ['skilled employee', 'unskilled resident',
                                'mangement self-employed', 'unemployed non-resident'])
    
    gender = st.selectbox("Gender", ["male", "female"])

    # Prepare input data
    input_data = {
        "checking_balance": checking_balance,
        "months_loan_duration": months_loan_duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "amount": amount,
        "savings_balance": savings_balance,
        "employment_length": employment_length,
        "installment_rate": installment_rate,
        "personal_status": personal_status,
        "other_debtors": other_debtors,
        "residence_history": residence_history,
        "property": property,
        "age": age,
        "installment_plan": installment_plan,
        "housing": housing,
        "existing_credits": existing_credits,
        "dependents": dependents,
        "has_telephone": int(has_telephone),
        "foreign_worker": foreign_worker,
        "job": job,
        "gender": gender,
    }

    # Submit button
    if st.button("Predict"):
        result = get_prediction(input_data)
        if result:
            st.success(f"Prediction: {result['prediction'][0]}")
            st.info(f"Probability of Default: {result['probability'][0]}")

elif input_method == "Batch Upload (CSV)":
    st.subheader("Batch Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", batch_data)

        # Submit button for batch prediction
        if st.button("Predict for Batch"):
            results = []
            for _, row in batch_data.iterrows():
                data = row.to_dict()
                result = get_prediction(data)
                if result:
                    results.append(result)

            # Show results
            if results:
                predictions = [r["prediction"][0] for r in results]
                probabilities = [r["probability"][0] for r in results]
                batch_data["Prediction"] = predictions
                batch_data["Probability"] = probabilities
                st.write("Prediction Results:", batch_data)
