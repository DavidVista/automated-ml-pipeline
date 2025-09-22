import streamlit as st
import requests
import os


# FastAPI endpoint
API_URL = os.environ.get('API_URL', 'http://ml-api:8000')
FASTAPI_URL = f"{API_URL}/predict"


# Streamlit app UI
st.title("Credit Score Classifier")

# Titles for inputs
title_revolving_utilization = ("Total balance on credit cards and personal lines"
                               " of credit except real estate and no installment"
                               " debt like car loans divided by the sum of credit limits")

title_age = "Age of borrower in years"

title_30_to_59_days_late = ("Number of times borrower has been 30-59 days past"
                            " due but no worse in the last 2 years")

title_60_to_89_days_late = ("Number of times borrower has been 60-89 days past"
                            " due but no worse in the last 2 years")

title_90_days_late = ("Number of times borrower has been 90 days or more past due")

title_monthly_income = "Monthly income"

title_debt_ratio = "Monthly debt payments, alimony, and living costs divided by monthy gross income"

title_number_of_credit_lines_and_loans = ("Number of Open loans (installment like car loan"
                                          "or mortgage) and Lines of credit (e.g. credit cards)")

title_number_of_real_estate_loans_or_lines = ("Number of mortgage and real estate loans"
                                              " including home equity lines of credit.")

title_number_of_dependents = "Number of dependents in family excluding themselves (spouse, children etc.)"

# Input fields for the credit score data

revolving_utilization_of_unsecured_lines = st.number_input(title_revolving_utilization, min_value=0.0, max_value=1.0)

age = st.number_input(title_age, min_value=18, max_value=100)

number_of_time_30_to_59_days_past_due_not_worse = st.number_input(title_30_to_59_days_late, min_value=0, max_value=100)

number_of_time_60_to_89_days_past_due_not_worse = st.number_input(title_60_to_89_days_late, min_value=0, max_value=100)

number_of_90_days_late = st.number_input(title_90_days_late, min_value=0, max_value=100)

monthly_income = st.number_input(title_monthly_income, min_value=0)

debt_ratio = st.number_input(title_debt_ratio, min_value=0.0, max_value=1.0)

number_of_open_credit_lines_and_loans = st.number_input(title_number_of_credit_lines_and_loans, min_value=0)

number_real_estate_loans_or_lines = st.number_input(title_number_of_real_estate_loans_or_lines, min_value=0)

number_of_dependents = st.number_input(title_number_of_dependents, min_value=0)


# Make prediction when the button is clicked
if st.button("Predict"):

    # Prepare the data for the API request
    input_data = {
        "revolving_utilization_of_unsecured_lines": revolving_utilization_of_unsecured_lines,
        "age": age,
        "number_of_time_30_to_59_days_past_due_not_worse": number_of_time_30_to_59_days_past_due_not_worse,
        "debt_ratio": debt_ratio,
        "monthly_income": monthly_income,
        "number_of_open_credit_lines_and_loans": number_of_open_credit_lines_and_loans,
        "number_of_90_days_late": number_of_90_days_late,
        "number_real_estate_loans_or_lines": number_real_estate_loans_or_lines,
        "number_of_time_60_to_89_days_past_due_not_worse": number_of_time_60_to_89_days_past_due_not_worse,
        "number_of_dependents": number_of_dependents
    } 

    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()["prediction"]

    # Display the result
    st.success(f"The model predicts class: {prediction}")
