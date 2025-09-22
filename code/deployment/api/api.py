from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os


model = None
imputer = None
scaler = None


# Define the FastAPI api
api = FastAPI()

path = os.environ.get("MODEL_PATH", "/app/models")
model = os.environ.get("MODEL_NAME", "model")

# Load the trained model
with open(os.path.join(path, f"{model}.pkl"), "rb") as f:
    model = pickle.load(f)

# Load transformations
with open(os.path.join(path, "imputer.pkl"), "rb") as f:
    imputer = pickle.load(f)

with open(os.path.join(path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)


# Define the input data schema
class CreditScoreInput(BaseModel):
    revolving_utilization_of_unsecured_lines: float
    age: int
    number_of_time_30_to_59_days_past_due_not_worse: int
    debt_ratio: float
    monthly_income: int
    number_of_open_credit_lines_and_loans: int
    number_of_90_days_late: int
    number_real_estate_loans_or_lines: int
    number_of_time_60_to_89_days_past_due_not_worse: int
    number_of_dependents: int


columns = [
    'revolving_utilization_of_unsecured_lines',
    'age',
    'number_of_time_30_to_59_days_past_due_not_worse',
    'debt_ratio',
    'monthly_income',
    'number_of_open_credit_lines_and_loans',
    'number_of_90_days_late',
    'number_real_estate_loans_or_lines',
    'number_of_time_60_to_89_days_past_due_not_worse',
    'number_of_dependents'
]


# Define the prediction endpoint

@api.post("/predict")
def predict(input_data: CreditScoreInput):

    data = [[
        feature[1] for feature in input_data
    ]]

    # Preprocess
    data = scaler.transform(imputer.transform(data))

    data = {column: value for column, value in zip(columns, *data)}

    # Extract Features
    total_late_days = data['number_of_time_30_to_59_days_past_due_not_worse'] + \
        data['number_of_time_60_to_89_days_past_due_not_worse'] + \
        data['number_of_90_days_late']

    income_expense_difference = data['monthly_income'] - data['monthly_income'] * data['debt_ratio']

    late_90_days_likelihood = (0.1 * data['number_of_time_30_to_59_days_past_due_not_worse'] +
                               0.2 * data['number_of_time_60_to_89_days_past_due_not_worse'] +
                               0.7 * data['number_of_90_days_late'])

    data = [[
        data['revolving_utilization_of_unsecured_lines'],
        data['age'],
        data['number_of_open_credit_lines_and_loans'],
        data['number_real_estate_loans_or_lines'],
        data['number_of_dependents'],
        total_late_days,
        income_expense_difference,
        late_90_days_likelihood
    ]]

    prediction = model.predict_proba(data)[0][1]

    threshold = float(os.environ.get("THRESHOLD", "0.5"))

    predicted_class = 1 if prediction > threshold else 0

    return {"prediction": predicted_class}
