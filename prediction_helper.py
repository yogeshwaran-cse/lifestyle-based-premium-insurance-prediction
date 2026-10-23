import os
from pathlib import Path
import joblib
import pandas as pd

# Define paths to artifacts
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"

# Load the saved model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature names as expected by scaler and model
FEATURE_NAMES = [
    'age',
    'number_of_dependants',
    'income_lakhs',
    'insurance_plan',
    'life_style_risk_score',
    'total_risk_score',
    'gender_Male',
    'region_Northwest',
    'region_Southeast',
    'region_Southwest',
    'marital_status_Unmarried',
    'bmi_category_Obesity',
    'bmi_category_Overweight',
    'bmi_category_Underweight',
    'smoking_status_Occasional',
    'smoking_status_Regular',
    'employment_status_Salaried',
    'employment_status_Self-Employed'
]

# Risk score dictionaries based on model training
PHYSICAL_ACTIVITY_SCORES = {
    'high': 0,
    'medium': 1,
    'low': 4
}

STRESS_LEVEL_SCORES = {
    'high': 4,
    'medium': 1,
    'low': 0
}

MEDICAL_RISK_SCORES = {
    'diabetes': 6,
    'heart disease': 8,
    'high blood pressure': 6,
    'thyroid': 5,
    'no disease': 0,
    'none': 0
}

INSURANCE_PLAN_MAP = {
    'bronze': 1,
    'silver': 2,
    'gold': 3
}


def calculate_lifestyle_risk_score(physical_activity: str, stress_level: str) -> int:
    """Calculate lifestyle risk score based on physical activity and stress level."""
    pa_score = PHYSICAL_ACTIVITY_SCORES.get(str(physical_activity).strip().lower(), 1)
    sl_score = STRESS_LEVEL_SCORES.get(str(stress_level).strip().lower(), 1)
    return pa_score + sl_score


def calculate_total_risk_score(medical_history) -> int:
    """Calculate medical total risk score from medical history string or list."""
    if not medical_history:
        return 0

    if isinstance(medical_history, str):
        # Handle formats like "High blood pressure & Heart disease" or comma-separated
        items = [item.strip().lower() for item in medical_history.replace('&', ',').split(',') if item.strip()]
    elif isinstance(medical_history, (list, tuple, set)):
        items = [str(item).strip().lower() for item in medical_history if str(item).strip()]
    else:
        return 0

    return sum(MEDICAL_RISK_SCORES.get(item, 0) for item in items)


def preprocess_input(
    age: int,
    number_of_dependants: int,
    income_lakhs: float,
    insurance_plan: str,
    physical_activity: str,
    stress_level: str,
    medical_history,
    gender: str,
    region: str,
    marital_status: str,
    bmi_category: str,
    smoking_status: str,
    employment_status: str
) -> pd.DataFrame:
    """
    Transform raw user inputs into the exact 18 features expected by scaler and model.
    """
    # 1. Calculated risk scores
    lifestyle_risk = calculate_lifestyle_risk_score(physical_activity, stress_level)
    total_risk = calculate_total_risk_score(medical_history)

    # 2. Ordinal / mapped plan
    plan_val = INSURANCE_PLAN_MAP.get(str(insurance_plan).strip().lower(), 1)

    # 3. Categorical one-hot encodings (drop_first=True reference categories)
    # Gender (Reference: Female)
    gender_male = 1 if str(gender).strip().lower() == 'male' else 0

    # Region (Reference: Northeast)
    region_str = str(region).strip().lower()
    region_northwest = 1 if region_str == 'northwest' else 0
    region_southeast = 1 if region_str == 'southeast' else 0
    region_southwest = 1 if region_str == 'southwest' else 0

    # Marital status (Reference: Married)
    marital_unmarried = 1 if str(marital_status).strip().lower() in ['unmarried', 'single'] else 0

    # BMI Category (Reference: Normal)
    bmi_str = str(bmi_category).strip().lower()
    bmi_obesity = 1 if bmi_str == 'obesity' else 0
    bmi_overweight = 1 if bmi_str == 'overweight' else 0
    bmi_underweight = 1 if bmi_str == 'underweight' else 0

    # Smoking Status (Reference: No Smoking)
    smoke_str = str(smoking_status).strip().lower()
    smoke_occasional = 1 if smoke_str == 'occasional' else 0
    smoke_regular = 1 if smoke_str == 'regular' else 0

    # Employment Status (Reference: Freelancer)
    emp_str = str(employment_status).strip().lower()
    emp_salaried = 1 if emp_str == 'salaried' else 0
    emp_self_employed = 1 if emp_str in ['self-employed', 'self employed'] else 0

    row_data = {
        'age': int(age),
        'number_of_dependants': int(number_of_dependants),
        'income_lakhs': float(income_lakhs),
        'insurance_plan': int(plan_val),
        'life_style_risk_score': int(lifestyle_risk),
        'total_risk_score': int(total_risk),
        'gender_Male': int(gender_male),
        'region_Northwest': int(region_northwest),
        'region_Southeast': int(region_southeast),
        'region_Southwest': int(region_southwest),
        'marital_status_Unmarried': int(marital_unmarried),
        'bmi_category_Obesity': int(bmi_obesity),
        'bmi_category_Overweight': int(bmi_overweight),
        'bmi_category_Underweight': int(bmi_underweight),
        'smoking_status_Occasional': int(smoke_occasional),
        'smoking_status_Regular': int(smoke_regular),
        'employment_status_Salaried': int(emp_salaried),
        'employment_status_Self-Employed': int(emp_self_employed)
    }

    return pd.DataFrame([row_data], columns=FEATURE_NAMES)


def predict(
    age: int,
    number_of_dependants: int,
    income_lakhs: float,
    insurance_plan: str,
    physical_activity: str,
    stress_level: str,
    medical_history,
    gender: str,
    region: str,
    marital_status: str,
    bmi_category: str,
    smoking_status: str,
    employment_status: str
) -> float:
    """
    Preprocess inputs, scale features, and predict annual premium using the loaded model.
    """
    df_features = preprocess_input(
        age=age,
        number_of_dependants=number_of_dependants,
        income_lakhs=income_lakhs,
        insurance_plan=insurance_plan,
        physical_activity=physical_activity,
        stress_level=stress_level,
        medical_history=medical_history,
        gender=gender,
        region=region,
        marital_status=marital_status,
        bmi_category=bmi_category,
        smoking_status=smoking_status,
        employment_status=employment_status
    )

    scaled_features = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_NAMES)
    prediction = model.predict(scaled_df)
    
    # Return as float, non-negative
    return float(max(0.0, prediction[0]))


def predict_from_dict(data: dict) -> float:
    """
    Convenience method to predict from a dictionary of inputs.
    """
    return predict(
        age=data.get('age', 30),
        number_of_dependants=data.get('number_of_dependants', 0),
        income_lakhs=data.get('income_lakhs', 10.0),
        insurance_plan=data.get('insurance_plan', 'Bronze'),
        physical_activity=data.get('physical_activity', 'Medium'),
        stress_level=data.get('stress_level', 'Medium'),
        medical_history=data.get('medical_history', 'No Disease'),
        gender=data.get('gender', 'Male'),
        region=data.get('region', 'Northeast'),
        marital_status=data.get('marital_status', 'Married'),
        bmi_category=data.get('bmi_category', 'Normal'),
        smoking_status=data.get('smoking_status', 'No Smoking'),
        employment_status=data.get('employment_status', 'Salaried')
    )
