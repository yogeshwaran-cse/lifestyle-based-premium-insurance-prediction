import streamlit as st
import prediction_helper

# Page Configuration
st.set_page_config(
    page_title="Life Style Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide"
)

# App Title and Description
st.title("🛡️ Life Style Insurance Premium Predictor")
st.write(
    "Estimate your annual insurance premium based on personal demographics, "
    "lifestyle factors, employment, and medical history."
)
st.divider()

# Input layout using 3 columns
col1, col2, col3 = st.columns(3)

# ----------------- Column 1: Personal & Demographic -----------------
with col1:
    st.subheader("👤 Demographics")
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Age of the primary insured (18 - 100 years)"
    )
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        help="Select gender"
    )
    marital_status = st.selectbox(
        "Marital Status",
        options=["Married", "Unmarried"],
        help="Current marital status"
    )
    number_of_dependants = st.number_input(
        "Number of Dependants",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
        help="Number of dependent family members (0 to 5)"
    )
    region = st.selectbox(
        "Region",
        options=["Northeast", "Northwest", "Southeast", "Southwest"],
        help="Geographic region of residence"
    )

# ----------------- Column 2: Financial & Policy -----------------
with col2:
    st.subheader("💼 Financial & Policy")
    income_lakhs = st.number_input(
        "Annual Income (in Lakhs ₹)",
        min_value=0.0,
        max_value=100.0,
        value=12.0,
        step=0.5,
        help="Annual gross income in Lakhs (e.g., 12.0 = ₹12,00,000)"
    )
    employment_status = st.selectbox(
        "Employment Status",
        options=["Salaried", "Self-Employed", "Freelancer"],
        help="Employment type"
    )
    insurance_plan = st.selectbox(
        "Insurance Plan",
        options=["Bronze", "Silver", "Gold"],
        index=1,
        help="Tier of insurance coverage"
    )

# ----------------- Column 3: Lifestyle & Health -----------------
with col3:
    st.subheader("🏃 Lifestyle & Health")
    bmi_category = st.selectbox(
        "BMI Category",
        options=["Normal", "Overweight", "Obesity", "Underweight"],
        help="Body Mass Index category"
    )
    smoking_status = st.selectbox(
        "Smoking Status",
        options=["No Smoking", "Occasional", "Regular"],
        help="Smoking habits"
    )
    physical_activity = st.selectbox(
        "Physical Activity",
        options=["High", "Medium", "Low"],
        index=1,
        help="Regular physical activity level"
    )
    stress_level = st.selectbox(
        "Stress Level",
        options=["Low", "Medium", "High"],
        index=1,
        help="Typical daily stress level"
    )
    medical_history = st.multiselect(
        "Medical Conditions",
        options=["High blood pressure", "Diabetes", "Thyroid", "Heart disease"],
        default=[],
        help="Select any diagnosed pre-existing conditions (leave empty if none)"
    )

st.divider()

# Prediction button
predict_btn = st.button("Calculate Insurance Premium", type="primary", use_container_width=True)

if predict_btn:
    try:
        # Compute predicted premium
        estimated_premium = prediction_helper.predict(
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

        lifestyle_risk = prediction_helper.calculate_lifestyle_risk_score(physical_activity, stress_level)
        medical_risk = prediction_helper.calculate_total_risk_score(medical_history)

        st.success("### Prediction Successful!")

        # Display result metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric(
                label="Estimated Annual Premium",
                value=f"₹ {estimated_premium:,.2f}"
            )
        with mcol2:
            monthly_premium = estimated_premium / 12
            st.metric(
                label="Estimated Monthly Premium",
                value=f"₹ {monthly_premium:,.2f}"
            )
        with mcol3:
            st.metric(
                label="Plan Selected",
                value=insurance_plan
            )

        # Risk breakdown expander
        with st.expander("📋 View Risk Score & Input Summary"):
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.write(f"**Lifestyle Risk Score:** {lifestyle_risk}")
                st.caption("Derived from Physical Activity & Stress Level")
                st.write(f"**Medical Risk Score:** {medical_risk}")
                st.caption("Derived from pre-existing medical conditions")
            with rcol2:
                st.write(f"**Selected Diseases:** {', '.join(medical_history) if medical_history else 'None (Healthy)'}")
                st.write(f"**BMI & Smoking:** {bmi_category} | {smoking_status}")
                st.write(f"**Employment & Region:** {employment_status} | {region}")

    except Exception as e:
        st.error(f"An error occurred while calculating the premium: {str(e)}")