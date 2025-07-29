import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoder
model = joblib.load("loan_risk_classifier.pkl")
encoder = joblib.load("risk_label_encoder.pkl")

st.set_page_config(page_title="Airtel Loan Risk Predictor", layout="centered")
st.title("📊 Airtel Mobile Loan Risk Predictor")
st.markdown("Enter customer features to predict **credit risk class** for Airtel Mobile Loans.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=60, value=30)
months_registered = st.slider("Months Registered", 0, 60, value=24)
total_spend = st.number_input("Monthly Mobile Money Spend (UGX)", value=50000)
txn_count = st.slider("Monthly Transaction Count", 0, 30, value=10)
loan_count = st.selectbox("Active Loans on Airtel Platform", [0, 1])
arrears_days = st.slider("CRB Arrears Days", 0, 120, value=0)

if st.button("Predict Risk Class"):
    input_df = pd.DataFrame([{
        "age": age,
        "months_registered": months_registered,
        "total_spend": total_spend,
        "txn_count": txn_count,
        "loan_count": loan_count,
        "arrears_days": arrears_days
    }])
    
    pred_index = model.predict(input_df)[0]
    pred_probs = model.predict_proba(input_df)[0]
    risk_class = encoder.inverse_transform([pred_index])[0]
    
    st.success(f"Predicted Risk Class: **{risk_class.upper()}**")
    
    st.markdown("### Prediction Probabilities")
    for i, class_name in enumerate(encoder.classes_):
        st.write(f"{class_name}: {pred_probs[i]:.2%}")
