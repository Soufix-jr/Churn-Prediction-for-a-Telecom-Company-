import streamlit as st
import pandas as pd
import joblib

# -------- Feature engineering --------
def create_features(df):
    df = df.copy()

    df["price_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["total_services"] = (df[service_cols] == "Yes").sum(axis=1)

    df["internet_no_protection"] = (
        (df["InternetService"] != "No") &
        (df["OnlineSecurity"] == "No") &
        (df["TechSupport"] == "No")
    ).astype(int)

    df["tenure_segment"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 60, 100],
        labels=["<1yr", "1-2yr", "2-5yr", ">5yr"]
    )

    df["senior_with_dependents"] = (
        (df["SeniorCitizen"] == 1) &
        (df["Dependents"] == "Yes")
    ).astype(int)

    return df


# -------- App --------
st.set_page_config(page_title="Churn Prediction")
st.title("📉 Telco Churn Prediction")

import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "xgboost_churn.pkl"
)

model = joblib.load(MODEL_PATH)


tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 18, 120, 70)
total = st.number_input("Total Charges", value=0.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["Yes", "No"])
support = st.selectbox("Tech Support", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

if st.button("Predict churn"):
    df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "OnlineSecurity": security,
        "TechSupport": support,
        "PaymentMethod": payment,
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PaperlessBilling": "Yes",
    }])

    df = create_features(df)
    df = pd.get_dummies(df)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    proba = model.predict_proba(df)[0][1]
    st.metric("Churn probability", f"{proba:.2%}")
