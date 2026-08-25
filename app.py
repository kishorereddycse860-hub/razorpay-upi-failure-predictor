# app.py

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# STEP 7: Streamlit Demo App
# -----------------------------

st.title("UPI Payment Failure Predictor")
st.write("Enter transaction details to predict whether it will SUCCEED or FAIL.")

# Load trained model and training columns (to match structure exactly)
rf_model = joblib.load("data/random_forest_model.pkl")
X_train = pd.read_csv("data/X_train.csv")
model_columns = X_train.columns

# --- Get list of banks from training columns ---
# Why: We extract bank names from the one-hot encoded column names
# (e.g., "sender_bank_okaxis" -> "okaxis") to build clean dropdowns.
sender_banks = [col.replace("sender_bank_", "") for col in model_columns if col.startswith("sender_bank_")]
receiver_banks = [col.replace("receiver_bank_", "") for col in model_columns if col.startswith("receiver_bank_")]

# --- User inputs ---
amount = st.number_input("Amount (INR)", min_value=1.0, max_value=100000.0, value=5000.0)
hour = st.slider("Hour of day (0-23)", 0, 23, 14)
day_of_week = st.selectbox("Day of week", options=list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
sender_bank = st.selectbox("Sender Bank", options=sender_banks)
receiver_bank = st.selectbox("Receiver Bank", options=receiver_banks)

if st.button("Predict"):
    # --- Build input row matching training structure exactly ---
    # Why: The model expects the EXACT same columns it was trained on.
    # We start with all zeros, then set the relevant fields.
    input_row = pd.DataFrame(0, index=[0], columns=model_columns)

    input_row["Amount (INR)"] = amount
    input_row["hour"] = hour
    input_row["day_of_week"] = day_of_week
    input_row["same_bank"] = 1 if sender_bank == receiver_bank else 0

    sender_col = f"sender_bank_{sender_bank}"
    receiver_col = f"receiver_bank_{receiver_bank}"
    if sender_col in input_row.columns:
        input_row[sender_col] = 1
    if receiver_col in input_row.columns:
        input_row[receiver_col] = 1

    # --- Predict ---
    prediction = rf_model.predict(input_row)[0]
    probability = rf_model.predict_proba(input_row)[0]

    if prediction == 1:
        st.error(f"Prediction: FAILED (confidence: {probability[1]:.2%})")
    else:
        st.success(f"Prediction: SUCCESS (confidence: {probability[0]:.2%})")

    st.write("Model confidence breakdown:", {"SUCCESS": f"{probability[0]:.2%}", "FAILED": f"{probability[1]:.2%}"})