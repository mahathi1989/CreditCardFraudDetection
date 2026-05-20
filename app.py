import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", 
                   page_icon="💳", 
                   layout="wide")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
sample_df = pd.read_csv("creditcard.csv")

# Title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter a transaction index to check if it is fraudulent or not")
st.write("---")

# Model stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "Logistic Regression")
with col2:
    st.metric("ROC-AUC Score", "0.98")
with col3:
    st.metric("Total Transactions", "284,807")

st.write("---")

# Input
sample_index = st.number_input(
    f"Enter Transaction Index (0 to {len(sample_df)-1})",
    min_value=0,
    max_value=len(sample_df)-1,
    value=0
)
sample_index = int(sample_index)

# Show transaction details
sample = sample_df.iloc[sample_index]
st.write("### Transaction Details:")

col1, col2 = st.columns(2)
with col1:
    st.metric("Amount", f"${sample['Amount']:.2f}")
with col2:
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    transaction_time = base_time + timedelta(seconds=sample['Time'])
    st.metric("Time", transaction_time.strftime("%H:%M"))

st.write("---")

# Predict button
if st.button("🔍 Predict"):
    input_data = sample_df.iloc[[sample_index]].drop(
                 columns=['Class'], errors='ignore')
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 FRAUD Transaction Detected!")
    else:
        st.success("✅ Normal Transaction - Safe!")