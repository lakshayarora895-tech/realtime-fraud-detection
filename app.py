import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="FinShield India", page_icon="🇮🇳", layout="wide")

@st.cache_resource
def load_assets():
    rf = joblib.load('fraud_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf, scaler

rf_model, scaler = load_assets()

# --- INDIAN TRANSLATION BRIDGE ---
# 1. Currency mapping (1 Euro = approx 90 INR)
EURO_TO_INR = 90.0
# 2. Indian Behavior mapping (Lower average transaction due to heavy UPI usage)
INDIAN_MEAN_TXN = 500.0 

st.title("🇮🇳 FinShield: Indian Payment Gateway")
st.markdown("Real-time fraud detection localized for INR, UPI, and domestic banking behaviors.")

st.markdown("### 💳 Live Transaction Terminal")

with st.form("manual_input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        amount_inr = st.number_input("Amount (₹)", min_value=0.0, value=250.0, step=50.0)
        hour_ist = st.slider("Hour of Day (IST)", min_value=0, max_value=23, value=14, help="14 = 2:00 PM IST")
        payment_type = st.selectbox("Routing Network", ["UPI", "RuPay", "Visa/Mastercard"])
        
    with col2:
        st.subheader("Encrypted Network Vectors")
        st.write("Simulated PCA components from the payment gateway.")
        v1 = st.number_input("Vector 1 (V1)", value=-1.50)
        v2 = st.number_input("Vector 2 (V2)", value=2.10)
        v3 = st.number_input("Vector 3 (V3)", value=-3.00)
        
    submitted = st.form_submit_button("Authenticate Transaction", type="primary")

if submitted:
    with st.spinner("Analyzing banking vectors..."):
        # 1. TRANSLATION: Map Indian context to the AI's mathematical baseline
        base_amount = amount_inr / EURO_TO_INR
        
        # 2. FEATURE ENGINEERING (Using local Indian logic)
        log_amount = np.log1p(base_amount)
        
        # Velocity ratio: How big is this compared to a normal Indian transaction?
        amt_ratio = amount_inr / INDIAN_MEAN_TXN 
        
        # 3. BUILD THE PAYLOAD
        input_data = {f'V{i}': 0.0 for i in range(1, 29)}
        input_data['V1'] = v1
        input_data['V2'] = v2
        input_data['V3'] = v3
        
        input_data['Hour'] = hour_ist
        input_data['Log_Amount'] = log_amount
        input_data['Amount_To_Mean_Ratio'] = amt_ratio
        
        df_live = pd.DataFrame([input_data])
        
        # 4. SCALE THE DATA
        cols_to_scale = ['Hour', 'Log_Amount', 'Amount_To_Mean_Ratio']
        df_live[cols_to_scale] = scaler.transform(df_live[cols_to_scale])
        
        # 5. AI INFERENCE
        probability = rf_model.predict_proba(df_live)[0][1]
        
        st.markdown("---")
        st.subheader("Gateway Decision")
        
        # Strict threshold for domestic transactions
        if probability >= 0.40:
            st.error("🚨 **TRANSACTION BLOCKED: SUSPICIOUS ACTIVITY**")
            st.write(f"**Threat Level:** {probability:.2%}")
            st.caption(f"System Note: ₹{amount_inr:,.2f} via {payment_type} was intercepted. Vectors matched known cybercrime signatures.")
        else:
            st.success("✅ **PAYMENT AUTHORIZED**")
            st.write(f"**Threat Level:** {probability:.2%}")
            st.caption(f"Cleared for standard inter-bank processing.")
