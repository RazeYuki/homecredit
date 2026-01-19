import streamlit as st
import numpy as np
import joblib

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction")

income = st.number_input("Pendapatan Pemohon", min_value=0.0)
credit_amount = st.number_input("Jumlah Pinjaman", min_value=0.0)
annuity = st.number_input("Angsuran Bulanan", min_value=0.0)
employment_days = st.number_input("Lama Bekerja (hari)", value=0)
age_days = st.number_input("Umur (hari)", value=0)
prev_app_count = st.number_input("Jumlah Pengajuan Sebelumnya", min_value=0)
bureau_loan_count = st.number_input("Jumlah Pinjaman di Bureau", min_value=0)

if st.button("Prediksi"):
    input_data = np.array([[
        income,
        credit_amount,
        annuity,
        employment_days,
        age_days,
        prev_app_count,
        bureau_loan_count
    ]])

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.success(f"✅ DISETUJUI (Probabilitas: {prob:.2f})")
    else:
        st.error(f"❌ DITOLAK (Probabilitas: {prob:.2f})")
