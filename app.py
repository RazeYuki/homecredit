import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# ===== HEADER =====
st.title("🏦 Prediksi Persetujuan Pinjaman")
st.markdown("""
Aplikasi ini digunakan untuk **memprediksi apakah pengajuan pinjaman akan disetujui atau ditolak**  
berdasarkan data keuangan dan riwayat pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final.*
""")

st.markdown("---")

# ===== INPUT SECTION =====
st.subheader("📝 Data Pemohon")

income = st.number_input(
    "Pendapatan Pemohon (Rp)",
    min_value=0.0,
    help="Total pendapatan pemohon per bulan"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman (Rp)",
    min_value=0.0,
    help="Total dana yang diajukan oleh pemohon"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0.0,
    help="Jumlah cicilan yang harus dibayar setiap bulan"
)

employment_days = st.number_input(
    "Lama Bekerja (hari)",
    min_value=0,
    help="Total hari pemohon telah bekerja di tempat saat ini"
)

age_days = st.number_input(
    "Umur Pemohon (hari)",
    min_value=0,
    help="Umur pemohon dalam satuan hari"
)

prev_app_count = st.number_input(
    "Jumlah Pengajuan Sebelumnya",
    min_value=0,
    help="Jumlah pengajuan pinjaman yang pernah dilakukan sebelumnya"
)

bureau_loan_count = st.number_input(
    "Jumlah Pinjaman di Bureau",
    min_value=0,
    help="Jumlah pinjaman aktif yang tercatat di lembaga kredit"
)

st.markdown("---")

# ===== PREDICTION =====
if st.button("🔍 Prediksi Persetujuan"):
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

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.success("✅ **Pinjaman Diprediksi DISETUJUI**")
    else:
        st.error("❌ **Pinjaman Diprediksi DITOLAK**")

    st.markdown(f"""
    **Probabilitas Persetujuan:** `{probability:.2%}`

    🔎 *Semakin tinggi nilai probabilitas, semakin besar kemungkinan pinjaman disetujui.*
    """)

    st.info("""
    ⚠️ **Catatan:**  
    Hasil prediksi ini bersifat **pendukung keputusan**, bukan keputusan mutlak.
    Keputusan akhir tetap berada pada pihak lembaga keuangan.
    """)
