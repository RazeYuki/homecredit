import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL & SCALER
# ==============================
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# ==============================
# HEADER
# ==============================
st.title("🏦 Prediksi Persetujuan Pinjaman")

st.markdown("""
Aplikasi ini digunakan untuk **memprediksi apakah pengajuan pinjaman akan disetujui atau ditolak**  
berdasarkan data keuangan dan riwayat pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final.*
""")

st.markdown("---")

# ==============================
# INPUT DATA PEMOHON
# ==============================
st.subheader("📝 Data Pemohon")

income = st.number_input(
    "Pendapatan Pemohon per Bulan (Rp)",
    min_value=0.0,
    help="Total pendapatan pemohon setiap bulan"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman yang Diajukan (Rp)",
    min_value=0.0,
    help="Total dana pinjaman yang diajukan"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0.0,
    help="Jumlah cicilan yang harus dibayarkan setiap bulan"
)

# ===== LAMA BEKERJA =====
employment_days = st.number_input(
    "Lama Bekerja (dalam hari 1 tahun = 365 hari)",
    min_value=0,
    help="Contoh: 1 tahun ≈ 365 hari, 5 tahun ≈ 1825 hari"
)

employment_years = employment_days / 365
st.caption(f"📌 Perkiraan lama bekerja: **{employment_years:.1f} tahun**")

# ===== UMUR =====
age_days = st.number_input(
    "Umur Pemohon (dalam hari 1 tahun = 365 hari)",
    min_value=0,
    help="Contoh: 25 tahun ≈ 9125 hari"
)

age_years = age_days / 365
st.caption(f"📌 Perkiraan umur pemohon: **{age_years:.1f} tahun**")

# ===== RIWAYAT PINJAMAN =====
prev_app_count = st.number_input(
    "Jumlah Pengajuan Pinjaman Sebelumnya",
    min_value=0,
    help="Jumlah pengajuan pinjaman yang pernah dilakukan sebelumnya"
)

bureau_loan_count = st.number_input(
    "Jumlah Pinjaman Aktif (di Lembaga Kredit)",
    min_value=0,
    help=(
        "Jumlah pinjaman yang masih aktif dan tercatat di lembaga penyedia "
        "informasi kredit, seperti bank, leasing, atau kartu kredit"
    )
)

st.caption(
    "📌 Lembaga kredit (credit bureau) adalah institusi yang mencatat riwayat "
    "pinjaman seseorang, seperti bank atau lembaga pembiayaan."
)

st.markdown("---")

# ==============================
# PREDIKSI
# ==============================
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

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
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
    ⚠️ **Catatan Penting:**  
    Hasil prediksi ini bersifat **pendukung keputusan**, bukan keputusan mutlak.
    Keputusan akhir tetap berada pada pihak lembaga keuangan.
    """)
