import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD DEPLOY PIPELINE
# ==============================
model = joblib.load("logistic_pipeline_deploy.pkl")

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
berdasarkan beberapa informasi utama pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final.*
""")

st.markdown("---")

# ==============================
# INPUT DATA PEMOHON
# ==============================

income = st.number_input(
    "Pendapatan Pemohon per Bulan (Rp)",
    min_value=0,
    value=0,
    step=100_000,
    format="%d",
    help="Total pendapatan pemohon setiap bulan"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman yang Diajukan (Rp)",
    min_value=0,
    value=0,
    step=500_000,
    format="%d",
    help="Total dana pinjaman yang diajukan"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0,
    value=0,
    step=100_000,
    format="%d",
    help="Jumlah cicilan yang harus dibayar setiap bulan"
)

# ===== RASIO CICILAN =====
if income > 0:
    dti = annuity / income
    st.caption(f"📌 Rasio cicilan terhadap pendapatan: **{dti:.0%}**")
    if dti > 0.4:
        st.warning(
            "⚠️ Angsuran cukup tinggi dibanding pendapatan. "
            "Hal ini dapat meningkatkan risiko penolakan."
        )

# ===== UMUR =====
age = st.number_input(
    "Umur Pemohon (tahun)",
    min_value=0,
    value=0,
    step=1,
    format="%d",
    help="Umur pemohon dalam tahun"
)

# ===== LAMA BEKERJA =====
years_employed = st.number_input(
    "Lama Bekerja (tahun)",
    min_value=0,
    value=0,
    step=1,
    format="%d",
    help="Jumlah tahun pemohon telah bekerja"
)

# ===== EXT SOURCE =====
ext_source_1 = st.number_input(
    "Skor Kredit Eksternal 1",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Skor kredit eksternal (0–1)"
)

ext_source_2 = st.number_input(
    "Skor Kredit Eksternal 2",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Skor kredit eksternal (0–1)"
)

st.markdown("---")

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Persetujuan"):

    # ⚠️ URUTAN HARUS SESUAI TRAINING
    input_array = np.array([[
        income,
        credit_amount,
        annuity,
        age,
        years_employed,
        ext_source_1,
        ext_source_2
    ]])

    probability = model.predict_proba(input_array)[0][1]
    prediction = model.predict(input_array)[0]

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

