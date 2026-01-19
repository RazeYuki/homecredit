import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# LOAD MODEL, SCALER, FEATURE NAMES
# ==============================
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# ==============================
# HEADER
# ==============================
st.title("🏦 Loan Approval Prediction")
st.markdown("### 📄 Data Pemohon")

st.markdown("""
Aplikasi ini digunakan untuk **memprediksi persetujuan pinjaman** berdasarkan
data keuangan dan riwayat pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final.*
""")

st.markdown("---")

# ==============================
# INPUT DATA (USER FRIENDLY)
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
            "Hal ini dapat meningkatkan risiko penolakan pinjaman."
        )

# ===== LAMA BEKERJA =====
employment_days = st.number_input(
    "Lama Bekerja (hari | 1 tahun = 365 hari)",
    min_value=0,
    value=0,
    step=30,
    format="%d",
    help="Contoh: 1 tahun ≈ 365 hari, 5 tahun ≈ 1825 hari"
)

st.caption(f"📌 Perkiraan lama bekerja: **{employment_days / 365:.1f} tahun**")

# ===== UMUR =====
age_days = st.number_input(
    "Umur Pemohon (hari | 1 tahun = 365 hari)",
    min_value=0,
    value=0,
    step=365,
    format="%d",
    help="Contoh: 25 tahun ≈ 9125 hari"
)

st.caption(f"📌 Perkiraan umur pemohon: **{age_days / 365:.1f} tahun**")

# ===== RIWAYAT PINJAMAN =====
prev_app_count = st.number_input(
    "Jumlah Pengajuan Pinjaman Sebelumnya",
    min_value=0,
    value=0,
    step=1,
    format="%d",
    help="Jumlah pengajuan pinjaman yang pernah dilakukan sebelumnya"
)

bureau_loan_count = st.number_input(
    "Jumlah Pinjaman Aktif (di Lembaga Kredit)",
    min_value=0,
    value=0,
    step=1,
    format="%d",
    help=(
        "Jumlah pinjaman yang masih aktif dan tercatat di lembaga "
        "penyedia informasi kredit, seperti bank, leasing, atau kartu kredit"
    )
)

st.caption(
    "📌 Lembaga kredit (credit bureau) adalah institusi yang mencatat "
    "riwayat pinjaman seseorang."
)

st.markdown("---")

    st.info("""
    ⚠️ **Catatan Penting:**  
    Hasil prediksi ini bersifat **pendukung keputusan**, bukan keputusan mutlak.
    Keputusan akhir tetap berada pada pihak lembaga keuangan.
    """)
