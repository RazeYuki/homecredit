import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD PIPELINE MODEL
# ==============================
model = joblib.load("logistic_pipeline.pkl")

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
Aplikasi ini digunakan untuk **memprediksi persetujuan pinjaman**
menggunakan model *Logistic Regression*.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final.*
""")

st.markdown("---")

# ==============================
# INPUT DATA (SESUIAI TRAINING RAW FEATURES)
# ==============================

income = st.number_input(
    "Pendapatan Pemohon per Bulan (Rp)",
    min_value=0,
    value=0,
    step=100_000,
    format="%d"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman yang Diajukan (Rp)",
    min_value=0,
    value=0,
    step=500_000,
    format="%d"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0,
    value=0,
    step=100_000,
    format="%d"
)

# ===== RASIO CICILAN =====
if income > 0:
    dti = annuity / income
    st.caption(f"📌 Rasio cicilan terhadap pendapatan: **{dti:.0%}**")
    if dti > 0.4:
        st.warning(
            "⚠️ Angsuran cukup tinggi dibanding pendapatan."
        )

# ===== LAMA BEKERJA =====
employment_days = st.number_input(
    "Lama Bekerja (hari | 1 tahun = 365 hari)",
    min_value=0,
    value=0,
    step=30,
    format="%d"
)
st.caption(f"📌 Perkiraan lama bekerja: **{employment_days / 365:.1f} tahun**")

# ===== UMUR =====
age_days = st.number_input(
    "Umur Pemohon (hari | 1 tahun = 365 hari)",
    min_value=0,
    value=0,
    step=365,
    format="%d"
)
st.caption(f"📌 Perkiraan umur pemohon: **{age_days / 365:.1f} tahun**")

# ===== RIWAYAT PINJAMAN =====
prev_app_count = st.number_input(
    "Jumlah Pengajuan Pinjaman Sebelumnya",
    min_value=0,
    value=0,
    step=1,
    format="%d"
)

bureau_loan_count = st.number_input(
    "Jumlah Pinjaman Aktif (di Lembaga Kredit)",
    min_value=0,
    value=0,
    step=1,
    format="%d"
)

st.caption(
    "📌 Lembaga kredit (credit bureau) adalah institusi yang mencatat "
    "riwayat pinjaman seseorang, seperti bank atau lembaga pembiayaan."
)

st.markdown("---")

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Persetujuan"):

    # ⚠️ URUTAN HARUS SAMA DENGAN TRAINING
    input_array = np.array([[
        income,
        credit_amount,
        annuity,
        employment_days,
        age_days,
        prev_app_count,
        bureau_loan_count
    ]])

    # Pipeline otomatis handle scaling
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
