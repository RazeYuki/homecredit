import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD DEPLOY MODEL (PIPELINE)
# ==============================
model = joblib.load("logistic_pipeline_deploy.pkl")

st.set_page_config(
    page_title="Prediksi Persetujuan Pinjaman",
    layout="centered"
)

# ==============================
# HEADER
# ==============================
st.title("🏦 Prediksi Persetujuan Pinjaman")

st.markdown("""
Aplikasi ini digunakan untuk **menilai tingkat risiko pengajuan pinjaman**
berdasarkan data dasar pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**,  
bukan keputusan final dari lembaga keuangan.*
""")

st.markdown("---")

# ==============================
# INPUT DATA PEMOHON
# ==============================
st.subheader("📝 Data Pemohon")

income = st.number_input(
    "Pendapatan Pemohon per Bulan (Rp)",
    min_value=0,
    value=5_000_000,
    step=100_000,
    format="%d",
    help="Total pendapatan pemohon setiap bulan"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman yang Diajukan (Rp)",
    min_value=0,
    value=1_500_000,
    step=500_000,
    format="%d",
    help="Total dana pinjaman yang diajukan"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0,
    value=500_000,
    step=100_000,
    format="%d",
    help="Jumlah cicilan per bulan"
)

# ==============================
# RASIO CICILAN
# ==============================
dti = 0
if income > 0:
    dti = annuity / income
    st.caption(f"📌 Rasio cicilan terhadap pendapatan: **{dti:.0%}**")

    if dti > 0.4:
        st.warning(
            "⚠️ Angsuran relatif tinggi dibanding pendapatan. "
            "Hal ini dapat meningkatkan risiko penolakan."
        )
    else:
        st.success("✅ Rasio cicilan tergolong aman.")

# ==============================
# UMUR & LAMA BEKERJA
# ==============================
age = st.number_input(
    "Umur Pemohon (tahun)",
    min_value=18,
    value=30,
    step=1,
    format="%d",
    help="Umur pemohon dalam tahun"
)

years_employed = st.number_input(
    "Lama Bekerja (tahun)",
    min_value=0,
    value=3,
    step=1,
    format="%d",
    help="Jumlah tahun pemohon telah bekerja"
)

st.markdown("---")

# ==============================
# ANALISIS RISIKO
# ==============================
if st.button("🔍 Analisis Risiko Pinjaman"):

    # ⚠️ URUTAN HARUS SAMA DENGAN TRAINING
    input_array = np.array([[
        income,
        credit_amount,
        annuity,
        age,
        years_employed
    ]])

    probability = model.predict_proba(input_array)[0][1]

    # ==============================
    # RISK-BASED OUTPUT (MASUK AKAL)
    # ==============================
    if probability >= 0.5:
        risk_level = "🟢 Risiko Rendah"
        decision = "DISETUJUI"
        color = "success"
    elif probability >= 0.3:
        risk_level = "🟡 Risiko Sedang"
        decision = "PERLU PERTIMBANGAN"
        color = "warning"
    else:
        risk_level = "🔴 Risiko Tinggi"
        decision = "DITOLAK"
        color = "error"

    st.subheader("📊 Hasil Analisis")

    getattr(st, color)(f"**Keputusan:** {decision}")

    st.markdown(f"""
    **Tingkat Risiko:** {risk_level}  
    **Probabilitas Persetujuan:** `{probability:.2%}`
    """)

    # ==============================
    # PENJELASAN LOGIS (HEURISTIK)
    # ==============================
    st.subheader("🔎 Penjelasan Singkat")

    reasons = []

    if dti > 0.4:
        reasons.append("Angsuran cukup besar dibanding pendapatan.")
    if years_employed < 2:
        reasons.append("Lama bekerja masih relatif singkat.")
    if age < 21:
        reasons.append("Usia pemohon tergolong muda.")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write(
            "Profil pemohon menunjukkan kondisi yang relatif stabil "
            "berdasarkan data yang tersedia."
        )

    st.info("""
    ⚠️ **Catatan Penting:**  
    Hasil analisis ini didasarkan pada pola data historis dan digunakan
    sebagai **pendukung keputusan**, bukan keputusan mutlak.
    """)
