import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL PIPELINE DEPLOY
# ==============================
model = joblib.load("logistic_pipeline_deploy.pkl")

st.set_page_config(
    page_title="Analisis Risiko Pinjaman",
    layout="centered"
)

# ==============================
# HEADER
# ==============================
st.title("🏦 Analisis Risiko Pinjaman")

st.markdown("""
Aplikasi ini digunakan untuk **menilai tingkat risiko pengajuan pinjaman**
berdasarkan data dasar pemohon menggunakan model *Logistic Regression*.

📌 *Hasil yang ditampilkan berupa **skor risiko relatif**,  
bukan probabilitas mutlak persetujuan.*
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
    format="%d"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman yang Diajukan (Rp)",
    min_value=0,
    value=1_500_000,
    step=500_000,
    format="%d"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0,
    value=300_000,
    step=100_000,
    format="%d"
)

# ==============================
# RASIO CICILAN
# ==============================
dti = 0
if income > 0:
    dti = annuity / income
    st.caption(f"📌 Rasio cicilan terhadap pendapatan: **{dti:.0%}**")

    if dti > 0.4:
        st.warning("⚠️ Angsuran relatif tinggi dibanding pendapatan.")
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
    format="%d"
)

years_employed = st.number_input(
    "Lama Bekerja (tahun)",
    min_value=0,
    value=3,
    step=1,
    format="%d"
)

st.markdown("---")

# ==============================
# FUNGSI SKOR RISIKO
# ==============================
def risk_score_from_logit(logit):
    """
    Mengubah logit Logistic Regression
    menjadi skor risiko relatif 0–100
    """
    score = 1 / (1 + np.exp(-logit))
    return int(score * 100)

# ==============================
# ANALISIS RISIKO
# ==============================
if st.button("🔍 Analisis Risiko Pinjaman"):

    # ⚠️ URUTAN HARUS SESUAI TRAINING
    input_array = np.array([[
        income,
        credit_amount,
        annuity,
        age,
        years_employed
    ]])

    # ==============================
    # AMBIL LOGIT SCORE (BUKAN PROBABILITAS)
    # ==============================
    scaled_input = model.named_steps["scaler"].transform(input_array)
    logit_score = model.named_steps["model"].decision_function(scaled_input)[0]

    risk_score = risk_score_from_logit(logit_score)

    # ==============================
    # KATEGORI RISIKO
    # ==============================
    if risk_score >= 60:
        risk_level = "🟢 Risiko Rendah"
        decision = "LAYAK DIPERTIMBANGKAN"
        color = "success"
    elif risk_score >= 40:
        risk_level = "🟡 Risiko Sedang"
        decision = "PERLU PERTIMBANGAN LANJUT"
        color = "warning"
    else:
        risk_level = "🔴 Risiko Tinggi"
        decision = "BERISIKO TINGGI"
        color = "error"

    # ==============================
    # OUTPUT
    # ==============================
    st.subheader("📊 Hasil Analisis Risiko")

    st.metric(
        label="Skor Risiko Kredit",
        value=f"{risk_score} / 100"
    )

    getattr(st, color)(f"**Kategori Risiko:** {risk_level}")
    st.write(f"**Rekomendasi:** {decision}")

    # ==============================
    # PENJELASAN LOGIS
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
            "Profil pemohon menunjukkan kondisi finansial yang relatif stabil "
            "berdasarkan data yang tersedia."
        )

    st.info("""
    ⚠️ **Catatan Penting:**  
    Skor risiko ini bersifat **relatif** dan digunakan sebagai **pendukung keputusan**.
    Keputusan akhir tetap berada pada pihak lembaga keuangan.
    """)
