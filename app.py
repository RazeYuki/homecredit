import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL & TRAIN LOGITS
# ==============================
model = joblib.load("logistic_pipeline_deploy.pkl")
train_logits = joblib.load("train_logits.pkl")

st.set_page_config(
    page_title="Analisis Risiko Pinjaman",
    layout="centered"
)

# ==============================
# HEADER
# ==============================
st.title("🏦 Analisis Risiko Pinjaman")

st.markdown("""
Aplikasi ini menilai **tingkat risiko pengajuan pinjaman secara relatif**
dengan membandingkan profil pemohon terhadap **data historis pemohon lain**.

📌 *Model digunakan untuk **pemeringkatan risiko (ranking)**,  
bukan untuk menghitung probabilitas persetujuan absolut.*
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
    step=1
)

years_employed = st.number_input(
    "Lama Bekerja (tahun)",
    min_value=0,
    value=3,
    step=1
)

st.markdown("---")

# ==============================
# ANALISIS RISIKO
# ==============================
if st.button("🔍 Analisis Risiko Pinjaman"):

    # URUTAN HARUS SAMA DENGAN TRAINING
    input_array = np.array([[
        income,
        credit_amount,
        annuity,
        age,
        years_employed
    ]])

    # ==============================
    # HITUNG LOGIT USER
    # ==============================
    user_logit = model.named_steps["model"].decision_function(
        model.named_steps["scaler"].transform(input_array)
    )[0]

    # ==============================
    # HITUNG PERSENTIL (KUNCI)
    # ==============================
    percentile = (train_logits < user_logit).mean() * 100

    # ==============================
    # OUTPUT
    # ==============================
    st.subheader("📊 Hasil Analisis Risiko")

    st.metric(
        label="Posisi Risiko Pemohon",
        value=f"{percentile:.0f} Persentil"
    )

    if percentile >= 70:
        st.success("🟢 Risiko Rendah – profil lebih aman dibanding mayoritas pemohon")
        recommendation = "LAYAK DIPERTIMBANGKAN"
    elif percentile >= 40:
        st.warning("🟡 Risiko Sedang – profil berada di kisaran rata-rata pemohon")
        recommendation = "PERLU PERTIMBANGAN LANJUT"
    else:
        st.error("🔴 Risiko Tinggi – profil lebih berisiko dibanding mayoritas pemohon")
        recommendation = "BERISIKO TINGGI"

    st.markdown(f"**Rekomendasi:** {recommendation}")

    # ==============================
    # PENJELASAN LOGIS
    # ==============================
    st.subheader("🔎 Penjelasan Singkat")

    explanations = []

    if dti > 0.4:
        explanations.append("Angsuran cukup besar dibanding pendapatan.")
    if years_employed < 2:
        explanations.append("Lama bekerja masih relatif singkat.")
    if age < 21:
        explanations.append("Usia pemohon tergolong muda.")

    if explanations:
        for e in explanations:
            st.write(f"• {e}")
    else:
        st.write(
            "Profil pemohon menunjukkan kondisi finansial yang relatif stabil "
            "berdasarkan data yang tersedia."
        )

    st.info("""
    ℹ️ **Catatan Penting**  
    Persentil risiko menunjukkan **posisi relatif pemohon dibanding data historis**.
    Semakin tinggi persentil, semakin rendah risiko relatif.
    Hasil ini digunakan sebagai **pendukung keputusan**, bukan keputusan mutlak.
    """)
