import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL PIPELINE DEPLOY
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
Aplikasi ini membantu memprediksi **apakah pengajuan pinjaman berpotensi disetujui atau ditolak**
berdasarkan informasi dasar pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final dari bank.*
""")

st.markdown("---")

# ==============================
# INPUT DATA PEMOHON
# ==============================
st.subheader("📝 Data Pemohon")

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
        st.warning("⚠️ Angsuran relatif tinggi dibanding pendapatan.")

# ===== UMUR & LAMA BEKERJA =====
age = st.number_input(
    "Umur Pemohon (tahun)",
    min_value=0,
    value=0,
    step=1,
    format="%d"
)

years_employed = st.number_input(
    "Lama Bekerja (tahun)",
    min_value=0,
    value=0,
    step=1,
    format="%d"
)

# ==============================
# RIWAYAT KREDIT (DROPDOWN)
# ==============================
st.subheader("📊 Riwayat Kredit")

credit_score_map = {
    "🟢 Baik (pembayaran lancar)": 0.8,
    "🟡 Sedang (pernah menunggak kecil)": 0.5,
    "🔴 Berisiko (sering menunggak)": 0.2
}

choice_1 = st.selectbox(
    "Riwayat Kredit – Sumber Eksternal 1",
    credit_score_map.keys()
)

choice_2 = st.selectbox(
    "Riwayat Kredit – Sumber Eksternal 2",
    credit_score_map.keys()
)

ext_source_1 = credit_score_map[choice_1]
ext_source_2 = credit_score_map[choice_2]

st.info("""
ℹ️ **Tentang Riwayat Kredit**  
Dalam praktik perbankan, skor riwayat kredit diperoleh dari lembaga penilai kredit.
Pada aplikasi ini, nilai digunakan sebagai **simulasi** untuk menunjukkan pengaruh
riwayat kredit terhadap persetujuan pinjaman.
""")

st.markdown("---")

# ==============================
# THRESHOLD (EDUKATIF)
# ==============================
st.subheader("⚙️ Pengaturan Keputusan")

threshold = st.slider(
    "Ambang batas persetujuan (threshold)",
    min_value=0.1,
    max_value=0.6,
    value=0.3,
    step=0.05,
    help=(
        "Threshold menentukan seberapa ketat model dalam menyetujui pinjaman. "
        "Semakin tinggi nilainya, semakin selektif keputusan."
    )
)

st.caption(f"📌 Pinjaman disetujui jika probabilitas ≥ **{threshold:.0%}**")

st.markdown("---")

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Persetujuan"):

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
    prediction = 1 if probability >= threshold else 0

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.success("✅ **Pinjaman Diprediksi DISETUJUI**")
    else:
        st.error("❌ **Pinjaman Diprediksi DITOLAK**")

    st.markdown(f"""
    **Probabilitas Persetujuan:** `{probability:.2%}`  
    **Threshold yang digunakan:** `{threshold:.0%}`
    """)

    # ==============================
    # ALASAN INTERPRETASI (HEURISTIK)
    # ==============================
    st.subheader("🔎 Interpretasi Hasil")

    reasons = []

    if dti > 0.4:
        reasons.append("Angsuran relatif tinggi dibanding pendapatan.")
    if years_employed < 2:
        reasons.append("Lama bekerja masih tergolong singkat.")
    if age < 21:
        reasons.append("Usia pemohon masih relatif muda.")
    if ext_source_1 < 0.5 or ext_source_2 < 0.5:
        reasons.append("Riwayat kredit eksternal menunjukkan risiko.")

    if reasons:
        st.warning("**Faktor yang memengaruhi hasil:**")
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.info(
            "Secara umum profil pemohon cukup baik, namun model tetap mempertimbangkan "
            "pola risiko historis pada data."
        )

    st.info("""
    ⚠️ **Catatan Penting:**  
    Hasil prediksi ini bersifat **pendukung keputusan** dan berbasis pola data historis.
    Keputusan akhir tetap berada pada pihak lembaga keuangan.
    """)
