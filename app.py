import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL PIPELINE (DEPLOY)
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

# ==============================
# SKOR RIWAYAT KREDIT (DROPDOWN)
# ==============================
st.subheader("📊 Riwayat Kredit")

credit_score_options = {
    "🟢 Baik (riwayat pembayaran lancar)": 0.8,
    "🟡 Sedang (pernah menunggak kecil)": 0.5,
    "🔴 Berisiko (sering menunggak)": 0.2
}

ext_choice_1 = st.selectbox(
    "Riwayat Kredit – Sumber Eksternal 1",
    list(credit_score_options.keys()),
    help="Penilaian riwayat kredit dari lembaga eksternal"
)

ext_choice_2 = st.selectbox(
    "Riwayat Kredit – Sumber Eksternal 2",
    list(credit_score_options.keys()),
    help="Penilaian tambahan dari sumber eksternal lain"
)

# Konversi dropdown ke angka
ext_source_1 = credit_score_options[ext_choice_1]
ext_source_2 = credit_score_options[ext_choice_2]

st.info("""
ℹ️ **Tentang Riwayat Kredit**  
Dalam praktik perbankan, skor riwayat kredit diperoleh langsung dari lembaga penilai kredit.
Pada aplikasi ini, nilai digunakan sebagai **simulasi** untuk menunjukkan pengaruh riwayat kredit
terhadap persetujuan pinjaman.
""")

st.markdown("---")

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Persetujuan"):

    # ⚠️ URUTAN SESUAI TRAINING DEPLOY MODEL
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

    🔎 *Semakin tinggi probabilitas, semakin besar peluang pinjaman disetujui.*
    """)

    st.info("""
    ⚠️ **Catatan Penting:**  
    Hasil prediksi ini bersifat **pendukung keputusan** dan bukan keputusan mutlak.
    Keputusan akhir tetap berada pada pihak lembaga keuangan.
    """)
