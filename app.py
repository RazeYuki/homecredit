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
Aplikasi ini membantu **menilai tingkat risiko pengajuan pinjaman**
berdasarkan data pemohon menggunakan model *Logistic Regression*.

📌 *Aplikasi ini adalah **sistem pendukung keputusan**, bukan keputusan final dari bank.*
""")

st.markdown("---")

# ==============================
# INPUT DATA PEMOHON
# ==============================
st.subheader("📝 Data Pemohon")

income = st.number_input(
    "Pendapatan Pemohon per Bulan (Rp)",
    min_value=0,
    step=100_000,
    format="%d"
)

credit_amount = st.number_input(
    "Jumlah Pinjaman yang Diajukan (Rp)",
    min_value=0,
    step=500_000,
    format="%d"
)

annuity = st.number_input(
    "Angsuran Bulanan (Rp)",
    min_value=0,
    step=100_000,
    format="%d"
)

# ===== RASIO CICILAN =====
dti = 0
if income > 0:
    dti = annuity / income
    st.caption(f"📌 Rasio cicilan terhadap pendapatan: **{dti:.0%}**")
    if dti > 0.4:
        st.warning("⚠️ Angsuran cukup tinggi dibanding pendapatan.")

age = st.number_input(
    "Umur Pemohon (tahun)",
    min_value=0,
    step=1,
    format="%d"
)

years_employed = st.number_input(
    "Lama Bekerja (tahun)",
    min_value=0,
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

ext_choice_1 = st.selectbox(
    "Riwayat Kredit – Sumber Eksternal 1",
    credit_score_map.keys()
)

ext_choice_2 = st.selectbox(
    "Riwayat Kredit – Sumber Eksternal 2",
    credit_score_map.keys()
)

ext_source_1 = credit_score_map[ext_choice_1]
ext_source_2 = credit_score_map[ext_choice_2]

st.info("""
ℹ️ **Tentang Riwayat Kredit**  
Pada praktik perbankan, skor riwayat kredit diperoleh langsung dari lembaga penilai kredit.
Dalam aplikasi ini, nilai digunakan sebagai **simulasi** untuk menunjukkan pengaruh
riwayat kredit terhadap tingkat risiko pinjaman.
""")

st.markdown("---")

# ==============================
# PREDIKSI & INTERPRETASI
# ==============================
if st.button("🔍 Analisis Risiko Pinjaman"):

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

    # ==============================
    # RISK-BASED DECISION
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
    # PENJELASAN HASIL (LOGIS & AWAM)
    # ==============================
    st.subheader("🔎 Penjelasan Singkat")

    explanations = []

    if dti > 0.4:
        explanations.append("Angsuran relatif besar dibanding pendapatan.")
    if years_employed < 2:
        explanations.append("Lama bekerja masih tergolong singkat.")
    if age < 21:
        explanations.append("Usia pemohon relatif muda.")
    if ext_source_1 < 0.5 or ext_source_2 < 0.5:
        explanations.append("Riwayat kredit eksternal menunjukkan risiko.")

    if explanations:
        for e in explanations:
            st.write(f"• {e}")
    else:
        st.write(
            "Profil pemohon menunjukkan karakteristik yang relatif stabil, "
            "namun keputusan tetap mempertimbangkan pola risiko historis."
        )

    st.info("""
    ⚠️ **Catatan Penting:**  
    Hasil ini merupakan **analisis berbasis data historis** dan digunakan
    sebagai **pendukung keputusan**, bukan keputusan mutlak lembaga keuangan.
    """)
