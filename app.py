import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Simulator Risiko Mesin",
    page_icon="⚙️",
    layout="wide"
)

# =========================
# LOAD CSS
# =========================
def load_css(file_name):
    css_path = Path(file_name)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# =========================
# LOAD MODEL DAN SCALER
# =========================
model = joblib.load("model_risiko_v1.joblib")
scaler = joblib.load("scaler_risiko_v1.joblib")

# Data training sesuai kode dosen
X_train = np.array([
    [60, 2],
    [70, 4],
    [80, 6],
    [90, 8],
    [100, 10]
])

TRAIN_MEAN_REF = np.mean(X_train)

# =========================
# FUNGSI DATA DRIFT
# =========================
def check_data_drift(new_data, train_mean, threshold=2.0):
    drift = np.abs(np.mean(new_data) - np.mean(train_mean))

    if drift > threshold:
        return "WARNING", f"Terdeteksi data drift sebesar {drift:.2f}. Model perlu dievaluasi ulang."
    else:
        return "STABIL", f"Data input masih stabil. Nilai drift sebesar {drift:.2f}."

# =========================
# FUNGSI ANONYMIZATION
# =========================
def clean_sensitive_data(df_input):
    cols_to_remove = ["Nama_Operator", "NIK_Petugas", "Alamat"]
    return df_input.drop(columns=[c for c in cols_to_remove if c in df_input.columns], errors="ignore")

# =========================
# FUNGSI KEPUTUSAN
# =========================
def decision_logic(risk_score):
    if risk_score >= 70:
        return "Risiko Tinggi", "Prioritas 1", "Lakukan pemeriksaan mesin segera dan prioritaskan pemeliharaan."
    elif risk_score >= 30:
        return "Risiko Sedang", "Prioritas 2", "Lakukan monitoring berkala dan jadwalkan pemeriksaan teknis."
    else:
        return "Risiko Rendah", "Prioritas 3", "Mesin masih aman, cukup lakukan pemantauan rutin."

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="hero">
        <p class="badge">Minggu 15 • Finalisasi Proyek dan MLOps</p>
        <h1>Simulator Risiko Kegagalan Mesin</h1>
        <p>
            Aplikasi ini memuat model machine learning dari file Joblib, melakukan inference,
            monitoring data drift, dan memberikan rekomendasi keputusan berbasis risiko.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# INPUT DAN INFORMASI
# =========================
col_input, col_info = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("Input Data Sensor")

    suhu = st.number_input(
        "Suhu Mesin",
        min_value=0.0,
        max_value=300.0,
        value=85.0,
        step=1.0
    )

    getaran = st.number_input(
        "Getaran Mesin",
        min_value=0.0,
        max_value=100.0,
        value=7.0,
        step=0.5
    )

    st.caption("Input yang digunakan hanya data teknis mesin, bukan data pribadi operator.")

    proses = st.button("Jalankan Simulasi Risiko", use_container_width=True)

with col_info:
    st.subheader("Informasi Sistem")

    st.write(
        """
        Sistem ini menerapkan alur MLOps sederhana. Model dan scaler disimpan ke dalam file,
        lalu dimuat ulang pada aplikasi Streamlit. Data baru diproses menggunakan scaler yang sama
        sebelum masuk ke model prediksi.
        """
    )

    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Model</h4>
                <p>Linear Regression</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Preprocessing</h4>
                <p>StandardScaler</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info3:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Deployment</h4>
                <p>Streamlit Cloud</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# HASIL SIMULASI
# =========================
if proses:
    st.markdown("---")
    st.subheader("Hasil Simulasi Risiko")

    raw_df = pd.DataFrame({
        "Nama_Operator": ["Disembunyikan"],
        "NIK_Petugas": ["Disembunyikan"],
        "Suhu": [suhu],
        "Getaran": [getaran]
    })

    cleaned_df = clean_sensitive_data(raw_df)

    data_baru = np.array([[suhu, getaran]])
    data_scaled = scaler.transform(data_baru)
    risk_score = model.predict(data_scaled)[0]

    kategori, prioritas, rekomendasi = decision_logic(risk_score)
    status_drift, pesan_drift = check_data_drift(data_baru, TRAIN_MEAN_REF)

    hasil1, hasil2, hasil3 = st.columns(3)

    with hasil1:
        st.metric("Skor Risiko", f"{risk_score:.2f}")

    with hasil2:
        st.metric("Kategori Risiko", kategori)

    with hasil3:
        st.metric("Prioritas", prioritas)

    if kategori == "Risiko Tinggi":
        st.error(rekomendasi)
    elif kategori == "Risiko Sedang":
        st.warning(rekomendasi)
    else:
        st.success(rekomendasi)

    st.subheader("Monitoring Data Drift")

    if status_drift == "WARNING":
        st.warning(pesan_drift)
    else:
        st.success(pesan_drift)

    st.subheader("Data Setelah Anonymization")
    st.dataframe(cleaned_df, use_container_width=True)

    st.subheader("Interpretasi Singkat")
    st.write(
        f"""
        Berdasarkan input suhu mesin sebesar **{suhu}** dan getaran mesin sebesar **{getaran}**,
        model menghasilkan skor risiko sebesar **{risk_score:.2f}**. Hasil tersebut masuk kategori
        **{kategori}**, sehingga rekomendasi sistem adalah **{rekomendasi}**
        """
    )

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="footer">
        Project Praktikum Minggu 15: Persistensi Model, Inference, Monitoring Drift,
        Etika Data, dan Deployment Streamlit.
    </div>
    """,
    unsafe_allow_html=True
)