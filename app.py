import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Simulator Risiko Kegagalan Mesin",
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
@st.cache_resource
def load_model_and_scaler():
    model_loaded = joblib.load("model_risiko_v1.joblib")
    scaler_loaded = joblib.load("scaler_risiko_v1.joblib")
    return model_loaded, scaler_loaded

model, scaler = load_model_and_scaler()

# =========================
# DATA TRAINING SESUAI PRAKTIKUM DOSEN
# =========================
FEATURE_NAMES = ["Suhu Mesin", "Getaran Mesin"]

X_train = np.array([
    [60, 2],
    [70, 4],
    [80, 6],
    [90, 8],
    [100, 10]
])

X_train_scaled = scaler.transform(X_train)
TRAIN_MEAN = np.mean(X_train, axis=0)
TRAIN_STD = np.std(X_train, axis=0)

# =========================
# FUNGSI DATA DRIFT
# =========================
def check_data_drift(new_data, train_mean, train_std, threshold=2.0):
    """
    Drift dicek berdasarkan z-score per fitur.
    Jika salah satu fitur melewati ±2 standar deviasi, sistem memberi warning.
    """
    z_scores = np.abs((new_data[0] - train_mean) / train_std)
    max_z = np.max(z_scores)

    if max_z > threshold:
        return (
            "WARNING",
            f"Terdeteksi data drift. Nilai penyimpangan maksimum adalah {max_z:.2f} standar deviasi. "
            "Model perlu dievaluasi ulang sebelum hasil digunakan sebagai dasar keputusan penting."
        )
    else:
        return (
            "STABIL",
            f"Data input masih sesuai dengan profil data training. "
            f"Nilai penyimpangan maksimum adalah {max_z:.2f} standar deviasi."
        )

# =========================
# FUNGSI ANONYMIZATION
# =========================
def clean_sensitive_data(df_input):
    cols_to_remove = ["Nama_Operator", "NIK_Petugas", "Alamat"]
    return df_input.drop(
        columns=[c for c in cols_to_remove if c in df_input.columns],
        errors="ignore"
    )

# =========================
# FUNGSI KATEGORI RISIKO
# =========================
def decision_logic(risk_score):
    if risk_score >= 70:
        return (
            "Risiko Tinggi",
            "Prioritas 1",
            "Lakukan pemeriksaan mesin segera dan prioritaskan pemeliharaan."
        )
    elif risk_score >= 30:
        return (
            "Risiko Sedang",
            "Prioritas 2",
            "Lakukan monitoring berkala dan jadwalkan pemeriksaan teknis."
        )
    else:
        return (
            "Risiko Rendah",
            "Prioritas 3",
            "Mesin masih aman, cukup lakukan pemantauan rutin."
        )

# =========================
# FUNGSI SPK SAW
# =========================
def calculate_saw_ranking(risk_score):
    """
    Metode SAW digunakan untuk membuat ranking rekomendasi tindakan.

    Kriteria:
    1. Risiko_Sisa = cost
    2. Biaya = cost
    3. Efektivitas = benefit
    4. Kecepatan = benefit
    """

    alternatives = pd.DataFrame({
        "Alternatif": [
            "Pemeriksaan Mesin Segera",
            "Monitoring Berkala",
            "Pemantauan Rutin"
        ],
        "Risiko_Sisa": [
            max(risk_score * 0.35, 5),
            max(risk_score * 0.65, 10),
            max(risk_score * 0.90, 15)
        ],
        "Biaya": [80, 45, 20],
        "Efektivitas": [95, 70, 40],
        "Kecepatan": [90, 65, 45]
    })

    bobot = {
        "Risiko_Sisa": 0.40,
        "Biaya": 0.20,
        "Efektivitas": 0.25,
        "Kecepatan": 0.15
    }

    normalisasi = alternatives.copy()

    # Cost: semakin kecil semakin baik
    normalisasi["Risiko_Sisa"] = alternatives["Risiko_Sisa"].min() / alternatives["Risiko_Sisa"]
    normalisasi["Biaya"] = alternatives["Biaya"].min() / alternatives["Biaya"]

    # Benefit: semakin besar semakin baik
    normalisasi["Efektivitas"] = alternatives["Efektivitas"] / alternatives["Efektivitas"].max()
    normalisasi["Kecepatan"] = alternatives["Kecepatan"] / alternatives["Kecepatan"].max()

    alternatives["Skor_SAW"] = (
        normalisasi["Risiko_Sisa"] * bobot["Risiko_Sisa"] +
        normalisasi["Biaya"] * bobot["Biaya"] +
        normalisasi["Efektivitas"] * bobot["Efektivitas"] +
        normalisasi["Kecepatan"] * bobot["Kecepatan"]
    )

    alternatives["Ranking"] = alternatives["Skor_SAW"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    alternatives = alternatives.sort_values("Ranking")

    return alternatives

# =========================
# FUNGSI SHAP / XAI
# =========================
def make_shap_explanation(data_scaled, data_asli):
    """
    SHAP digunakan untuk menjelaskan kontribusi fitur terhadap hasil prediksi.
    Nilai SHAP positif berarti fitur menaikkan skor risiko.
    Nilai SHAP negatif berarti fitur menurunkan skor risiko.
    """
    explainer = shap.Explainer(
        model,
        X_train_scaled,
        feature_names=FEATURE_NAMES
    )

    shap_values = explainer(data_scaled)
    nilai_shap = shap_values.values[0]

    shap_df = pd.DataFrame({
        "Fitur": FEATURE_NAMES,
        "Nilai Input": data_asli[0],
        "Nilai SHAP": nilai_shap,
        "Dampak": [
            "Meningkatkan risiko" if nilai > 0 else "Menurunkan risiko"
            for nilai in nilai_shap
        ]
    })

    shap_df["Pengaruh Absolut"] = shap_df["Nilai SHAP"].abs()
    shap_df = shap_df.sort_values("Pengaruh Absolut", ascending=False)

    return shap_df[["Fitur", "Nilai Input", "Nilai SHAP", "Dampak"]], shap_values

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="hero">
        <p class="badge">Minggu 16 • UAS Integrasi Akhir</p>
        <h1>Simulator Risiko Kegagalan Mesin</h1>
        <p>
            Aplikasi ini mengintegrasikan input Streamlit, preprocessing scaler,
            model machine learning, monitoring data drift, explainability menggunakan SHAP,
            anonymization, dan sistem pendukung keputusan berbasis metode SAW.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# INPUT DAN INFORMASI SISTEM
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

    proses = st.button("Jalankan Simulasi Risiko", width="stretch")

with col_info:
    st.subheader("Informasi Sistem")

    st.write(
        """
        Sistem ini menerapkan alur integrasi akhir. Data dari input pengguna diproses menggunakan scaler,
        kemudian diprediksi oleh model machine learning. Hasil prediksi digunakan sebagai dasar sistem
        pendukung keputusan untuk menghasilkan ranking rekomendasi tindakan.
        """
    )

    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown(
            """
            <div class="mini-card">
                <h4>Model ML</h4>
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
                <h4>SPK</h4>
                <p>SAW Ranking</p>
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

    # Contoh data mentah yang masih mengandung data sensitif
    raw_df = pd.DataFrame({
        "Nama_Operator": ["Disembunyikan"],
        "NIK_Petugas": ["Disembunyikan"],
        "Suhu": [suhu],
        "Getaran": [getaran]
    })

    cleaned_df = clean_sensitive_data(raw_df)

    # Data input untuk model
    data_baru = np.array([[suhu, getaran]])
    data_scaled = scaler.transform(data_baru)

    # Prediksi ML
    risk_score = float(model.predict(data_scaled)[0])

    # Kategori risiko
    kategori, prioritas, rekomendasi = decision_logic(risk_score)

    # Drift check
    status_drift, pesan_drift = check_data_drift(
        data_baru,
        TRAIN_MEAN,
        TRAIN_STD
    )

    hasil1, hasil2, hasil3 = st.columns(3)

    with hasil1:
        st.metric("Skor Risiko", f"{risk_score:.2f}")

    with hasil2:
        st.metric("Kategori Risiko", kategori)

    with hasil3:
        st.metric("Prioritas Awal", prioritas)

    if kategori == "Risiko Tinggi":
        st.error(rekomendasi)
    elif kategori == "Risiko Sedang":
        st.warning(rekomendasi)
    else:
        st.success(rekomendasi)

    # =========================
    # MONITORING DATA DRIFT
    # =========================
    st.subheader("Monitoring Data Drift")

    if status_drift == "WARNING":
        st.warning(pesan_drift)
    else:
        st.success(pesan_drift)

    # =========================
    # SPK SAW
    # =========================
    st.subheader("Ranking Rekomendasi Tindakan Menggunakan SAW")

    ranking_df = calculate_saw_ranking(risk_score)

    st.dataframe(
        ranking_df,
        hide_index=True,
        width="stretch"
    )

    rekomendasi_terbaik = ranking_df.iloc[0]["Alternatif"]

    st.success(
        f"Berdasarkan perhitungan SAW, rekomendasi tindakan terbaik adalah: {rekomendasi_terbaik}."
    )

    # =========================
    # XAI / SHAP
    # =========================
    st.subheader("Penjelasan Model Menggunakan SHAP")

    try:
        shap_df, shap_values = make_shap_explanation(data_scaled, data_baru)

        st.write(
            """
            Bagian ini menunjukkan kontribusi setiap fitur terhadap hasil prediksi menggunakan nilai SHAP.
            Nilai SHAP positif berarti fitur tersebut menaikkan skor risiko, sedangkan nilai SHAP negatif
            berarti fitur tersebut menurunkan skor risiko.
            """
        )

        st.dataframe(
            shap_df,
            hide_index=True,
            width="stretch"
        )

        st.write("Grafik SHAP berikut memperlihatkan arah dan besar kontribusi fitur terhadap prediksi saat ini.")

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
        plt.close(fig)

    except Exception as e:
        st.warning("Visualisasi SHAP tidak dapat ditampilkan, tetapi sistem tetap menampilkan hasil prediksi.")
        st.caption(f"Detail teknis: {e}")

    # =========================
    # ANONYMIZATION
    # =========================
    st.subheader("Data Setelah Anonymization")

    st.dataframe(
        cleaned_df,
        width="stretch"
    )

    # =========================
    # INTERPRETASI
    # =========================
    st.subheader("Interpretasi Singkat")

    st.write(
        f"""
        Berdasarkan input suhu mesin sebesar **{suhu}** dan getaran mesin sebesar **{getaran}**,
        model menghasilkan skor risiko sebesar **{risk_score:.2f}**. Hasil tersebut masuk kategori
        **{kategori}**. Setelah dihitung menggunakan metode SAW, rekomendasi tindakan terbaik adalah
        **{rekomendasi_terbaik}**.
        """
    )

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="footer">
        Project UAS Minggu 16: Integrasi Streamlit, Scaler, Machine Learning, SHAP,
        Data Drift, Anonymization, dan SPK SAW.
    </div>
    """,
    unsafe_allow_html=True
)