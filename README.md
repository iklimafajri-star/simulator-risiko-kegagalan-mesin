# Simulator Risiko Kegagalan Mesin

Project ini merupakan implementasi Praktikum Minggu 15 tentang finalisasi proyek dan operasionalisasi model berbasis MLOps.

## Fitur Aplikasi
- Memuat model machine learning dari file `.joblib`
- Memuat scaler dari file `.joblib`
- Melakukan prediksi risiko kegagalan mesin
- Menampilkan kategori risiko
- Menampilkan rekomendasi keputusan sederhana
- Mendeteksi data drift
- Menerapkan anonymization sederhana

## Struktur File
- `app.py`: file utama aplikasi Streamlit
- `style.css`: file tampilan CSS
- `model_risiko_v1.joblib`: model machine learning
- `scaler_risiko_v1.joblib`: scaler preprocessing
- `requirements.txt`: daftar library untuk deployment
- `T152.ipynb`: notebook praktikum

## Etika Data
Aplikasi ini tidak menggunakan data sensitif seperti nama, NIK, atau alamat sebagai dasar prediksi. Data sensitif dibuang sebelum data diproses oleh sistem.