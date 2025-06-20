import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import joblib
import mysql.connector
from datetime import datetime
import pandas as pd

# Load model & encoder
model = joblib.load("model_knn.pkl")
le_jenis = joblib.load("le_jenis.pkl")
le_kondisi = joblib.load("le_kondisi.pkl")

# Ekstraksi fitur
def extract_glcm_features(image, distances=[1], angles=[0], levels=256):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]

# Simpan ke MySQL
def insert_to_db(filename, jenis, kondisi):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="prediksi_daun"
        )
        cursor = conn.cursor()
        sql = "INSERT INTO data_prediksi (filename, jenis_daun, kondisi) VALUES (%s, %s, %s)"
        cursor.execute(sql, (filename, jenis, kondisi))
        conn.commit()
    except mysql.connector.Error as e:
        st.error(f"Gagal menyimpan ke database: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# ====== Streamlit UI ======
st.set_page_config(page_title="Prediksi Daun", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>Prediksi Jenis & Kondisi Daun Tanaman Hortikultura🌿</h1>
    <p style='text-align: center; font-size: 16px; margin-bottom: 5px;'>
        Sistem akan memprediksi jenis serta kesehatan daun dengan menggunakan fitur tekstur <br> <b>Gray Level Co-Occurrence Matrix (GLCM)</b> dan <b> Model K-Nearest Neighbors (KNN).</b>
    </p>
    <p style='text-align: center; font-size: 14px; color: #555; margin-top: 0;'>
        <i>Catatan: Sistem hanya dapat memprediksi tiga jenis daun yaitu <b>Kentang, Jagung, dan Tomat</b>.</i>
    </p>
    <hr style='margin: 10px 0 25px 0;'>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Silakan upload gambar daun disini", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("Ukuran file terlalu besar! Maksimal 5 MB.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file).convert("L")
            st.image(image, caption="Gambar Daun (Grayscale)", use_container_width=True)

        with col2:
            img_array = np.array(image)
            img_resized = cv2.resize(img_array, (128, 128))
            fitur = extract_glcm_features(img_resized)
            pred = model.predict([fitur])[0]

            jenis = le_jenis.inverse_transform([pred[0]])[0]
            kondisi = le_kondisi.inverse_transform([pred[1]])[0]

            allowed_jenis = ['jagung', 'kentang', 'tomat']

            if jenis.lower() not in allowed_jenis:
                st.error(f"Jenis daun yang diprediksi bukan jagung, kentang, atau tomat: {jenis}. Silakan upload gambar daun yang sesuai.")
            else:
                # Tentukan warna berdasarkan kondisi
                color_kondisi = "#228B22" if kondisi.lower() == "sehat" else "#FF0000"

                st.markdown(f"""
                <div style='background-color: #EAFBF3; padding: 20px; border-radius: 10px;'>
                    <h3 style='color: #1E90FF;'>Jenis Daun: <b>{jenis.upper()}</b></h3>
                    <h3 style='color: {color_kondisi};'>Kondisi Daun: <b>{kondisi.upper()}</b></h3>
                </div>
                """, unsafe_allow_html=True)

            # # Simpan file dan ke database hanya sekali
            # if "saved_to_db" not in st.session_state:
            #     st.session_state.saved_to_db = False

            if not st.session_state.saved_to_db:
                os.makedirs("uploaded", exist_ok=True)
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                save_path = os.path.join("uploaded", filename)
                image.save(save_path)
                insert_to_db(filename, jenis, kondisi)
                st.session_state.saved_to_db = True

# Tampilkan riwayat prediksi
if st.checkbox("Tampilkan Riwayat Prediksi"):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="prediksi_daun"
        )
        df = pd.read_sql("SELECT * FROM data_prediksi ORDER BY id DESC", conn)
        st.dataframe(df)
    except:
        st.warning("Gagal memuat data dari database.")
