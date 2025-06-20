import os
import cv2
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

def extract_glcm_features(image, distances=[1], angles=[0], levels=256):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]

# Direktori dataset
base_dir = "dataset"
data = []

for jenis in os.listdir(base_dir):
    jenis_path = os.path.join(base_dir, jenis)
    if os.path.isdir(jenis_path):
        for kondisi in os.listdir(jenis_path):
            kondisi_path = os.path.join(jenis_path, kondisi)
            if os.path.isdir(kondisi_path):
                for file in tqdm(os.listdir(kondisi_path), desc=f"{jenis}/{kondisi}"):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(kondisi_path, file)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            continue
                        image = cv2.resize(image, (128, 128))
                        features = extract_glcm_features(image)
                        data.append(features + [jenis, kondisi])

# Simpan ke CSV
df = pd.DataFrame(data, columns=['contrast', 'correlation', 'energy', 'homogeneity', 'jenis_daun', 'kondisi'])
df.to_csv("dataset_daun.csv", index=False)
print("Fitur daun berhasil disimpan")
