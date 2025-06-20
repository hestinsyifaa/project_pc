import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data fitur + label
df = pd.read_csv("dataset_daun.csv")  

# Fitur dan label
X = df[['contrast', 'correlation', 'energy', 'homogeneity']]
y = df[['jenis_daun', 'kondisi']].copy()

# Encode label
le_jenis = LabelEncoder()
le_kondisi = LabelEncoder()
y['jenis_daun'] = le_jenis.fit_transform(y['jenis_daun'])
y['kondisi'] = le_kondisi.fit_transform(y['kondisi'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
multi_knn = MultiOutputClassifier(knn)
multi_knn.fit(X_train, y_train)

# Simpan model dan label encoder
joblib.dump(multi_knn, "model_knn.pkl")
joblib.dump(le_jenis, "le_jenis.pkl")
joblib.dump(le_kondisi, "le_kondisi.pkl")
