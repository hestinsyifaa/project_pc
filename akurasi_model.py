import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('dataset_daun.csv')

# Mapping label kondisi
label_map = {'sehat': 0, 'tidak sehat': 1}
df['label'] = df['kondisi'].map(label_map)

# Fitur dan label
X = df[['contrast', 'correlation', 'energy', 'homogeneity']].values
y = df['label'].values

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAINING MODEL
# SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# EVALUASI MODEL 
models = {
    "SVM": (y_test, y_pred_svm),
    "KNN": (y_test, y_pred_knn)
}

# CONFUSION MATRIX 
for name, (true, pred) in models.items():
    acc = accuracy_score(true, pred)
    print(f"\n=== {name} ===")
    print("Akurasi:", acc)
    print("Classification Report:")
    print(classification_report(true, pred, target_names=["Sehat", "Tidak Sehat"]))

    cm = confusion_matrix(true, pred)
    print("Confusion Matrix:")
    print(cm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, (true, pred)) in zip(axes, models.items()):
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sehat", "Tidak Sehat"],
                yticklabels=["Sehat", "Tidak Sehat"],
                ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# BAR CHART AKURASI DENGAN PERSENTASE
accuracies = {
    "SVM": accuracy_score(y_test, y_pred_svm),
    "KNN": accuracy_score(y_test, y_pred_knn)
}

plt.figure(figsize=(6, 4))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['lightgreen', 'salmon'])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02,
             f"{height * 100:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylabel('Akurasi')
plt.title('Perbandingan Akurasi Model')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
