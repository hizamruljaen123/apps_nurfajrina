# Analisis Efektivitas Marketing Penjualan Ikan dengan C4.5

## Overview
Proyek ini menggunakan **algoritma C4.5** untuk menganalisis **efektivitas strategi marketing pada penjualan ikan**. Tujuannya adalah:
- Menentukan faktor-faktor yang mempengaruhi keberhasilan penjualan ikan
- Mengklasifikasikan strategi marketing menjadi **Efektif** atau **Tidak Efektif**
- Memberikan insight cepat menggunakan **very fast C4.5** (optimasi implementasi agar lebih efisien)

## Features
- Input: Data penjualan ikan, termasuk:
  - Jenis ikan
  - Harga
  - Media marketing (online/offline)
  - Diskon/promo
  - Waktu penjualan
  - Lokasi pasar
- Output: Klasifikasi efektivitas strategi marketing
- Evaluasi: Accuracy, Precision, Recall, F1-score

## Workflow
1. **Data Collection**: Kumpulkan data penjualan ikan harian/mingguan  
2. **Data Preprocessing**: Membersihkan data, normalisasi, handling missing value  
3. **C4.5 Model**: Membuat pohon keputusan berdasarkan fitur-fitur marketing  
4. **Prediction & Evaluation**: Klasifikasi strategi marketing dan evaluasi akurasi  
5. **Insight & Recommendation**: Menyediakan rekomendasi untuk strategi marketing yang lebih efektif

## Python Example (C4.5 Very Fast)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv('penjualan_ikan.csv')
X = data.drop(columns=['efektivitas'])
y = data['efektivitas']  # target: 'Efektif' / 'Tidak Efektif'

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# C4.5 (Decision Tree with info gain ratio)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)  # depth dibatasi untuk kecepatan
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

# Optional: Visualisasi pohon keputusan
from sklearn.tree import export_text
print(export_text(clf, feature_names=list(X.columns)))
