import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Memuat data dari file Excel
file_path = 'data.xlsx'
data_ikan = pd.read_excel(file_path)

# Menampilkan beberapa baris pertama untuk memahami struktur data
print(data_ikan.head())

# Mengonversi variabel kategorikal ke numerik
data_ikan_encoded = pd.get_dummies(data_ikan.drop(columns=['ID']))
print(data_ikan_encoded.columns)

# Membagi data menjadi fitur dan label
X = data_ikan_encoded.drop(columns=['status_penjualan_Tidak Laris'])
y = data_ikan_encoded['status_penjualan_Tidak Laris']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model Decision Tree dengan parameter yang dioptimalkan
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, random_state=42)
clf.fit(X_train, y_train)

# Menyimpan model
joblib.dump(clf, 'decision_tree_model.pkl')

# Memuat model
clf = joblib.load('decision_tree_model.pkl')

# Membuat prediksi pada data uji
y_pred = clf.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy}")

# Visualisasi pohon keputusan
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['Tidak Laris', 'Laris'], rounded=True)
plt.savefig('decision_tree_visualization.png')
plt.show()

# Membaca data uji dan mengonversinya
file_path_test = 'test.xlsx'
data_test = pd.read_excel(file_path_test)
data_test_encoded = pd.get_dummies(data_test.drop(columns=['ID']))

# Memastikan data uji memiliki kolom yang sama dengan data latih
missing_cols = set(X.columns) - set(data_test_encoded.columns)
for c in missing_cols:
    data_test_encoded[c] = 0
data_test_encoded = data_test_encoded[X.columns]

# Membuat prediksi pada data uji
predictions = clf.predict(data_test_encoded)
data_test['Predicted Status Penjualan'] = predictions
data_test.to_excel('predicted_data_test.xlsx', index=False)

# Visualisasi hasil
plt.figure(figsize=(14, 8))
ax1 = sns.countplot(data=data_ikan, x='lokasi', hue='jenis_ikan', palette='viridis')
plt.title('Distribusi Jenis Ikan Berdasarkan Lokasi Penjualan')
plt.xlabel('Lokasi Penjualan')
plt.ylabel('Jumlah')
plt.xticks(rotation=45)
legend1 = ax1.legend(title='Jenis Ikan', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig('distribution_by_location_with_legend.png', bbox_extra_artists=(legend1,), bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 8))
ax2 = sns.countplot(data=data_ikan, x='kategori_pemasaran', hue='jenis_ikan', palette='viridis')
plt.title('Distribusi Jenis Ikan Berdasarkan Kategori Pemasaran')
plt.xlabel('Kategori Pemasaran')
plt.ylabel('Jumlah')
plt.xticks(rotation=45)
legend2 = ax2.legend(title='Jenis Ikan', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig('distribution_by_marketing_category_with_legend.png', bbox_extra_artists=(legend2,), bbox_inches='tight')
plt.show()

predicted_data_test = pd.read_excel('predicted_data_test.xlsx')
status_counts = predicted_data_test['Predicted Status Penjualan'].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 6))
sns.barplot(x=status_counts.index, y=status_counts.values, palette='viridis')
plt.title('Persentase Status Penjualan yang Diprediksi')
plt.xlabel('Status Penjualan yang Diprediksi')
plt.ylabel('Persentase')
plt.tight_layout()
plt.savefig('predicted_sales_status_percentage.png')
plt.show()

print(predicted_data_test.head())

status_counts_df = status_counts.reset_index()
status_counts_df.columns = ['Status Penjualan yang Diprediksi', 'Persentase']
print(status_counts_df)
