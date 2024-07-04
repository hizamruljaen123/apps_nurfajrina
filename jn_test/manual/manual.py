import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Memuat data dari file Excel
file_path = '../data.xlsx'
data_ikan = pd.read_excel(file_path)

# Menampilkan beberapa baris pertama untuk memahami struktur data
print(data_ikan.head())

# Mengonversi variabel kategorikal ke numerik
data_ikan_encoded = pd.get_dummies(data_ikan.drop(columns=['ID']))
print(data_ikan_encoded.columns)

# Menghitung entropi
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Menghitung gain
def gain(data, split_attribute_name, target_name="status_penjualan_Tidak Laris"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Memilih atribut terbaik berdasarkan gain ratio dengan subset data secara acak
def best_split(data, attributes, target_name="status_penjualan_Tidak Laris"):
    sample_data = data.sample(frac=0.1)  # Menggunakan 10% dari data untuk perhitungan gain
    IGs = [gain(sample_data, attr, target_name) for attr in attributes]
    return attributes[np.argmax(IGs)]

# Membuat pohon keputusan
def create_decision_tree(data, attributes, target_name="status_penjualan_Tidak Laris", parent_node_class=None):
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]
    elif len(data) == 0:
        return np.unique(parent_node_class)[np.argmax(np.unique(parent_node_class, return_counts=True)[1])]
    elif len(attributes) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
        best_attr = best_split(data, attributes, target_name)
        tree = {str(best_attr): {}}
        attributes = [i for i in attributes if i != best_attr]
        for value in np.unique(data[best_attr]):
            sub_data = data.where(data[best_attr] == value).dropna()
            subtree = create_decision_tree(sub_data, attributes, target_name, parent_node_class)
            tree[str(best_attr)][str(value)] = subtree
        return tree

# Atribut-atribut yang akan digunakan
attributes = list(data_ikan_encoded.columns)
attributes.remove('status_penjualan_Tidak Laris')

# Membuat pohon keputusan
decision_tree = create_decision_tree(data_ikan_encoded, attributes)
print("Pohon Keputusan:")
print(json.dumps(decision_tree, indent=4))

# Fungsi untuk mengklasifikasi sampel baru menggunakan pohon keputusan
def classify(sample, tree):
    for key in tree.keys():
        value = sample[key]
        print(f"Classifying key: {key}, value: {value}")
        if str(value) in tree[key]:
            tree = tree[key][str(value)]
        else:
            return "Unknown"
        if isinstance(tree, dict):
            return classify(sample, tree)
        else:
            return tree

# Membaca data uji dan mengonversinya
file_path_test = '../test.xlsx'
data_test = pd.read_excel(file_path_test)
data_test_encoded = pd.get_dummies(data_test.drop(columns=['ID']))

# Memastikan data uji memiliki kolom yang sama dengan data latih
missing_cols = set(data_ikan_encoded.columns) - set(data_test_encoded.columns)
for c in missing_cols:
    data_test_encoded[c] = 0
data_test_encoded = data_test_encoded[data_ikan_encoded.columns]

# Mengklasifikasi data uji
predictions = data_test_encoded.apply(lambda x: classify(x, decision_tree), axis=1)
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
