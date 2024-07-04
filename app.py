from flask import Flask, jsonify, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import textwrap

app = Flask(__name__)

# Path to the data file
data_dir = 'public/data/'
model_path = 'model.pkl'
data_split_path = 'data_split.pkl'
train_data_path = os.path.join(data_dir, 'data.xlsx')
test_data_path = os.path.join(data_dir, 'test.xlsx')
dummy_data_path = os.path.join(data_dir, 'dummy_data.xlsx')



def log_process(logs, message):
    logs.append(message)
    print(message)

@app.route('/load_train_data', methods=['GET'])
def load_train_data():
    try:
        data_ikan = pd.read_excel(train_data_path)
        return data_ikan.to_json(orient='records')
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/load_test_data', methods=['GET'])
def load_test_data():
    try:
        data_test = pd.read_excel(test_data_path)
        return data_test.to_json(orient='records')
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route('/load_dummy_data', methods=['GET'])
def load_dummy_data():
    try:
        data_test = pd.read_excel(dummy_data_path)
        return data_test.to_json(orient='records')
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    logs = []
    
    # Hapus model dan data split yang sudah ada
    if os.path.exists(model_path):
        os.remove(model_path)
        log_process(logs, "Existing model deleted.")
        
    if os.path.exists(data_split_path):
        os.remove(data_split_path)
        log_process(logs, "Existing data split deleted.")
    
    log_process(logs, "Loading data...")
    data_ikan = pd.read_excel(train_data_path)
    
    log_process(logs, "Preparing features and labels...")
    features = data_ikan.drop(columns=['ID', 'status_penjualan'])
    labels = data_ikan['status_penjualan']
    features_encoded = pd.get_dummies(features)
    
    log_process(logs, "Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.3, random_state=42)
    
    log_process(logs, "Training the model...")
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42, min_impurity_decrease=0)
    clf.fit(X_train, y_train)
    
    log_process(logs, "Saving the model and split data...")
    joblib.dump((clf, features_encoded.columns, list(clf.classes_)), model_path)
    joblib.dump((X_train, X_test, y_train, y_test, features_encoded.columns), data_split_path)
    
    log_process(logs, "Saving train and test data to files...")
    X_train.to_excel(os.path.join(data_dir, 'X_train.xlsx'), index=False)
    X_test.to_excel(os.path.join(data_dir, 'X_test.xlsx'), index=False)
    y_train.to_frame().to_excel(os.path.join(data_dir, 'y_train.xlsx'), index=False)
    y_test.to_frame().to_excel(os.path.join(data_dir, 'y_test.xlsx'), index=False)
    
    log_process(logs, "Model and data split trained and saved successfully.")
    return jsonify({"status": "success", "logs": logs})


@app.route('/train_data')
def train_page():
    return render_template('train.html')
@app.route('/visualize_tree', methods=['GET'])
def visualize_tree_with_custom_text():
    clf, feature_names, class_names = joblib.load('model.pkl')
    fig, ax = plt.subplots(figsize=(48, 36), dpi=300)
    
    # Visualize tree with smaller font size
    plot_tree(clf, 
              filled=True, 
              feature_names=list(feature_names), 
              class_names=class_names, 
              rounded=True, 
              precision=4, 
              fontsize=10, 
              ax=ax)

    # Extract and format rules
    def extract_rules(clf, feature_names, class_names):
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        value = clf.tree_.value

        def recurse(node, depth=0):
            indent = "  " * depth
            if children_left[node] != children_right[node]:
                name = feature_names[feature[node]]
                threshold_value = threshold[node]
                if "jenis_ikan" in name:
                    condition = f"{name} IS TRUE" if threshold_value <= 0.5 else f"{name} IS FALSE"
                else:
                    condition = f"{name} <= {threshold_value:.4f}"
                left_rule = recurse(children_left[node], depth + 1)
                right_rule = recurse(children_right[node], depth + 1)
                if depth == 0:
                    return (
                        f"{indent}JIKA {condition} MAKA:\n{left_rule}\n"
                        f"{indent}JIKA TIDAK MAKA:\n{right_rule}"
                    )
                else:
                    return (
                        f"{indent}DAN JIKA {condition} MAKA:\n{left_rule}\n"
                        f"{indent}DAN JIKA TIDAK MAKA:\n{right_rule}"
                    )
            else:
                value_counts = value[node]
                class_value = np.argmax(value_counts)
                class_name = class_names[class_value]
                return f"{indent}KELAS = {class_name}"

        return recurse(0)

    tree_rules = extract_rules(clf, feature_names, class_names)

    # Add custom text to the plot
    def add_custom_text(ax, tree_rules):
        for i, text in enumerate(ax.texts):
            lines = text.get_text().split("\n")
            new_text = []
            for line in lines:
                if "entropy" in line or "samples" in line or "value" in line:
                    continue
                if "class" in line:
                    new_text.append(line.replace("class", "KELAS"))
                else:
                    new_text.append(line)
            wrapped_text = "\n".join(textwrap.wrap(" ".join(new_text), width=15))
            text.set_text(wrapped_text)
            text.set_fontsize(10)

    add_custom_text(ax, tree_rules)

    plt.subplots_adjust(hspace=10, wspace=10)
    plt.savefig('static/images/decision_tree_visualization.png', bbox_inches='tight')
    plt.close()
    return jsonify({"status": "Decision tree visualization saved"})


@app.route('/extract_rules', methods=['GET'])
def extract_rules_text():
    clf, feature_names, class_names = joblib.load('model.pkl')
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    def get_rules_to_json(node=0):
        if children_left[node] != children_right[node]:
            name = feature_names[feature[node]]
            threshold_value = threshold[node]
            left_rule = get_rules_to_json(children_left[node])
            right_rule = get_rules_to_json(children_right[node])
            return {
                "feature": name,
                "threshold": threshold_value,
                "left": left_rule,
                "right": right_rule
            }
        else:
            value_counts = value[node]
            class_value = np.argmax(value_counts)
            class_name = class_names[class_value]
            return {"class": class_name}

    def get_rules_to_text(node=0, depth=0, is_first=True):
        indent = "  " * depth
        if children_left[node] != children_right[node]:
            name = feature_names[feature[node]]
            threshold_value = threshold[node]
            if "jenis_ikan" in name:
                condition = f"{name} IS TRUE" if threshold_value <= 0.5 else f"{name} IS FALSE"
            else:
                condition = f"{name} <= {threshold_value}"
            left_rule = get_rules_to_text(children_left[node], depth + 1, False)
            right_rule = get_rules_to_text(children_right[node], depth + 1, False)
            if is_first:
                return (
                    f"{indent}JIKA {condition} MAKA:\n{left_rule}\n"
                    f"{indent}JIKA TIDAK MAKA:\n{right_rule}"
                )
            else:
                return (
                    f"{indent}DAN JIKA {condition} MAKA:\n{left_rule}\n"
                    f"{indent}DAN JIKA TIDAK MAKA:\n{right_rule}"
                )
        else:
            value_counts = value[node]
            class_value = np.argmax(value_counts)
            class_name = class_names[class_value]
            return f"{indent}KELAS = {class_name}"

    rules_json = get_rules_to_json()
    rules_text = get_rules_to_text()
    
    with open('decision_tree_rules.json', 'w') as f:
        json.dump(rules_json, f, indent=4)
    
    with open('decision_tree_rules.txt', 'w') as f:
        f.write(rules_text)

    return jsonify({"status": "Rules extracted and saved in both JSON and TXT format"})


@app.route('/classify_test_data', methods=['GET'])
def classify_test_data():
    rules_json = None
    with open('decision_tree_rules.json', 'r') as f:
        rules_json = json.load(f)

    data_test = pd.read_excel(test_data_path)
    features_test = data_test.drop(columns=['ID'])
    features_encoded_test = pd.get_dummies(features_test)

    X_train, X_test, y_train, y_test, feature_names = joblib.load('data_split.pkl')
    missing_cols = set(feature_names) - set(features_encoded_test.columns)
    for c in missing_cols:
        features_encoded_test[c] = 0
    features_encoded_test = features_encoded_test[feature_names]

    def classify_with_rules(rules, sample):
        if "class" in rules:
            return rules["class"]
        feature = rules["feature"]
        threshold = rules["threshold"]
        if sample[feature] <= threshold:
            return classify_with_rules(rules["left"], sample)
        else:
            return classify_with_rules(rules["right"], sample)

    predictions = features_encoded_test.apply(lambda x: classify_with_rules(rules_json, x), axis=1)
    data_test['Predicted Status Penjualan'] = predictions
    data_test.to_excel(os.path.join(data_dir, 'predicted_data_test.xlsx'), index=False)
    return data_test[['ID', 'Predicted Status Penjualan']].to_json(orient='records')

@app.route('/view_predicted_data', methods=['GET'])
def view_predicted_data():
    try:
        # Load classified data from the Excel file
        classified_data_path = os.path.join(data_dir, 'predicted_data_test.xlsx')
        classified_data = pd.read_excel(classified_data_path)
        
        # Convert the data to JSON format
        classified_json = classified_data.to_json(orient='records')
        
        return classified_json
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/visualize_data', methods=['GET'])
def visualize_data():
    data_ikan = pd.read_excel(train_data_path)

    # Plot for 'Lokasi Penjualan'
    plt.figure(figsize=(14, 8))
    ax1 = sns.countplot(data=data_ikan, x='Lokasi Penjualan', hue='Jenis Ikan', palette='viridis')
    plt.title('Distribution of Fish Types by Location')
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    legend1 = ax1.legend(title='Fish Type', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig('distribution_by_location_with_legend.png', bbox_extra_artists=(legend1,), bbox_inches='tight')
    plt.close()

    # Plot for 'Kategori Pemasaran'
    plt.figure(figsize=(14, 8))
    ax2 = sns.countplot(data=data_ikan, x='Kategori Pemasaran', hue='Jenis Ikan', palette='viridis')
    plt.title('Distribution of Fish Types by Marketing Category')
    plt.xlabel('Marketing Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    legend2 = ax2.legend(title='Fish Type', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig('distribution_by_marketing_category_with_legend.png', bbox_extra_artists=(legend2,), bbox_inches='tight')
    plt.close()

    predicted_data_test = pd.read_excel(os.path.join(data_dir, 'predicted_data_test.xlsx'))
    status_counts = predicted_data_test['Predicted Status Penjualan'].value_counts(normalize=True) * 100

    plt.figure(figsize=(8, 6))
    sns.barplot(x=status_counts.index, y=status_counts.values, palette='viridis')
    plt.title('Percentage of Predicted Sales Status')
    plt.xlabel('Predicted Status Penjualan')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig('predicted_sales_status_percentage.png')
    plt.close()

    return jsonify({"status": "Visualizations saved"})


@app.route('/generate_dummy_data', methods=['GET'])
def generate_dummy_data():
    # Ambil jumlah data dari parameter query, default 100 jika tidak diberikan
    num_data = request.args.get('num', default=500, type=int)
    
    # Data dummy
    data_dummy = {
        'ID': np.arange(1, num_data + 1),
        'jenis_ikan': np.random.choice(['Udang', 'Bawal', 'Tongkol', 'Kerapu', 'Tuna'], num_data),
        'tahun': np.random.choice([2020, 2021, 2022], num_data),
        'bulan': np.random.choice(['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'], num_data),
        'berat': np.round(np.random.uniform(0.5, 5.0, num_data), 1),
        'lokasi': np.random.choice(['Aceh Utara', 'Lhokseumawe'], num_data),
        'stok': np.random.randint(50, 200, num_data),
        'terjual_harian': np.random.randint(20, 150, num_data),
        'kategori_pemasaran': np.random.choice(['Jual langsung ke masyarakat', 'Suplai ke FNB atau pelaku bisnis lain', 'Penjualan langsung ke pedagang UMKM'], num_data),
        'status_penjualan': np.random.choice(['Laris', 'Tidak Laris'], num_data)
    }
    
    # Membuat DataFrame
    df_dummy = pd.DataFrame(data_dummy)
    
    # Menyimpan ke file Excel
    output_path = 'public/data/dummy_data.xlsx'
    df_dummy.to_excel(output_path, index=False)
    
    return jsonify({"status": "Dummy data generated and saved", "file_path": output_path})

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data_latih')
def data_latih():
    return render_template('data_latih.html')

@app.route('/data_uji')
def data_uji():
    return render_template('data_uji.html')

@app.route('/hasil')
def hasil():
    return render_template('hasil.html')


@app.route('/show_rule')
def rule():
    try:
        with open('decision_tree_rules.txt', 'r') as file:
            rules = file.read()
    except FileNotFoundError:
        rules = "File not found."

    return render_template('rule.html', rules=rules)

@app.route('/show_tree')
def show_tree():
    return render_template('tree.html')

@app.route('/image/<filename>')
def image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)
