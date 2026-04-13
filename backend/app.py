from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import pandas as pd
import os
import threading
import kagglehub
import glob
import numpy as np

# Automatically inject the HuggingFace token for gated models
os.environ["HF_TOKEN"] = "[hf token here]"

app = Flask(__name__)
CORS(app)

# Global variables for models
models = {
    'model_1': { 'clf': None, 'encoder': None, 'features': None, 'ready': False, 'accuracy': 0.0, 'error': None },
    'model_2': { 'clf': None, 'encoder': None, 'features': None, 'ready': False, 'accuracy': 0.0, 'error': None }
}

def train_model_1():
    try:
        print("[Model 1] Fetching German Credit Dataset...")
        data = fetch_openml('credit-g', version=1, as_frame=True)
        df = data.frame
        
        X = df.drop(columns=['class'])
        y = df['class']
        
        models['model_1']['features'] = X.columns.tolist()
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded = X.copy()
        if categorical_cols:
            X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])
        
        y_encoded = y.map({'good': 1, 'bad': 0})
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
        
        print("[Model 1] Fitting TabPFN Classifier...")
        clf = TabPFNClassifier()
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        models['model_1'].update({
            'clf': clf, 'encoder': encoder, 'accuracy': acc, 'ready': True
        })
        print(f"[Model 1] Ready! Accuracy: {acc:.2f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        models['model_1']['error'] = str(e)

def train_model_2():
    try:
        print("[Model 2] Fetching LaoTse Credit Risk Dataset via Kagglehub...")
        path = kagglehub.dataset_download('laotse/credit-risk-dataset')
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError("Could not find CSV file in Kaggle dataset!")
            
        df = pd.read_csv(csv_files[0])
        # Drop rows with NaNs to keep it simple for categorical encoder and TabPFN
        df = df.dropna()
        
        # TabPFN is optimized for smaller datasets. We'll sample 8000 rows to prevent slow CPU inference and hard limits.
        if len(df) > 8000:
            df = df.sample(8000, random_state=42)
            
        # The target column is usually 'loan_status' (0 is non-default/good, 1 is default/bad)
        # TabPFN is fine predicting any labels, but we'll map 0 -> 1 (Approved), 1 -> 0 (Rejected) for consistency with Model 1
        X = df.drop(columns=['loan_status'])
        y = df['loan_status'].map({0: 1, 1: 0}) 
        
        models['model_2']['features'] = X.columns.tolist()
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded = X.copy()
        if categorical_cols:
            X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])
            
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        print("[Model 2] Fitting TabPFN Classifier...")
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        models['model_2'].update({
            'clf': clf, 'encoder': encoder, 'accuracy': acc, 'ready': True
        })
        print(f"[Model 2] Ready! Accuracy: {acc:.2f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        models['model_2']['error'] = str(e)

def initialize_models():
    # To save time, we run them sequentially but it's fast
    train_model_1()
    train_model_2()

INIT_THREAD = threading.Thread(target=initialize_models)
INIT_THREAD.start()

@app.route('/status', methods=['GET'])
def get_status():
    safe_models = {}
    for mod_id, mod_data in models.items():
        safe_models[mod_id] = {
            'ready': mod_data['ready'],
            'accuracy': mod_data['accuracy'],
            'error': mod_data['error'],
            'features': mod_data['features']
        }
    return jsonify(safe_models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json
        model_id = user_input.get('model_id', 'model_1')
        data_fields = user_input.get('data', {})
        
        m = models[model_id]
        if not m['ready']:
            return jsonify({'error': f'{model_id} is still initializing or failed.'}), 503
            
        # Extract fields
        input_data = {}
        for col in m['features']:
            input_data[col] = [data_fields.get(col, None)] 
            
        input_df = pd.DataFrame(input_data)
        
        categorical_cols = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols and m['encoder']:
             input_df[categorical_cols] = m['encoder'].transform(input_df[categorical_cols])
             
        # Make sure datatypes are numeric
        input_df = input_df.astype(float)
        
        prediction = m['clf'].predict(input_df)
        probability = m['clf'].predict_proba(input_df)
        
        # Mapping: 1 is Approved, 0 is Rejected
        result = int(prediction[0]) 
        # probability shape [1, 2] usually ordered by classes [0, 1]
        try:
            class_idx = list(m['clf'].classes_).index(result)
            confidence = float(probability[0][class_idx])
        except Exception:
            confidence = float(np.max(probability[0]))
        
        return jsonify({
            'status': 'success',
            'prediction': 'Approved' if result == 1 else 'Rejected',
            'confidence': confidence
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
