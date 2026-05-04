import kagglehub
import glob
import os
import pandas as pd
import numpy as np
import threading
import torch
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

app = Flask(__name__)
CORS(app)

engines = {
    "model_1": {"tabpfn": None, "catboost": None, "features": [], "ready": False},
    "model_2": {"tabpfn": None, "catboost": None, "features": [], "ready": False}
}


def save_model_assets(m_id):
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    engine = engines[m_id]
    engine["catboost"].save_model(f"saved_models/{m_id}_catboost.cbm")
    joblib.dump(engine["tabpfn"], f"saved_models/{m_id}_tabpfn.joblib")
    joblib.dump(engine["features"], f"saved_models/{m_id}_features.joblib")
    print(f">>> {m_id} assets saved to disk.")

def load_model_assets(m_id):
    cb_path = f"saved_models/{m_id}_catboost.cbm"
    tp_path = f"saved_models/{m_id}_tabpfn.joblib"
    ft_path = f"saved_models/{m_id}_features.joblib"

    if os.path.exists(cb_path) and os.path.exists(tp_path) and os.path.exists(ft_path):
        try:
            engines[m_id]["catboost"] = CatBoostClassifier().load_model(cb_path)
            engines[m_id]["tabpfn"] = joblib.load(tp_path)
            engines[m_id]["features"] = joblib.load(ft_path)
            engines[m_id]["ready"] = True
            print(f">>> {m_id} loaded from disk. Ready!")
            return True
        except Exception as e:
            print(f">>> Failed to load {m_id}, will re-train: {e}")
    return False


def train_german_model():
    if load_model_assets("model_1"): return
    
    try:
        print("Training Model 1 (German Credit) on CPU...")
        df = pd.read_csv("german_credit.csv")
        X = df.drop(columns=['class'])
        y = df['class'].map({'good': 1, 'bad': 0})
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        engines["model_1"]["features"] = X.columns.tolist()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_tab = X_train.copy()
        for col in cat_cols:
            X_train_tab[col] = X_train_tab[col].astype('category').cat.codes
        
        
        t_model = TabPFNClassifier(device='cpu')
        t_model.fit(X_train_tab, y_train)
        
        c_model = CatBoostClassifier(iterations=500, verbose=False)
        c_model.fit(X_train, y_train, cat_features=cat_cols)
        
        engines["model_1"]["tabpfn"] = t_model
        engines["model_1"]["catboost"] = c_model
        engines["model_1"]["ready"] = True
        save_model_assets("model_1")
        print("Model 1 training complete!")
    except Exception as e:
        print(f"Model 1 Error: {e}")

def train_kaggle_model():
    if load_model_assets("model_2"): return
    
    try:
        print(">>> Training Model 2 (Kaggle)")
        df = pd.read_csv("L_credit_risk_dataset.csv") 
        df = df.dropna()
        
        y = df['loan_status'].apply(lambda x: 0 if x == 1 else 1)
        
        X = df.drop(columns=['loan_status', 'loan_grade'])
        
        df = df.sample(n=min(8000, len(df)))
        y = y.loc[X.index] 

        cat_cols = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
        engines["model_2"]["features"] = X.columns.tolist()

        for col in cat_cols: X[col] = X[col].astype(str)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        
        X_train_tab = X_train.copy()
        for col in cat_cols:
            X_train_tab[col] = X_train_tab[col].astype('category').cat.codes

        t_model = TabPFNClassifier(device='cuda', ignore_pretraining_limits=True)
        t_model.fit(X_train_tab, y_train)

        c_model = CatBoostClassifier(iterations=500, task_type="GPU", devices='0', verbose=False)
        c_model.fit(X_train, y_train, cat_features=cat_cols)

        engines["model_2"]["tabpfn"] = t_model
        engines["model_2"]["catboost"] = c_model
        engines["model_2"]["ready"] = True
        save_model_assets("model_2")
        print(" Model 2 training complete!")
    except Exception as e:
        print(f"Model 2 Error: {e}")


threading.Thread(target=train_german_model).start()
threading.Thread(target=train_kaggle_model).start()

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    m_id = req.get('model_id', 'model_1')
    user_input = req.get('data')
    if not engines[m_id]["ready"]: return jsonify({"error": "Model not ready"}), 503

    try:
        engine = engines[m_id]
        df_input = pd.DataFrame([user_input]).reindex(columns=engine["features"])

        if m_id == "model_2":
            cat_cols = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
            for col in cat_cols: df_input[col] = df_input[col].astype(str)
        else:
            for col in df_input.columns:
                if df_input[col].dtype == 'object' or df_input[col].dtype == 'category':
                    df_input[col] = df_input[col].astype(str)

        df_tab = df_input.copy()
        for col in df_tab.columns:
            if df_tab[col].dtype == 'object' or df_tab[col].dtype == 'category':
                df_tab[col] = df_tab[col].astype('category').cat.codes
        
        df_tab_final = df_tab.astype('float32') 
        prob_tab = float(engine["tabpfn"].predict_proba(df_tab_final)[0][1])
        prob_cat = float(engine["catboost"].predict_proba(df_input)[0][1])
        avg_score = (prob_tab + prob_cat) / 2
        
        return jsonify({
            "status": "success", "score": round(avg_score, 2),
            "tabpfn_result": "Approved" if prob_tab > 0.5 else "Rejected",
            "catboost_result": "Approved" if prob_cat > 0.5 else "Rejected",
            "final_decision": "Approved" if avg_score > 0.5 else "Rejected",
            "conflict": abs(prob_tab - prob_cat) > 0.3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000)