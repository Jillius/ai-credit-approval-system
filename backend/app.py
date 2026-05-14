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
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv


load_dotenv('../.env')

app = Flask(__name__)
CORS(app)


engines = {
    "model_1": {"tabpfn": None, "catboost": None, "features": [], "scaler": None, "limits": {}, "ready": False},
    "model_2": {"tabpfn": None, "catboost": None, "features": [], "scaler": None, "limits": {}, "ready": False}
}

def save_model_assets(m_id):
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    engine = engines[m_id]
    engine["catboost"].save_model(f"saved_models/{m_id}_catboost.cbm")
    joblib.dump(engine["tabpfn"], f"saved_models/{m_id}_tabpfn.joblib")
    joblib.dump(engine["features"], f"saved_models/{m_id}_features.joblib")
    joblib.dump(engine["scaler"], f"saved_models/{m_id}_scaler.joblib")
    joblib.dump(engine["limits"], f"saved_models/{m_id}_limits.joblib")
    print(f">>> {m_id} assets (including scaler and limits) saved to disk.")

def load_model_assets(m_id):
    paths = {
        "cb": f"saved_models/{m_id}_catboost.cbm",
        "tp": f"saved_models/{m_id}_tabpfn.joblib",
        "ft": f"saved_models/{m_id}_features.joblib",
        "sc": f"saved_models/{m_id}_scaler.joblib",
        "lm": f"saved_models/{m_id}_limits.joblib"
    }

    if all(os.path.exists(p) for p in paths.values()):
        try:
            engines[m_id]["catboost"] = CatBoostClassifier().load_model(paths["cb"])
            engines[m_id]["tabpfn"] = joblib.load(paths["tp"])
            engines[m_id]["features"] = joblib.load(paths["ft"])
            engines[m_id]["scaler"] = joblib.load(paths["sc"])
            engines[m_id]["limits"] = joblib.load(paths["lm"])
            engines[m_id]["ready"] = True
            print(f">>> {m_id} loaded from disk. Ready with limits!")
            return True
        except Exception as e:
            print(f">>> Failed to load {m_id}, will re-train: {e}")
    return False

def train_german_model():
    if load_model_assets("model_1"): return
    
    try:
        print("Training Model 1 (German Credit)...")
        df = pd.read_csv("german_credit.csv")
        X = df.drop(columns=['class'])
        y = df['class'].map({'good': 1, 'bad': 0})
        
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Eğitim setindeki max değerleri kaydet (Troll engeli için)
        engines["model_1"]["limits"] = X[num_cols].max().to_dict()
        engines["model_1"]["features"] = X.columns.tolist()
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # TabPFN Normalizasyon
        scaler = StandardScaler()
        X_train_tab = X_train.copy()
        X_train_tab[num_cols] = scaler.fit_transform(X_train_tab[num_cols])
        for col in cat_cols:
            X_train_tab[col] = X_train_tab[col].astype('category').cat.codes
        
        t_model = TabPFNClassifier(device='cpu')
        t_model.fit(X_train_tab, y_train)
        
        c_model = CatBoostClassifier(iterations=500, verbose=False)
        c_model.fit(X_train, y_train, cat_features=cat_cols)
        
        engines["model_1"].update({"tabpfn": t_model, "catboost": c_model, "scaler": scaler, "ready": True})
        save_model_assets("model_1")
        print("Model 1 training complete!")
    except Exception as e:
        print(f"Model 1 Error: {e}")

def train_kaggle_model():
    if load_model_assets("model_2"): return
    
    try:
        print(">>> Training Model 2 (Kaggle)")
        df = pd.read_csv("L_credit_risk_dataset.csv").dropna()
        y = df['loan_status'].apply(lambda x: 0 if x == 1 else 1)
        X = df.drop(columns=['loan_status', 'loan_grade'])
        
        df = df.sample(n=min(8000, len(df)))
        y = y.loc[X.index] 

        cat_cols = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
        num_cols = [c for c in X.columns if c not in cat_cols]
        
        # Limitleri kaydet
        engines["model_2"]["limits"] = X[num_cols].max().to_dict()
        engines["model_2"]["features"] = X.columns.tolist()

        for col in cat_cols: X[col] = X[col].astype(str)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        
        scaler = StandardScaler()
        X_train_tab = X_train.copy()
        X_train_tab[num_cols] = scaler.fit_transform(X_train_tab[num_cols])
        for col in cat_cols:
            X_train_tab[col] = X_train_tab[col].astype('category').cat.codes

        t_model = TabPFNClassifier(device='cuda', ignore_pretraining_limits=True)
        t_model.fit(X_train_tab, y_train)

        c_model = CatBoostClassifier(iterations=500, task_type="GPU", devices='0', verbose=False)
        c_model.fit(X_train, y_train, cat_features=cat_cols)

        engines["model_2"].update({"tabpfn": t_model, "catboost": c_model, "scaler": scaler, "ready": True})
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

    if not engines[m_id]["ready"]:
        return jsonify({"error": "Model not ready."}), 503

    try:
        engine = engines[m_id]
        df_input = pd.DataFrame([user_input]).reindex(columns=engine["features"])

        
        for col, max_val in engine["limits"].items():
            if col in df_input.columns:
                df_input[col] = df_input[col].clip(upper=max_val)

       
        df_cat = df_input.copy()
        for col in df_cat.columns:
            if df_cat[col].dtype == 'object' or df_cat[col].dtype == 'category':
                df_cat[col] = df_cat[col].astype(str)

        
        df_tab = df_input.copy()
        num_cols = df_tab.select_dtypes(include=[np.number]).columns.tolist()
        if engine["scaler"] is not None and num_cols:
            df_tab[num_cols] = engine["scaler"].transform(df_tab[num_cols])
        
        for col in df_tab.columns:
            if df_tab[col].dtype == 'object' or df_tab[col].dtype == 'category':
                df_tab[col] = pd.factorize(df_tab[col])[0]
        
        df_tab_final = df_tab.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

       
        prob_tab = float(engine["tabpfn"].predict_proba(df_tab_final)[0][1])
        prob_cat = float(engine["catboost"].predict_proba(df_cat)[0][1])
        
       
        limit_exceeded = False
        for col, max_val in engine["limits"].items():
            if col in user_input and isinstance(user_input[col], (int, float)):
                if user_input[col] > max_val * 1.2:
                    limit_exceeded = True
                    break

        
        if limit_exceeded:
            prob_tab = min(prob_tab, 0.44)
            prob_cat = min(prob_cat, 0.44)
            avg_score = min((prob_tab + prob_cat) / 2, 0.44)
            final_decision = "Rejected"
        else:
            avg_score = (prob_tab + prob_cat) / 2
            final_decision = "Approved" if avg_score > 0.5 else "Rejected"

        return jsonify({
            "status": "success",
            "score": round(avg_score, 4),
            "tabpfn_result": "Approved" if prob_tab > 0.5 else "Rejected",
            "catboost_result": "Approved" if prob_cat > 0.5 else "Rejected",
            "final_decision": final_decision,
            "conflict": abs(prob_tab - prob_cat) > 0.3
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000)