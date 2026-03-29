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

# Automatically inject the HuggingFace token for gated models
os.environ["HF_TOKEN"] = "[hf token here]"

app = Flask(__name__)
CORS(app)

# Global variables to store our model and preprocessors
clf = None
encoder = None
feature_columns = None
model_ready = False
model_error = None
model_accuracy = 0.0

def initialize_model():
    global clf, encoder, feature_columns, model_ready, model_error, model_accuracy
    try:
        print("Fetching German Credit Dataset...")
        data = fetch_openml('credit-g', version=1, as_frame=True)
        df = data.frame
        
        # The target is 'class' ('good' or 'bad')
        X = df.drop(columns=['class'])
        y = df['class']
        
        feature_columns = X.columns.tolist()
        print(f"Features loaded: {len(feature_columns)}")
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Initialize the encoder for categorical data
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Fit and transform categorical data to numerical
        X_encoded = X.copy()
        if categorical_cols:
            X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])
        
        # Convert 'good'/'bad' to 1/0
        y_encoded = y.map({'good': 1, 'bad': 0})
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
        
        print("Initializing TabPFN 2.5 Classifier...")
        # Initialize TabPFN 2.5
        clf = TabPFNClassifier()
        
        print("Fitting model...")
        clf.fit(X_train, y_train)
        
        print("Evaluating model...")
        preds = clf.predict(X_test)
        model_accuracy = accuracy_score(y_test, preds)
        print(f"Model initialized successfully. Accuracy: {model_accuracy:.2f}")
        
        model_ready = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_error = str(e)

# Start model initialization in background so server starts quickly
INIT_THREAD = threading.Thread(target=initialize_model)
INIT_THREAD.start()

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'ready': model_ready,
        'error': model_error,
        'accuracy': model_accuracy,
        'features': feature_columns
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_ready:
        return jsonify({'error': 'Model is still initializing, please try again in a few moments.'}), 503
        
    try:
        user_input = request.json
        
        # Create a single row DataFrame from user input, matching the feature columns order
        input_data = {}
        for col in feature_columns:
            # We expect the frontend to pass values. If missing, we might need a default
            input_data[col] = [user_input.get(col, None)] 
            
        input_df = pd.DataFrame(input_data)
        
        # We need to ensure types match. Categorical to 'category' or 'object'
        categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols and encoder:
             input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
             
        # Make sure datatypes are numeric
        input_df = input_df.astype(float)
        
        prediction = clf.predict(input_df)
        probability = clf.predict_proba(input_df)
        
        result = int(prediction[0]) # 1 = good, 0 = bad
        confidence = float(probability[0][result])
        
        return jsonify({
            'status': 'success',
            'prediction': 'Approved' if result == 1 else 'Rejected',
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
