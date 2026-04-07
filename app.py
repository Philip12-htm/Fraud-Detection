from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# --- FIX 1: INITIALIZE WITH EMPTY DATA INSTEAD OF NONE ---
# This prevents the "NoneType" error even if loading fails.
models = {}
scaler = None
features = []  # Empty list is iterable; None is not.
encoders = {}  # Empty dict is iterable; None is not.

# --- LOAD ML ASSETS ---
def load_asset(filename):
    # 1. Get all files in the current directory
    files_in_root = os.listdir('.')
    
    # 2. Look for a file that matches the name even if it has extra spaces
    target_file = None
    for f in files_in_root:
        if f.strip() == filename: # This removes spaces from 'tuned_rf_model.pkl '
            target_file = f
            break
            
    if target_file and os.path.exists(target_file):
        try:
            print(f"DEBUG: Found match for {filename} as '{target_file}'")
            return joblib.load(target_file)
        except Exception as e:
            print(f"Error loading {target_file}: {e}")
    return None

try:
    # We still try to load with the correct names
    models['rf'] = load_asset('tuned_rf_model.pkl')
    models['xgb'] = load_asset('xgboost_model.pkl')
    scaler = load_asset('scaler.pkl')
    features = load_asset('feature_list.pkl')
    encoders = load_asset('label_encoders_dict.pkl')

    if models.get('rf') and scaler and features and encoders:
        print("✅ SUCCESS: All assets loaded perfectly despite filename spaces.")
    else:
        missing = [f for f, v in [("RF", models.get('rf')), ("Scaler", scaler), ("Features", features), ("Encoders", encoders)] if v is None]
        print(f"⚠️ WARNING: Missing: {missing}")
        # Print a list of what IS actually in the root to help us debug
        print(f"DEBUG: Files currently in root: {os.listdir('.')}")

except Exception as e:
    print(f"FATAL STARTUP ERROR: {e}")

# Global history storage
transaction_history = []

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def analyze_risk(data, prob):
    reasons = []
    # Force conversion to float/int to avoid comparison errors
    amt = float(data.get('amt', 0))
    dist = float(data.get('distance', 0))
    hour = int(data.get('hour', 12))  
    
    # 1. Collect Reasons
    if amt > 1000: reasons.append("High Transaction Value") # Increased threshold to match report
    if dist > 200: reasons.append("Extreme Distance Anomaly") 
    
    # Only flag Irregular Time if it's actually 0-5 AM
    if 0 <= hour <= 5: 
        reasons.append("Irregular Execution Time (Night)")
    
    # 2. Risk Level Logic
    if prob >= 0.8 or dist > 1000 or amt > 5000:
        level = "CRITICAL"
    elif prob >= 0.5 or dist > 300:
        level = "ELEVATED"
    else:
        level = "STABLE"
        
    return level, reasons

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_choice = data.get("model", "rf")
        
        # --- TIME HANDLING LOGIC ---
        raw_hour = int(data.get('hour', 12))
        ampm = data.get('ampm', 'AM') # Ensure your HTML sends this or handles it
        
        # Convert to 24-hour format for the ML Model
        if ampm == 'PM' and raw_hour < 12:
            final_hour = raw_hour + 12
        elif ampm == 'AM' and raw_hour == 12:
            final_hour = 0
        else:
            final_hour = raw_hour
            
        input_df = pd.DataFrame([data])

        # 1. Feature Engineering
        input_df['age'] = int(data.get('age', 30))
        input_df['hour'] = final_hour # Use the converted 24h time
        input_df['day'] = datetime.now().day
        input_df['month'] = datetime.now().month
        
        u_lat, u_long = float(data.get('lat', 40.0)), float(data.get('long', -74.0))
        m_lat, m_long = float(data.get('merch_lat', 40.05)), float(data.get('merch_long', -74.05))
        input_df['distance'] = haversine(u_lat, u_long, m_lat, m_long)
        
        # 2. Keep the signals!
        input_df['merchant'] = float(data.get('merchant', 0.001))
        input_df['city'] = float(data.get('city', 0.001))
        input_df['transactions_per_hour'] = int(data.get('transactions_per_hour', 1))

        # 3. Fill missing for Scaler
        missing = {'zip': 0, 'lat': u_lat, 'long': u_long, 'city_pop': 1000, 
                   'merch_lat': m_lat, 'merch_long': m_long, 'transactions_per_hour': input_df['transactions_per_hour'].iloc[0]}
        for feat, val in missing.items():
            if feat not in input_df.columns:
                input_df[feat] = val

        # 4. Label Encoding
        categorical_cols = ['gender', 'category', 'state', 'job']
        for col in categorical_cols:
            if col in encoders:
                le = encoders[col]
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    # Fallback to avoid crash
                    input_df[col] = le.transform([le.classes_[0]])[0]

        # 5. Scaling & Prediction
        final_df = input_df[features]
        scaled_data = scaler.transform(final_df)
        target_model = models.get(model_choice) or models.get('rf')
        
        if target_model is None:
            return jsonify({"error": "Machine Learning model not initialized. Check server logs."}), 500
            
        probability = float(target_model.predict_proba(scaled_data)[0][1])
        probability = float(target_model.predict_proba(scaled_data)[0][1])

        # 6. SMART BOOST SYSTEM
        amt = float(data.get('amt', 0))
        dist = input_df['distance'].iloc[0]
        hour = input_df['hour'].iloc[0]
        tph = input_df['transactions_per_hour'].iloc[0]

        boost = 0
        if amt > 1000: boost += 0.15
        if amt > 3000: boost += 0.25
        if dist > 200: boost += 0.15
        if dist > 500: boost += 0.25
        if 0 <= hour <= 5: boost += 0.15
        if tph >= 3: boost += 0.2

        probability = min(probability + boost, 0.99)
        
        # 7. Risk Analysis & Logic Layer
        risk_level, reasons = analyze_risk(input_df.iloc[0], probability)

        if risk_level == "CRITICAL":
            prediction = 1
        else:
            prediction = 1 if probability >= 0.5 else 0
        
        result = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "reasons": reasons if reasons else ["No Major Deviations"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_choice.upper(),
            "amt": float(data.get('amt', 0)),
            "category": str(data.get('category', 'general'))
        }
        
        transaction_history.insert(0, result)
        return jsonify(result)

    except Exception as e:
        print(f"DEPLOYMENT ERROR: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/')
def index():
    transaction_history = [] 
    return render_template('dashboard.html')

@app.route('/predict-page')
def pred_page(): return render_template('predict.html')

@app.route('/history-page')
def hist_page():
    transaction_history = []
    return render_template('history.html')
@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    global transaction_history
    transaction_history = []
    return jsonify({"status": "reset"})

@app.route('/api/stats')
def get_stats():
    total = len(transaction_history)
    if total == 0: return jsonify({"total":0, "fraud":0, "rate":0, "avg_risk":0, "dist":{"low":0,"med":0,"high":0}})
    fraud = sum(1 for x in transaction_history if x['prediction'] == 1)
    return jsonify({
        "total": total, "fraud": fraud, "rate": round((fraud/total*100), 2),
        "avg_risk": round(np.mean([x['probability'] for x in transaction_history]), 2),
        "dist": {
            "low": sum(1 for x in transaction_history if x['risk_level'] == "STABLE"),
            "med": sum(1 for x in transaction_history if x['risk_level'] == "ELEVATED"),
            "high": sum(1 for x in transaction_history if x['risk_level'] == "CRITICAL")
        }
    })

@app.route('/api/history')
def get_history(): return jsonify(transaction_history)
# --- ADD THIS NEW ROUTE AT THE BOTTOM OF app.py ---

# --- REPLACE FROM @app.route('/predict-csv') DOWN TO return jsonify(...) ---
@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        model_choice = request.form.get("model", "rf")
        df = pd.read_csv(file)

        # 1. SMART COLUMN AUTO-DETECTION
        if 'trans_date_trans_time' in df.columns:
            df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
        
        if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
            df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

        mapping = {'amt': ['amt', 'amount'], 'category': ['category', 'type'], 'hour': ['hour', 'time'], 'distance': ['distance', 'dist']}
        for target, aliases in mapping.items():
            if target not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df[target] = df[alias]
                        break

        # 2. VECTORIZED PROCESSING
        process_df = df.copy()
        for feat in features:
            if feat not in process_df.columns: process_df[feat] = 0
        
        process_df['day'] = datetime.now().day
        process_df['month'] = datetime.now().month
        process_df['age'] = process_df['age'] if 'age' in process_df.columns else 30
        process_df['transactions_per_hour'] = 1

        for col in ['gender', 'category', 'state', 'job']:
            if col in encoders:
                le = encoders[col]
                process_df[col] = process_df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else le.transform([le.classes_[0]])[0])

        # 3. PREDICTION
        scaled_data = scaler.transform(process_df[features])
        probs = models.get(model_choice, models['rf']).predict_proba(scaled_data)[:, 1]
        
        # 4. BOOST & LOGIC
        boosts = np.where(df['amt'] > 1000, 0.15, 0) + np.where(df['hour'] <= 5, 0.15, 0)
        final_probs = np.clip(probs + boosts, 0, 0.99)

        # 5. SAVE ALL TO HISTORY (This is what updates the dashboard)
        for i in range(len(df)):
            p = float(final_probs[i])
            risk = "CRITICAL" if p >= 0.8 else ("ELEVATED" if p >= 0.5 else "STABLE")
            transaction_history.insert(0, {
                "prediction": 1 if p >= 0.5 else 0,
                "probability": round(p, 4),
                "risk_level": risk,
                "amt": float(df.iloc[i]['amt']),
                "category": str(df.iloc[i]['category']),
                "model": model_choice.upper(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # 6. RETURN RESULTS (Limit UI display to 100)
        display_df = df.head(100).copy()
        display_df['prob'] = (final_probs[:100] * 100).round(2)
        display_df['verdict'] = np.where(final_probs[:100] >= 0.5, "FRAUD", "LEGIT")
        display_df['risk'] = [("CRITICAL" if p >= 0.8 else ("ELEVATED" if p >= 0.5 else "STABLE")) for p in final_probs[:100]]

        return jsonify({
            "results": display_df[['amt', 'category', 'prob', 'risk', 'verdict']].to_dict(orient='records'),
            "total_processed": len(df)
        })

    except Exception as e:
        print(f"BATCH ERROR: {e}")
        return jsonify({"error": str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug=True, port=5000)
