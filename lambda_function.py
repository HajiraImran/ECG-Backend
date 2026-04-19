import json
import os
import numpy as np
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import find_peaks, butter, filtfilt

# 1. Firebase Initialization
if not firebase_admin._apps:
    # Ensure 'firebase_key.json' is in your backend folder
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://trilyte-37e53-default-rtdb.firebaseio.com'
    })

# 2. Global Model & Scalers (Must be in the same folder)
# We use .h5 for better compatibility with older Lambda environments
MODEL = tf.keras.models.load_model('my_model.h5', compile=False)
SCALER = joblib.load('scaler.pkl') 
TARGET_SCALER = joblib.load('target_scaler.pkl')

def get_level(value, low, high):
    if value < low: return "LOW ⬇"
    elif value > high: return "HIGH ⬆"
    else: return "NORMAL ✓"

def extract_features(signal, fs=250):
    """
    Extracts ECG intervals and matches the 8-feature format 
    used during model training.
    """
    signal = np.array(signal, dtype=float)
    
    # --- Step 1: Cleaning (Bandpass Filter) ---
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, 40.0 / nyq 
    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, signal) 
    
    # Normalize
    norm = (filtered - filtered.mean()) / filtered.std()
    
    # --- Step 2: R-Peak Detection ---
    # distance=180 samples (approx 0.7s at 250Hz)
    peaks, _ = find_peaks(norm, height=0.3, distance=150, prominence=0.5) 
    
    if len(peaks) < 3:
        return None

    # --- Step 3: Intervals Calculation ---
    rr_ms = np.diff(peaks) / fs * 1000
    RR = np.mean(rr_ms)
    
    pr_list, qrs_list, qt_list = [], [], []

    for r in peaks[1:-1]:
        # QRS Detection
        q_idx = max(0, r - int(0.05 * fs)) + np.argmin(norm[max(0, r - int(0.05 * fs)):r])
        s_idx = r + np.argmin(norm[r:min(len(norm), r + int(0.05 * fs))])
        
        qrs_ms = (s_idx - q_idx) / fs * 1000
        if 40 < qrs_ms < 200: qrs_list.append(qrs_ms)
        
        # PR Detection (P-wave search)
        p_start, p_end = max(0, r - int(0.2 * fs)), max(0, r - int(0.06 * fs))
        if p_end > p_start:
            p_idx = p_start + np.argmax(norm[p_start:p_end])
            pr_ms = (r - p_idx) / fs * 1000
            if 80 < pr_ms < 300: pr_list.append(pr_ms)
            
        # QT Detection (T-wave search)
        t_win = norm[s_idx:min(len(norm), s_idx + int(0.4 * fs))]
        if len(t_win) > 10:
            tp = np.argmax(t_win)
            above = np.where(t_win[tp:] > 0.1 * t_win[tp])[0]
            if len(above):
                qt_ms = (s_idx + tp + above[-1] - q_idx) / fs * 1000
                if 200 < qt_ms < 600: qt_list.append(qt_ms)

    PR = np.median(pr_list) if pr_list else 160.0
    QRS = np.median(qrs_list) if qrs_list else 85.0
    QT = np.median(qt_list) if qt_list else 400.0
    QTc = QT / np.sqrt(RR / 1000)

    # Returning 8 features to match training: [RR, PR, QRS, QT, QTc, P_axis, QRS_axis, T_axis]
    # Axes are set to normal defaults (60, 60, 45) as raw signal axis needs multi-lead ECG.
    return [RR, PR, QRS, QT, QTc, 60.0, 60.0, 45.0]

def handler(event, context):
    try:
        # Request Parsing
        body = event.get('body', {})
        if isinstance(body, str): body = json.loads(body)
        uid = body.get('uid') or (event.get('queryStringParameters', {}).get('uid'))

        if not uid:
            return {'statusCode': 400, 'body': json.dumps("Error: Missing uid")}

        # Fetch Data from Firebase
        ref = db.reference(f'users/{uid}/ecg_data')
        snapshot = ref.order_by_child('timestamp').limit_to_last(1).get()
        
        if not snapshot:
            return {'statusCode': 404, 'body': json.dumps("Error: No data found")}

        key = list(snapshot.keys())[0]
        data_entry = snapshot[key]
        raw_ecg_values = data_entry.get('values')

        # Feature Extraction
        features = extract_features(raw_ecg_values, fs=250)
        
        if features is None:
            db.reference(f'users/{uid}/latest_results').update({"Status": "Signal Error: Stay Still"})
            return {'statusCode': 422, 'body': json.dumps({"error": "Poor signal quality"})}

        # AI Prediction
        features_scaled = SCALER.transform([features])
        preds = MODEL.predict(features_scaled, verbose=0)
        
        # Classification (Imbalance check)
        # preds[0] is classification, preds[1] is regression
        is_imbalanced = preds[0][0] > 0.5 
        
        # Regression (Actual Values) via Target Scaler
        reg_out_real = TARGET_SCALER.inverse_transform(preds[1])[0]
        
        k_val = round(float(reg_out_real[0]), 2)
        ca_val = round(float(reg_out_real[1]), 2)
        mg_val = round(float(reg_out_real[2]), 2)

        # Build Final Results Object
        final_results = {
            "Potassium": {
                "Value": k_val, 
                "Level": get_level(k_val, 3.5, 5.0), 
                "Range": "3.5 - 5.0", 
                "Unit": "mEq/L"
            },
            "Calcium": {
                "Value": ca_val, 
                "Level": get_level(ca_val, 8.5, 10.5), 
                "Range": "8.5 - 10.5", 
                "Unit": "mg/dL"
            },
            "Magnesium": {
                "Value": mg_val, 
                "Level": get_level(mg_val, 1.7, 2.2), 
                "Range": "1.7 - 2.2", 
                "Unit": "mg/dL"
            },
            "BPM": round(60000 / features[0], 1),
            "Status": "Normal" if not is_imbalanced else "Imbalance Detected",
            "Timestamp": data_entry.get('timestamp', 0)
        }

        # Update Firebase
        db.reference(f'users/{uid}/latest_results').set(final_results)
        db.reference(f'users/{uid}/ecg_data/{key}/results').set(final_results)

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(final_results)
        }

    except Exception as e:
        print(f"System Error: {str(e)}")
        return {
            'statusCode': 500, 
            'body': json.dumps({"details": str(e)})
        }