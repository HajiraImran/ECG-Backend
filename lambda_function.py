import json
import os
import numpy as np
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import find_peaks, butter, lfilter

# 1. Firebase Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://trilyte-37e53-default-rtdb.firebaseio.com'
    })

# 2. Global Model & Scaler
MODEL = tf.keras.models.load_model('my_model.h5', compile=False)
SCALER = joblib.load('scaler.pkl') 

def get_level(value, low, high):
    if value < low: return "LOW ⬇"
    elif value > high: return "HIGH ⬆"
    else: return "NORMAL ✓"

def extract_features(signal, fs=250):
    signal = np.array(signal)
    
    # --- Step 1: Bandpass Filter ---
    nyq = 0.5 * fs
    low, high = 1.0 / nyq, 35.0 / nyq # 1.0Hz filter for better baseline
    b, a = butter(1, [low, high], btype='band')
    filtered = lfilter(b, a, signal)
    
    # --- Step 2: Flexible R-Peak Detection ---
    peaks, _ = find_peaks(filtered, 
                          distance=int(fs*0.3), 
                          prominence=np.std(filtered) * 0.7,
                          height=None) 
    
    if len(peaks) < 3:
        return None

    # --- Step 3: Calculate Intervals ---
    rr_intervals = np.diff(peaks) * (1000 / fs)
    rr_avg = np.mean(rr_intervals)
    
    qrs_list = []
    qt_list = []

    for r in peaks[1:-1]:
        q_idx = r - int(0.04 * fs)
        s_idx = r + int(0.04 * fs)
        qrs_ms = (s_idx - q_idx) * (1000 / fs)
        qrs_list.append(qrs_ms)
        
        t_start = r + int(0.1 * fs)
        t_end = r + int(0.45 * fs)
        if t_end < len(filtered):
            t_segment = filtered[t_start:t_end]
            t_peak_relative = np.argmax(t_segment)
            t_peak_idx = t_start + t_peak_relative
            qt_ms = (t_peak_idx - q_idx) * (1000 / fs)
            qt_list.append(qt_ms)

    qrs_avg = np.median(qrs_list) if qrs_list else 95.0
    qt_avg = np.median(qt_list) if qt_list else 400.0
    qtc_avg = qt_avg / np.sqrt(rr_avg / 1000)

    return [rr_avg, 160.0, qrs_avg, qt_avg, qtc_avg, 50.0, 60.0, 45.0]

def handler(event, context):
    try:
        # Request Parsing
        body = event.get('body', {})
        if isinstance(body, str): body = json.loads(body)
        uid = body.get('uid') or (event.get('queryStringParameters', {}).get('uid'))

        if not uid:
            return {'statusCode': 400, 'body': json.dumps("Error: Missing uid")}

        # Get Latest Data
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

        # Prediction
        features_scaled = SCALER.transform([features])
        preds = MODEL.predict(features_scaled)
        
        reg_out = preds[1][0]
        
        # --- Value Adjustments (Offsets) ---
        k_val = round(float(reg_out[0]), 2)
        ca_val = round(float(reg_out[1]) + 3.0, 2) 
        mg_val = round(float(reg_out[2]) + 1.2, 2) 

        # --- Formatting with Ranges for Dashboard ---
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
            "Status": "Normal", # This will replace "Analyzing" on the home screen
            "Timestamp": data_entry.get('timestamp', 0)
        }

        # Save to Firebase
        db.reference(f'users/{uid}/latest_results').set(final_results)
        db.reference(f'users/{uid}/ecg_data/{key}/results').set(final_results)

        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'},
            'body': json.dumps(final_results)
        }

    except Exception as e:
        print(f"System Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({"details": str(e)})}