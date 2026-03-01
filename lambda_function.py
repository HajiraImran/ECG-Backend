import json
import os
import numpy as np
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import find_peaks, butter, lfilter

# 1. Firebase Initialization (Same as before)
if not firebase_admin._apps:
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://trilyte-37e53-default-rtdb.firebaseio.com'
    })

# 2. Global Model & Scaler (Compile=False added for safety)
MODEL = tf.keras.models.load_model('my_model.h5', compile=False)
SCALER = joblib.load('scaler.pkl') 

def get_level(value, low, high):
    if value < low:
        return "LOW ⬇"
    elif value > high:
        return "HIGH ⬆"
    else:
        return "NORMAL ✓"

def extract_features(signal, fs=250):
    # ... (Aapka purana extract_features code yahan rahega)
    signal = np.array(signal)
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, 40.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered_signal = lfilter(b, a, signal)
    peaks, _ = find_peaks(filtered_signal, distance=fs*0.6, height=np.mean(filtered_signal))
    if len(peaks) < 2:
        return [800.0, 160.0, 90.0, 400.0, 410.0, 50.0, 60.0, 40.0]
    rr_intervals = np.diff(peaks) * (1000 / fs)
    rr_avg = np.mean(rr_intervals)
    pr_avg, qrs_avg, qt_avg = 160.0, 95.0, 400.0
    qtc_avg = qt_avg / np.sqrt(rr_avg/1000)
    p_axis, qrs_axis, t_axis = 50.0, 60.0, 40.0
    return [rr_avg, pr_avg, qrs_avg, qt_avg, qtc_avg, p_axis, qrs_axis, t_axis]

def handler(event, context):
    try:
        # --- STEP A: Enhanced Request Parsing ---
        uid = None

        # 1. Check URL Query Parameters (For Browser/ESP32 URL hits)
        if event.get('queryStringParameters'):
            uid = event['queryStringParameters'].get('uid')

        # 2. Check JSON Body (For POST requests/Mobile App)
        if not uid:
            if isinstance(event.get('body'), str):
                body = json.loads(event['body'])
                uid = body.get('uid')
            elif isinstance(event, dict):
                uid = event.get('uid')

        # If still no UID, return error
        if not uid:
            return {
                'statusCode': 400, 
                'body': json.dumps("Error: Missing uid. Please provide ?uid=YOUR_ID in the URL")
            }

        # Step B: Firebase Snapshot
        print(f"Processing data for UID: {uid}")
        ref = db.reference(f'users/{uid}/ecg_data')
        ref = db.reference(f'users/{uid}/ecg_data')
        snapshot = ref.order_by_child('timestamp').limit_to_last(1).get()
        
        if not snapshot or not isinstance(snapshot, dict):
            return {'statusCode': 404, 'body': json.dumps("Error: No data found")}

        key = list(snapshot.keys())[0]
        data_entry = snapshot[key]
        raw_ecg_values = data_entry.get('values')
        bpm_from_esp32 = data_entry.get('bpm', 'N/A')

        if not raw_ecg_values:
            return {'statusCode': 404, 'body': json.dumps("Error: Values missing")}

        # Step C: Features & Prediction
        features = extract_features(raw_ecg_values, fs=250) 
        features_scaled = SCALER.transform([features])
        preds = MODEL.predict(features_scaled)
        
        # Electrolytes Prediction (reg_out)
        reg_out = preds[1][0]
        k_val = round(float(reg_out[0]), 2)
        ca_val = round(float(reg_out[1]), 2)
        mg_val = round(float(reg_out[2]), 2)

        # Step D: New Format - Lab Report Style
        final_results = {
            "Potassium": {
                "Value": f"{k_val} mEq/L",
                "Level": get_level(k_val, 3.5, 5.0),
                "Range": "Normal is 3.5–5.0 mEq/L"
            },
            "Calcium": {
                "Value": f"{ca_val} mg/dL",
                "Level": get_level(ca_val, 8.5, 10.5),
                "Range": "Normal is 8.5–10.5 mg/dL"
            },
            "Magnesium": {
                "Value": f"{mg_val} mg/dL",
                "Level": get_level(mg_val, 1.7, 2.2),
                "Range": "Normal is 1.7–2.2 mg/dL"
            },
            "BPM": bpm_from_esp32,
            "Overall_Status": "Imbalanced" if any(l in [get_level(k_val, 3.5, 5.0), get_level(ca_val, 8.5, 10.5), get_level(mg_val, 1.7, 2.2)] for l in ["LOW ⬇", "HIGH ⬆"]) else "Normal",
            "Timestamp": data_entry.get('timestamp', 'N/A')
        }

        # Step E: Results Save
        db.reference(f'users/{uid}/results').set(final_results)

        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(final_results)
        }

    except Exception as e:
        print(f"System Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({"details": str(e)})}