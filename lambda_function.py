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
    low, high = 0.5 / nyq, 40.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered = lfilter(b, a, signal)
    
    # --- Step 2: Flexible R-Peak Detection (UPDATED) ---
    # Prominence 1.5 se kam kar ke 0.7 kiya taake round peaks bhi mil jayen
    # Distance 0.5s se kam kar ke 0.3s kiya (BPM range 40-200 allow)
    peaks, _ = find_peaks(filtered, 
                          distance=int(fs*0.3), 
                          prominence=np.std(filtered) * 0.7,
                          height=None) # Height check hata diya taake baseline shift masla na kare
    
    # Minimum 3 peaks zaroori hain features ke liye
    if len(peaks) < 3:
        print(f"[DEBUG] Only {len(peaks)} peaks found. Signal quality poor.")
        return None

    # --- Step 3: Calculate Intervals ---
    rr_intervals = np.diff(peaks) * (1000 / fs)
    rr_avg = np.mean(rr_intervals)
    
    qrs_list = []
    qt_list = []

    for r in peaks[1:-1]:
        # QRS Window: R ke aage piche 40ms
        q_idx = r - int(0.04 * fs)
        s_idx = r + int(0.04 * fs)
        qrs_ms = (s_idx - q_idx) * (1000 / fs)
        qrs_list.append(qrs_ms)
        
        # QT Window: R ke baad T-wave dhoondna
        t_start = r + int(0.1 * fs)
        t_end = r + int(0.45 * fs) # Window thodi barhai (0.4 to 0.45)
        if t_end < len(filtered):
            t_segment = filtered[t_start:t_end]
            t_peak_relative = np.argmax(t_segment)
            t_peak_idx = t_start + t_peak_relative
            qt_ms = (t_peak_idx - q_idx) * (1000 / fs)
            qt_list.append(qt_ms)

    # Medians calculate karein
    qrs_avg = np.median(qrs_list) if qrs_list else 95.0
    qt_avg = np.median(qt_list) if qt_list else 400.0
    qtc_avg = qt_avg / np.sqrt(rr_avg / 1000)

    # Defaults for single lead
    pr_avg = 160.0 
    p_axis, qrs_axis, t_axis = 50.0, 60.0, 45.0

    return [rr_avg, pr_avg, qrs_avg, qt_avg, qtc_avg, p_axis, qrs_axis, t_axis]

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
            # Dashboard update karein taake user ko pata chale signal kharab tha
            db.reference(f'users/{uid}/latest_results').update({"Status": "Signal Error: Stay Still"})
            return {
                'statusCode': 422,
                'body': json.dumps({"error": "Poor signal quality. Check electrode placement."})
            }

        # Prediction
        features_scaled = SCALER.transform([features])
        preds = MODEL.predict(features_scaled)
        
        reg_out = preds[1][0]
        k_val = round(float(reg_out[0]), 2)
        ca_val = round(float(reg_out[1]), 2)
        mg_val = round(float(reg_out[2]), 2)

        # Formatting
        final_results = {
            "Potassium": {"Value": k_val, "Level": get_level(k_val, 3.5, 5.0), "Unit": "mEq/L"},
            "Calcium": {"Value": ca_val, "Level": get_level(ca_val, 8.5, 10.5), "Unit": "mg/dL"},
            "Magnesium": {"Value": mg_val, "Level": get_level(mg_val, 1.7, 2.2), "Unit": "mg/dL"},
            "BPM": round(60000 / features[0], 1),
            "Status": "NORMAL ✓" if all(get_level(v, l, h) == "NORMAL ✓" for v, l, h in [(k_val,3.5,5.0), (ca_val,8.5,10.5), (mg_val,1.7,2.2)]) else "Imbalanced",
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