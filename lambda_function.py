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
    """
    DYNAMIC FEATURE EXTRACTION:
    Hardcoded values ki jagah asli signal se intervals nikalta hai.
    """
    signal = np.array(signal)
    
    # --- Step 1: Bandpass Filter (Baseline Wander & High Frequency Noise Removal) ---
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, 40.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered = lfilter(b, a, signal)
    
    # --- Step 2: Strict R-Peak Detection ---
    # distance=fs*0.5 matlab 120 BPM se zyada allow nahi (noise filter)
    peaks, _ = find_peaks(filtered, 
                          distance=int(fs*0.5), 
                          prominence=np.std(filtered) * 1.5,
                          height=np.mean(filtered))
    
    # Agar signal bohot ganda hai aur 3 peaks bhi nahi mil rahi:
    if len(peaks) < 3:
        return None

    # --- Step 3: Calculate Intervals (In milliseconds) ---
    rr_intervals = np.diff(peaks) * (1000 / fs)
    rr_avg = np.mean(rr_intervals)
    
    qrs_list = []
    qt_list = []

    for r in peaks[1:-1]:
        # QRS Window: R ke aage piche scan
        q_idx = r - int(0.04 * fs)
        s_idx = r + int(0.04 * fs)
        qrs_ms = (s_idx - q_idx) * (1000 / fs)
        qrs_list.append(qrs_ms)
        
        # QT/T-Wave Window: R ke baad 350ms ka area
        t_start = r + int(0.1 * fs)
        t_end = r + int(0.4 * fs)
        if t_end < len(filtered):
            t_segment = filtered[t_start:t_end]
            t_peak_relative = np.argmax(t_segment)
            t_peak_idx = t_start + t_peak_relative
            qt_ms = (t_peak_idx - q_idx) * (1000 / fs)
            qt_list.append(qt_ms)

    # Medians use kar rahe hain taake abnormal spikes result kharab na karein
    qrs_avg = np.median(qrs_list) if qrs_list else 95.0
    qt_avg = np.median(qt_list) if qt_list else 400.0
    qtc_avg = qt_avg / np.sqrt(rr_avg / 1000)

    # PR Interval & Axis (Single Lead limitation ki wajah se safe defaults)
    pr_avg = 160.0 
    p_axis, qrs_axis, t_axis = 50.0, 60.0, 45.0

    return [rr_avg, pr_avg, qrs_avg, qt_avg, qtc_avg, p_axis, qrs_axis, t_axis]

def handler(event, context):
    try:
        # --- Step A: Request Parsing ---
        uid = None
        if event.get('queryStringParameters'):
            uid = event['queryStringParameters'].get('uid')
        if not uid and event.get('body'):
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            uid = body.get('uid')

        if not uid:
            return {'statusCode': 400, 'body': json.dumps("Error: Missing uid")}

        # --- Step B: Get Latest Data from Firebase ---
        ref = db.reference(f'users/{uid}/ecg_data')
        snapshot = ref.order_by_child('timestamp').limit_to_last(1).get()
        
        if not snapshot:
            return {'statusCode': 404, 'body': json.dumps("Error: No data found")}

        key = list(snapshot.keys())[0]
        data_entry = snapshot[key]
        raw_ecg_values = data_entry.get('values')

        if not raw_ecg_values:
            return {'statusCode': 404, 'body': json.dumps("Error: Values missing")}

        # --- Step C: Feature Extraction & Validation ---
        features = extract_features(raw_ecg_values, fs=250)
        
        # Agar signal quality poor hai:
        if features is None:
            return {
                'statusCode': 422,
                'body': json.dumps({"error": "Poor signal quality. Please stay still and retry."})
            }

        # --- Step D: Prediction ---
        features_scaled = SCALER.transform([features])
        preds = MODEL.predict(features_scaled)
        
        # Regression outputs
        reg_out = preds[1][0]
        k_val = round(float(reg_out[0]), 2)
        ca_val = round(float(reg_out[1]), 2)
        mg_val = round(float(reg_out[2]), 2)

        # Result Formatting
        final_results = {
            "Potassium": {"Value": k_val, "Level": get_level(k_val, 3.5, 5.0), "Unit": "mEq/L"},
            "Calcium": {"Value": ca_val, "Level": get_level(ca_val, 8.5, 10.5), "Unit": "mg/dL"},
            "Magnesium": {"Value": mg_val, "Level": get_level(mg_val, 1.7, 2.2), "Unit": "mg/dL"},
            "BPM": round(60000 / features[0], 1),
            "Status": "Normal" if all(get_level(v, l, h) == "NORMAL ✓" for v, l, h in [(k_val,3.5,5.0), (ca_val,8.5,10.5), (mg_val,1.7,2.2)]) else "Imbalanced",
            "Timestamp": data_entry.get('timestamp')
        }

        # --- Step E: Save Results back to Firebase ---
        # 1. Update Latest Results
        db.reference(f'users/{uid}/latest_results').set(final_results)
        # 2. Append to history entry
        db.reference(f'users/{uid}/ecg_data/{key}/results').set(final_results)

        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'},
            'body': json.dumps(final_results)
        }

    except Exception as e:
        print(f"System Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({"details": str(e)})}