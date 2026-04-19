import json
import numpy as np
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import find_peaks, butter, lfilter

# --- Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://trilyte-37e53-default-rtdb.firebaseio.com'
    })

# --- ✅ MODEL LOAD (FINAL FIX) ---
MODEL = tf.keras.models.load_model('model_dir')
SCALER = joblib.load('scaler.pkl')
TARGET_SCALER = joblib.load('target_scaler.pkl')

# --- Utils ---
def get_level(value, low, high):
    if value < low: return "LOW ⬇"
    elif value > high: return "HIGH ⬆"
    else: return "NORMAL ✓"

# --- Feature Extraction ---
def extract_features(signal, fs=250):
    signal = np.array(signal)
    
    nyq = 0.5 * fs
    low, high = 1.0 / nyq, 35.0 / nyq 
    b, a = butter(1, [low, high], btype='band')
    filtered = lfilter(b, a, signal)

    peaks, _ = find_peaks(filtered, 
                          distance=int(fs*0.3), 
                          prominence=np.std(filtered) * 0.7) 

    if len(peaks) < 3:
        return None

    rr_intervals = np.diff(peaks) * (1000 / fs)
    RR = np.mean(rr_intervals)

    qrs_list, qt_list = [], []
    for r in peaks[1:-1]:
        q_idx = r - int(0.04 * fs)
        s_idx = r + int(0.04 * fs)
        qrs_ms = (s_idx - q_idx) * (1000 / fs)
        qrs_list.append(qrs_ms)

        t_start, t_end = r + int(0.1 * fs), r + int(0.45 * fs)
        if t_end < len(filtered):
            t_segment = filtered[t_start:t_end]
            t_peak_idx = t_start + np.argmax(t_segment)
            qt_ms = (t_peak_idx - q_idx) * (1000 / fs)
            qt_list.append(qt_ms)

    QRS = np.median(qrs_list) if qrs_list else 95.0
    QT  = np.median(qt_list) if qt_list else 400.0
    QTc = QT / np.sqrt(RR / 1000)

    # 8 features match training
    return [RR, 160.0, QRS, QT, QTc, 60.0, 60.0, 45.0]

# --- Handler ---
def handler(event, context):
    try:
        body = event.get('body', {})
        if isinstance(body, str):
            body = json.loads(body)

        uid = body.get('uid') or (event.get('queryStringParameters', {}).get('uid'))

        if not uid:
            return {'statusCode': 400, 'body': json.dumps("Missing uid")}

        ref = db.reference(f'users/{uid}/ecg_data')
        snapshot = ref.order_by_child('timestamp').limit_to_last(1).get()

        if not snapshot:
            return {'statusCode': 404, 'body': json.dumps("No data")}

        key = list(snapshot.keys())[0]
        data_entry = snapshot[key]
        raw_ecg_values = data_entry.get('values')

        features = extract_features(raw_ecg_values)

        if features is None:
            return {'statusCode': 422, 'body': json.dumps("Poor signal")}

        # --- Prediction ---
        features_scaled = SCALER.transform([features])
        pred_class, pred_reg_scaled = MODEL.predict(features_scaled, verbose=0)

        # --- Convert back ---
        pred_reg = TARGET_SCALER.inverse_transform(pred_reg_scaled)[0]

        k_val  = round(float(pred_reg[0]), 2)
        ca_val = round(float(pred_reg[1]), 2)
        mg_val = round(float(pred_reg[2]), 2)

        final_results = {
            "Potassium": {"Value": k_val, "Level": get_level(k_val, 3.5, 5.0)},
            "Calcium": {"Value": ca_val, "Level": get_level(ca_val, 8.5, 10.5)},
            "Magnesium": {"Value": mg_val, "Level": get_level(mg_val, 1.7, 2.2)},
            "BPM": round(60000 / features[0], 1),
            "Status": "OK"
        }

        db.reference(f'users/{uid}/latest_results').set(final_results)

        return {
            'statusCode': 200,
            'body': json.dumps(final_results)
        }

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(str(e))}