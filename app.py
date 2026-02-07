from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import librosa
import io
import webbrowser  # <-- add this
import threading 
import os

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)  # Allow all origins (fixes CORS errors)

# ================= LOAD MODEL & SCALER =================
MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "scaler.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ================= CONFIG =================
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

# ================= FEATURE EXTRACTION =================
def extract_features_from_file(file_bytes):
    # Load audio from bytes
    audio, sr = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, duration=DURATION)

    # Pad/truncate
    expected_len = SAMPLE_RATE * DURATION
    if len(audio) < expected_len:
        audio = np.pad(audio, (0, expected_len - len(audio)))
    else:
        audio = audio[:expected_len]

    # ===== MFCC =====
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    # ===== Chroma =====
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = np.std(chroma, axis=1)

    # ===== Spectral Contrast =====
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std  = np.std(contrast, axis=1)

    # Combine all features
    features = np.hstack([mfcc_mean, mfcc_std, chroma_mean, chroma_std, contrast_mean, contrast_std])
    return features
# ===== Route for home page =====
@app.route('/')
def home():
    return render_template('index.html') 

# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    audio_bytes = file.read()

    try:
        # Extract features
        features = extract_features_from_file(audio_bytes)

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba) * 100)
        label = "human" if pred == 0 else "ai"

        return jsonify({"label": label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= RUN APP =================
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")
if __name__ == '__main__':
    # This ensures the browser opens only in the actual server, not the reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1, open_browser).start()

    app.run(debug=True, host='0.0.0.0', port=5000)

