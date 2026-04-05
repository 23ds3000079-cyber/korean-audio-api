from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io

app = FastAPI()

# ===== Request Schema =====
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


# ===== Feature Extraction =====
def extract_features(y, sr):
    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i}"] = mfcc[i]

    return features


# ===== Statistics Function =====
def compute_stats(features):
    result = {
        "rows": 0,
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": []
    }

    for key, values in features.items():
        values = np.array(values)

        result["columns"].append(key)
        result["mean"][key] = float(np.mean(values))
        result["std"][key] = float(np.std(values))
        result["variance"][key] = float(np.var(values))
        result["min"][key] = float(np.min(values))
        result["max"][key] = float(np.max(values))
        result["median"][key] = float(np.median(values))

        # Mode (safe)
        try:
            vals, counts = np.unique(values, return_counts=True)
            result["mode"][key] = float(vals[np.argmax(counts)])
        except:
            result["mode"][key] = 0.0

        result["range"][key] = float(np.max(values) - np.min(values))

        # Extra required keys
        result["allowed_values"][key] = []
        result["value_range"][key] = [float(np.min(values)), float(np.max(values))]

    result["rows"] = len(features)

    return result


# ===== API Endpoint =====
@app.post("/predict")
def predict(req: AudioRequest):
    try:
        # Decode base64
        audio_bytes = base64.b64decode(req.audio_base64)

        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # Extract features
        features = extract_features(y, sr)

        # Compute stats
        result = compute_stats(features)

        return result

    except Exception as e:
        return {
            "rows": 0,
            "columns": [],
            "mean": {},
            "std": {},
            "variance": {},
            "min": {},
            "max": {},
            "median": {},
            "mode": {},
            "range": {},
            "allowed_values": {},
            "value_range": {},
            "correlation": []
        }