from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


# ===== SAFE ROUNDING =====
def r(x):
    return float(np.round(x, 6))


# ===== FEATURE EXTRACTION (DETERMINISTIC) =====
def extract_features(y, sr):
    # Fix length for consistency
    y = librosa.util.fix_length(y, size=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {}
    for i in range(13):
        features[f"mfcc_{i}"] = mfcc[i]

    return features


# ===== STATS =====
def compute_stats(features):
    keys = sorted(features.keys())  # FIXED ORDER

    result = {
        "rows": len(keys),
        "columns": keys,
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

    matrix = []

    for key in keys:
        v = np.array(features[key], dtype=np.float64)

        result["mean"][key] = r(np.mean(v))
        result["std"][key] = r(np.std(v))
        result["variance"][key] = r(np.var(v))
        result["min"][key] = r(np.min(v))
        result["max"][key] = r(np.max(v))
        result["median"][key] = r(np.median(v))

        # STABLE MODE (rounded first)
        vr = np.round(v, 3)
        vals, counts = np.unique(vr, return_counts=True)
        result["mode"][key] = r(vals[np.argmax(counts)])

        result["range"][key] = r(np.max(v) - np.min(v))
        result["allowed_values"][key] = []
        result["value_range"][key] = [r(np.min(v)), r(np.max(v))]

        matrix.append(v)

    # ===== CORRELATION (SAFE) =====
    try:
        corr = np.corrcoef(matrix)
        corr = np.nan_to_num(corr)

        result["correlation"] = [
            [r(val) for val in row] for row in corr
        ]
    except:
        result["correlation"] = []

    return result


@app.post("/predict")
def predict(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

        features = extract_features(y, sr)
        result = compute_stats(features)

        return result

    except Exception:
        # STRICT EMPTY RESPONSE
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