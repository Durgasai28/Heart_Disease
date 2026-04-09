"""
CardioScan – Heart Disease Prediction API
Two modes: Basic (6 self-reportable features) & Advanced (8 features incl. MaxHR, ST_Slope, Oldpeak)

Improvements over v1:
  - joblib model persistence (train once, load on restart)
  - Pydantic Field validators with realistic medical ranges
  - /health endpoint
  - Structured logging
  - Graceful startup error handling
  - Removed stale commented-out code

Run with: uvicorn main:app --reload
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Logging ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cardioscan")

# ─── Paths ───────────────────────────────────────────

DATA_PATH          = "heart.csv"
BASIC_MODEL_PATH   = "basic_model.joblib"
ADVANCED_MODEL_PATH= "advanced_model.joblib"

# ─── Feature lists ───────────────────────────────────

BASIC_FEATURES    = ["Sex", "Age", "ChestPainType", "FastingBS", "ExerciseAngina", "RestingBP"]
ADVANCED_FEATURES = ["Sex", "Age", "ChestPainType", "FastingBS", "ExerciseAngina",
                     "RestingBP", "MaxHR", "ST_Slope", "Cholesterol", "Oldpeak"]

# ─── Shared encoding ─────────────────────────────────

def load_and_encode() -> pd.DataFrame:
    """Load heart.csv and encode categorical columns to integer codes."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Please place heart.csv in the same directory as main.py."
        )
    df = pd.read_csv(DATA_PATH)
    df["Sex"]            = df["Sex"].map({"M": 0, "F": 1})
    mapping = {"ASY": 0,"ATA": 1,"NAP": 2,"TA": 3}
    df["ChestPainType"] = df["ChestPainType"].map(mapping)
    df["ExerciseAngina"] = df["ExerciseAngina"].astype("category").cat.codes
    df["ST_Slope"]       = df["ST_Slope"].astype("category").cat.codes
    return df

# ─── Generic trainer + persister ─────────────────────

def get_model(features: list, save_path: str, label: str):
    """
    Load a saved (model, scaler) pair from disk if it exists,
    otherwise train from scratch and save it with joblib.
    """
    if os.path.exists(save_path):
        log.info(f"[{label}] Loading saved model from '{save_path}'")
        model, scaler = joblib.load(save_path)
        return model, scaler

    log.info(f"[{label}] No saved model found – training from '{DATA_PATH}'")
    df = load_and_encode()
    X  = df[features]
    y  = df["HeartDisease"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    model  = LogisticRegression(max_iter=1000)
    model.fit(scaler.fit_transform(X_tr), y_tr)

    accuracy = model.score(scaler.transform(X_te), y_te)
    log.info(f"[{label}] accuracy={accuracy:.3f}  features={features}")

    joblib.dump((model, scaler), save_path)
    log.info(f"[{label}] Model saved to '{save_path}'")
    return model, scaler

# ─── Load / train models at startup ──────────────────

try:
    basic_model,    basic_scaler    = get_model(BASIC_FEATURES,    BASIC_MODEL_PATH,    "Basic   ")
    advanced_model, advanced_scaler = get_model(ADVANCED_FEATURES, ADVANCED_MODEL_PATH, "Advanced")
except FileNotFoundError as exc:
    log.critical(str(exc))
    raise SystemExit(1) from exc

# ─── Shared prediction helper ─────────────────────────

def make_response(model, scaler, data: list) -> dict:
    arr          = scaler.transform([data])
    pred         = int(model.predict(arr)[0])
    proba        = float(model.predict_proba(arr)[0][1])
    prob_percent = round(proba * 100, 2)

    if prob_percent >= 70:
        risk_level = "🔴 High Risk"
    elif prob_percent >= 40:
        risk_level = "🟠 Moderate Risk"
    else:
        risk_level = "🟢 Low Risk"

    label = (
        f"⚠ Heart Disease Likely ({risk_level})"
        if pred == 1
        else f"✅ Low Chance of Heart Disease ({risk_level})"
    )

    return {
        "prediction":  pred,
        "probability": prob_percent,
        "risk_level":  risk_level,
        "label":       label,
    }

# ─── App ─────────────────────────────────────────────

app = FastAPI(
    title="CardioScan – Dual-Mode Heart Disease Predictor",
    description="Predict heart-disease risk using a Basic (6 features) or Advanced (8 features) logistic-regression model.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── GET /health ──────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    """Quick liveness probe – returns 200 if the API is running."""
    return {"status": "ok", "models_loaded": True}

# ─── GET /features ───────────────────────────────────

@app.get("/features", tags=["Utility"])
def get_features():
    """Return the feature lists used by each model."""
    return {"basic": BASIC_FEATURES, "advanced": ADVANCED_FEATURES}

# ─── POST /predict/basic ─────────────────────────────

class BasicInput(BaseModel):
    """
    Six self-reportable features that don't require a clinical test.
    """
    Sex:            int   = Field(..., ge=0, le=1,   description="0 = Male, 1 = Female")
    Age:            int   = Field(..., ge=1, le=120,  description="Age in years")
    ChestPainType:  int   = Field(..., ge=0, le=3,   description="0=ASY, 1=ATA, 2=NAP, 3=TA")
    FastingBS:      int   = Field(..., ge=0, le=1,   description="Fasting blood sugar: 0 = ≤120 mg/dL, 1 = >120 mg/dL")
    ExerciseAngina: int   = Field(..., ge=0, le=1,   description="0 = No, 1 = Yes")
    RestingBP:      int   = Field(..., ge=60, le=250, description="Resting blood pressure in mm Hg")


@app.post("/predict/basic", tags=["Prediction"])
def predict_basic(p: BasicInput):
    """Predict heart-disease risk from 6 self-reportable features."""
    try:
        return make_response(
            basic_model, basic_scaler,
            [p.Sex, p.Age, p.ChestPainType, p.FastingBS, p.ExerciseAngina, p.RestingBP],
        )
    except Exception as exc:
        log.error(f"Basic prediction failed: {exc}")
        raise HTTPException(status_code=500, detail="Prediction error. Please check your inputs.")

# ─── POST /predict/advanced ──────────────────────────

class AdvancedInput(BaseModel):
    """
    Eight features including clinical measurements from an ECG / stress test.
    """
    Sex:            int   = Field(..., ge=0, le=1,    description="0 = Male, 1 = Female")
    Age:            int   = Field(..., ge=1, le=120,  description="Age in years")
    ChestPainType:  int   = Field(..., ge=0, le=3,    description="0=ASY, 1=ATA, 2=NAP, 3=TA")
    FastingBS:      int   = Field(..., ge=0, le=1,    description="Fasting blood sugar: 0 = ≤120 mg/dL, 1 = >120 mg/dL")
    ExerciseAngina: int   = Field(..., ge=0, le=1,    description="0 = No, 1 = Yes")
    RestingBP:      int   = Field(..., ge=60, le=250,  description="Resting blood pressure in mm Hg")
    MaxHR:          int   = Field(..., ge=60, le=220,  description="Maximum heart rate achieved (bpm)")
    ST_Slope:       int   = Field(..., ge=0, le=2,    description="ST slope: 0=Down, 1=Flat, 2=Up")
    Cholesterol:   int   = Field(..., ge=0, le=600,    description="Cholesterol 0-600")
    Oldpeak:        float = Field(..., ge=-2.6, le=6.2, description="ST depression induced by exercise")


@app.post("/predict/advanced", tags=["Prediction"])
def predict_advanced(p: AdvancedInput):
    """Predict heart-disease risk from 8 clinical features."""
    try:
        return make_response(
            advanced_model, advanced_scaler,
            [p.Sex, p.Age, p.ChestPainType, p.FastingBS, p.ExerciseAngina,
             p.RestingBP, p.MaxHR, p.ST_Slope, p.Cholesterol, p.Oldpeak],
        )
    except Exception as exc:
        log.error(f"Advanced prediction failed: {exc}")
        raise HTTPException(status_code=500, detail="Prediction error. Please check your inputs.")
