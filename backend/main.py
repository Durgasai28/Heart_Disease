"""
CardioScan-Heart Disease Prediction API
Two modes:
  - Basic: 12 self-reportable lifestyle and vital features
  - Advanced: 20 features including comprehensive clinical & laboratory markers

Model persistence: joblib bundle (trained model, standard scaler, feature list, calibration bounds)
FastAPI endpoints:
  - GET  /                  (Web Interface)
  - GET  /health            (Liveness & model status)
  - GET  /features          (Lists of features per mode)
  - POST /predict/basic     (Prediction using 12 basic lifestyle features)
  - POST /predict/advanced  (Prediction using 20 complete clinical features)

Run with: uvicorn main:app --reload
"""

import os
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Logging ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cardioscan")

# ─── Directories & Paths ──────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

def find_file(filename: str, *search_dirs: Path) -> Path:
    """Search for a file across given directories, returning the first existing match or fallback."""
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return search_dirs[0] / filename

DATA_PATH           = str(find_file("heart.csv", BACKEND_DIR, PROJECT_DIR))
BASIC_MODEL_PATH    = str(find_file("basic_model.joblib", BACKEND_DIR, PROJECT_DIR))
ADVANCED_MODEL_PATH = str(find_file("advanced_model.joblib", BACKEND_DIR, PROJECT_DIR))

INDEX_HTML = find_file("index.html", FRONTEND_DIR, PROJECT_DIR, BACKEND_DIR)
STYLESHEET = find_file("stylesheet_ids.css", FRONTEND_DIR, PROJECT_DIR, BACKEND_DIR)
HEART_BG   = find_file("heartbg.jpg", FRONTEND_DIR, PROJECT_DIR, BACKEND_DIR)

# ─── Feature lists ───────────────────────────────────

BASIC_FEATURES = ["Age","Gender","BloodPressure","BMI","ExerciseHabits","Smoking",
            "FamilyHeartDisease","Diabetes","AlcoholConsumption","StressLevel","SleepHours","SugarConsumption",
    ]

ADVANCED_FEATURES = BASIC_FEATURES + ["CholesterolLevel","TriglycerideLevel","FastingBloodSugar","CRPLevel",
    "HomocysteineLevel","HighBloodPressure","LowHDLCholesterol","HighLDLCholesterol",
    ]

# ─── Dataset Column Mapping ──────────────────────────

COLUMN_RENAME_MAP = {
    "Blood Pressure": "BloodPressure",
    "Cholesterol Level": "CholesterolLevel",
    "Exercise Habits": "ExerciseHabits",
    "Family Heart Disease": "FamilyHeartDisease",
    "High Blood Pressure": "HighBloodPressure",
    "Low HDL Cholesterol": "LowHDLCholesterol",
    "High LDL Cholesterol": "HighLDLCholesterol",
    "Alcohol Consumption": "AlcoholConsumption",
    "Stress Level": "StressLevel",
    "Sleep Hours": "SleepHours",
    "Sugar Consumption": "SugarConsumption",
    "Triglyceride Level": "TriglycerideLevel",
    "Fasting Blood Sugar": "FastingBloodSugar",
    "CRP Level": "CRPLevel",
    "Homocysteine Level": "HomocysteineLevel",
    "Heart Disease Status": "HeartDisease",
}

# ─── Shared encoding ─────────────────────────────────

def load_and_encode() -> pd.DataFrame:
    """Load heart.csv, clean missing values, and encode categoricals to integers."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Please ensure heart.csv exists in the backend directory."
        )

    # Note: keep_default_na=False and na_values=[''] so 'None' in Alcohol Consumption is kept as string
    df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[""])
    df = df.rename(columns=COLUMN_RENAME_MAP)

    categorical_columns = [
        "Gender", "Smoking", "FamilyHeartDisease", "Diabetes", "HighBloodPressure",
        "LowHDLCholesterol", "HighLDLCholesterol", "ExerciseHabits", "StressLevel",
        "SugarConsumption", "AlcoholConsumption", "HeartDisease"
    ]

    # Impute missing values
    for col in df.columns:
        if col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Map categorical values
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)

    yes_no_cols = [
        "Smoking", "FamilyHeartDisease", "Diabetes",
        "HighBloodPressure", "LowHDLCholesterol", "HighLDLCholesterol"
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    df["ExerciseHabits"] = df["ExerciseHabits"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(1).astype(int)
    df["StressLevel"] = df["StressLevel"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(1).astype(int)
    df["SugarConsumption"] = df["SugarConsumption"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(1).astype(int)
    df["AlcoholConsumption"] = df["AlcoholConsumption"].map({"None": 0, "Low": 1, "Medium": 2, "High": 3}).fillna(0).astype(int)

    if "HeartDisease" in df.columns:
        df["HeartDisease"] = df["HeartDisease"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    return df

# ─── Generic trainer + persister ─────────────────────

def get_model(features: list, save_path: str, label: str) -> dict:
    """
    Load a saved model bundle (model, scaler, calibration stats) from disk if compatible,
    otherwise train from scratch and save with joblib.
    """
    if os.path.exists(save_path):
        try:
            bundle = joblib.load(save_path)
            if isinstance(bundle, dict) and bundle.get("features") == features:
                log.info(f"[{label}] Loading valid saved model bundle from '{save_path}'")
                return bundle
        except Exception as err:
            log.warning(f"[{label}] Failed to load existing model ({err}). Retraining...")

    log.info(f"[{label}] Training fresh model from '{DATA_PATH}'")
    df = load_and_encode()
    X  = df[features]
    y  = df["HeartDisease"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_tr_s, y_tr)

    accuracy = float(model.score(X_te_s, y_te))
    all_probs = model.predict_proba(scaler.transform(X))[:, 1]
    min_p = float(all_probs.min())
    max_p = float(all_probs.max())

    log.info(f"[{label}] accuracy={accuracy:.3f}  min_p={min_p:.3f}  max_p={max_p:.3f}  features={len(features)}")

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "min_p": min_p,
        "max_p": max_p,
        "accuracy": accuracy,
    }

    joblib.dump(bundle, save_path, compress=3)
    log.info(f"[{label}] Model bundle saved to '{save_path}' (compressed)")
    return bundle

# ─── Load / train models at startup ──────────────────

try:
    basic_bundle    = get_model(BASIC_FEATURES,    BASIC_MODEL_PATH,    "Basic   ")
    advanced_bundle = get_model(ADVANCED_FEATURES, ADVANCED_MODEL_PATH, "Advanced")
except FileNotFoundError as exc:
    log.critical(str(exc))
    raise SystemExit(1) from exc

# ─── Shared prediction helper ─────────────────────────

def make_response(bundle: dict, data: list) -> dict:
    model  = bundle["model"]
    scaler = bundle["scaler"]
    min_p  = bundle.get("min_p", 0.40)
    max_p  = bundle.get("max_p", 0.60)

    # Use DataFrame with feature names to match StandardScaler fitting
    df_input = pd.DataFrame([data], columns=bundle["features"])
    arr = scaler.transform(df_input)
    raw_pred = int(model.predict(arr)[0])
    raw_prob = float(model.predict_proba(arr)[0][1])

    # Calibrate probability percentage across relative population risk distribution
    spread = max_p - min_p if max_p > min_p else 1.0
    norm_score = (raw_prob - min_p) / spread
    prob_percent = round(float(np.clip(norm_score * 90.0 + 5.0, 3.0, 97.0)), 1)

    if prob_percent >= 65.0:
        risk_level = "🔴 High Risk"
        pred = 1
    elif prob_percent >= 40.0:
        risk_level = "🟠 Moderate Risk"
        pred = 1 if (raw_pred == 1 or prob_percent >= 50.0) else 0
    else:
        risk_level = "🟢 Low Risk"
        pred = 0

    label = (f"⚠ Heart Disease Risk Detected ({risk_level})" if pred == 1 else f"✅ Low Chance of Heart Disease ({risk_level})"
    )

    return {
        "prediction":       pred,
        "probability":      prob_percent,
        "raw_probability":  round(raw_prob * 100, 2),
        "risk_level":       risk_level,
        "label":            label,
    }

# ─── App ─────────────────────────────────────────────

app = FastAPI(
    title="CardioScan – Heart Disease Predictor",
    description="Dual-mode clinical cardiovascular risk assessment using Basic (12 features) or Advanced (20 features) Logistic Regression.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static / Web UI routes ──────────────────────────

@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
@app.get("/frontend/index.html", include_in_schema=False)
def read_root():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return {"message": "CardioScan API is online. Please visit /docs for API documentation."}

@app.get("/stylesheet_ids.css", include_in_schema=False)
@app.get("/frontend/stylesheet_ids.css", include_in_schema=False)
def get_stylesheet():
    if STYLESHEET.exists():
        return FileResponse(str(STYLESHEET), media_type="text/css")
    raise HTTPException(status_code=404, detail="Stylesheet not found")

@app.get("/heartbg.jpg", include_in_schema=False)
@app.get("/frontend/heartbg.jpg", include_in_schema=False)
def get_background():
    if HEART_BG.exists():
        return FileResponse(str(HEART_BG), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Background image not found")

# ─── GET /health ──────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    return {
        "status": "ok",
        "models_loaded": True,
        "basic_accuracy": round(basic_bundle.get("accuracy", 0.0), 3),
        "advanced_accuracy": round(advanced_bundle.get("accuracy", 0.0), 3),
        "features_count": {
            "basic": len(BASIC_FEATURES),
            "advanced": len(ADVANCED_FEATURES),
        },
    }

# ─── GET /features ───────────────────────────────────

@app.get("/features", tags=["Utility"])
def get_features():
    """Return the feature lists used by each model mode."""
    return {"basic": BASIC_FEATURES, "advanced": ADVANCED_FEATURES}

# ─── POST /predict/basic ─────────────────────────────

class BasicInput(BaseModel):
    #12 self-reportable lifestyle and physiological features.

    Age:                float = Field(..., ge=18.0, le=100.0, description="Age in years (18–100)")
    Gender:             int   = Field(..., ge=0,    le=1,     description="0 = Male, 1 = Female")
    BloodPressure:      float = Field(..., ge=80.0, le=240.0, description="Systolic Blood Pressure (mm Hg, 120–180)")
    BMI:                float = Field(..., ge=10.0, le=60.0,  description="Body Mass Index (kg/m², 18–40)")
    ExerciseHabits:     int   = Field(..., ge=0,    le=2,     description="Exercise level: 0 = Low, 1 = Medium, 2 = High")
    Smoking:            int   = Field(..., ge=0,    le=1,     description="Smoking status: 0 = No, 1 = Yes")
    FamilyHeartDisease: int   = Field(..., ge=0,    le=1,     description="Family history of heart disease: 0 = No, 1 = Yes")
    Diabetes:           int   = Field(..., ge=0,    le=1,     description="Diabetes status: 0 = No, 1 = Yes")
    AlcoholConsumption: int   = Field(..., ge=0,    le=3,     description="Alcohol intake: 0 = None, 1 = Low, 2 = Medium, 3 = High")
    StressLevel:        int   = Field(..., ge=0,    le=2,     description="Stress level: 0 = Low, 1 = Medium, 2 = High")
    SleepHours:         float = Field(..., ge=2.0,  le=16.0,  description="Average nightly sleep in hours (4–10)")
    SugarConsumption:   int   = Field(..., ge=0,    le=2,     description="Sugar consumption: 0 = Low, 1 = Medium, 2 = High")


@app.post("/predict/basic", tags=["Prediction"])
def predict_basic(p: BasicInput):
    """Predict cardiovascular risk from 12 self-reportable features."""
    try:
        data = [p.Age,p.Gender,p.BloodPressure,p.BMI,p.ExerciseHabits,p.Smoking,
            p.FamilyHeartDisease,p.Diabetes,p.AlcoholConsumption,p.StressLevel,p.SleepHours,p.SugarConsumption,
        ]
        return make_response(basic_bundle, data)
    except Exception as exc:
        log.error(f"Basic prediction failed: {exc}")
        raise HTTPException(status_code=500, detail="Prediction error. Please check your inputs.")

# ─── POST /predict/advanced ──────────────────────────

class AdvancedInput(BasicInput):
    #20 features: All 12 Basic features + 8 clinical & laboratory findings.

    CholesterolLevel:   float = Field(..., ge=100.0, le=500.0, description="Total Cholesterol in mg/dL (150–300)")
    TriglycerideLevel:  float = Field(..., ge=50.0,  le=600.0, description="Triglyceride Level in mg/dL (100–400)")
    FastingBloodSugar:  float = Field(..., ge=50.0,  le=300.0, description="Fasting Blood Sugar in mg/dL (80–160)")
    CRPLevel:           float = Field(..., ge=0.0,   le=30.0,  description="C-Reactive Protein in mg/L (0.0–15.0)")
    HomocysteineLevel:  float = Field(..., ge=1.0,   le=50.0,  description="Homocysteine in µmol/L (5.0–20.0)")
    HighBloodPressure:  int   = Field(..., ge=0,     le=1,     description="Diagnosed Hypertension: 0 = No, 1 = Yes")
    LowHDLCholesterol:  int   = Field(..., ge=0,     le=1,     description="Diagnosed Low HDL: 0 = No, 1 = Yes")
    HighLDLCholesterol: int   = Field(..., ge=0,     le=1,     description="Diagnosed High LDL: 0 = No, 1 = Yes")


@app.post("/predict/advanced", tags=["Prediction"])
def predict_advanced(p: AdvancedInput):
    #Predict cardiovascular risk from complete 20-feature clinical & lab profile.
    try:
        data = [p.Age,p.Gender,p.BloodPressure,p.BMI,p.ExerciseHabits,p.Smoking,p.FamilyHeartDisease,p.Diabetes,
            p.AlcoholConsumption,p.StressLevel,p.SleepHours,p.SugarConsumption,p.CholesterolLevel,p.TriglycerideLevel,
            p.FastingBloodSugar,p.CRPLevel,p.HomocysteineLevel,p.HighBloodPressure,p.LowHDLCholesterol,p.HighLDLCholesterol,
        ]
        return make_response(advanced_bundle, data)
    except Exception as exc:
        log.error(f"Advanced prediction failed: {exc}")
        raise HTTPException(status_code=500, detail="Prediction error. Please check your inputs.")