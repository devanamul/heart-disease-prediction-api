
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from app.schemas import HeartDiseaseInput


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML API using FastAPI and Scikit-learn",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/info")
def info():
    return {
        "model_type": "Logistic Regression",
        "features": FEATURES,
        "target": "Heart Disease",
        "classes": {
            "0": "No Heart Disease",
            "1": "Heart Disease",
        },
    }


@app.post("/predict")
def predict(data: HeartDiseaseInput):
    try:
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURES,
        )

        prediction = int(model.predict(input_data)[0])

        probability = model.predict_proba(input_data)[0][1]

        return {
            "heart_disease": bool(prediction),
            "prediction": prediction,
            "probability": round(float(probability), 4),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
