from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    weight_kg: float = Field(..., ge=0.5, le=50)
    distance_km: float = Field(..., ge=5, le=500)
    is_express: int = Field(..., ge=0, le=1)
    warehouse_load: int = Field(..., ge=1, le=5)

@app.get("/health")
def health():
    return {"status": "operational", "service": "SpeedCargo API"}

@app.post("/estimate")
def estimate(data: InputData):
    features = np.array([[data.weight_kg, data.distance_km, data.is_express, data.warehouse_load]])
    pred = model.predict(features)[0]
    return {"prediction": float(pred)}