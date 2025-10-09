from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import joblib
import numpy as np
import pandas as pd

# Load trained logistic regression model pipeline
model = joblib.load("model_1mvp.pkl")

app = FastAPI(title="Logistic Regression API")

# Schema for input data (matches features used in training)
class InputData(BaseModel):
    age: int
    balance: float
    day: int
    campaign: int
    job: str
    education: str
    default: Literal["yes", "no", "unknown"]
    housing: Literal["yes", "no", "unknown"]
    loan: Literal["yes", "no", "unknown"]
    months_since_previous_contact: str
    n_previous_contacts: str
    poutcome: str
    had_contact: bool
    is_single: bool
    uknown_contact: bool

class BatchInputData(BaseModel):
    data: List[InputData]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(batch: BatchInputData):
    try:
        X = pd.DataFrame([item.dict() for item in batch.data])
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
