from final_assignment_bank_campaign.ml_api.fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("model.pkl")

app = FastAPI()

# Define request schema
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
