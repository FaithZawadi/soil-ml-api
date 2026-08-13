from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from app.model import load_model, predict_and_rank

app = FastAPI(title="Soil ML Crop Recommendation")

# Input schema
class SoilConditions(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float

# Load model at startup
model = load_model()
crop_labels = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee"
]

@app.get("/")
async def root():
    return {"message": "Soil ML Crop Recommendation API"}

@app.post("/predict")
async def predict_crop(conditions: SoilConditions):
    try:
        # Feature generation: Convert input to numpy array
        input_data = np.array([[
            conditions.N, conditions.P, conditions.K,
            conditions.temperature, conditions.humidity, conditions.ph
        ]])

        # Inference
        scaled_probs, ranked_crops = predict_and_rank(model, input_data, crop_labels)

        # Process output: Top 5 crops
        rankings = [
            {"rank": rank + 1, "crop": crop_labels[crop_idx], "score": float(scaled_probs[0][crop_idx])}
            for rank, crop_idx in enumerate(ranked_crops[:5])
        ]
        return {"ranked_crops": rankings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")