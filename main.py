from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
 
loaded_model = joblib.load('linear_regression_model.h5')
 
app = FastAPI()
 
class Features(BaseModel):
    Soil_Quality: float
    Seed_Variety: float
    Fertilizer_Amount_kg_per_hectare: float
    Sunny_Days: float
    Rainfall_mm: float
    Irrigation_Schedule: float
 
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Define the prediction endpoint
@app.post("/predict/")
async def predict(data: Features): 
    custom_sample = np.array([[
        data.Soil_Quality,
        data.Seed_Variety,
        data.Fertilizer_Amount_kg_per_hectare,
        data.Sunny_Days,
        data.Rainfall_mm,
        data.Irrigation_Schedule
    ]], dtype=np.float32)
     
    predictions = loaded_model.predict(custom_sample)
    predicted_yield = predictions[0]
    
    return {"predicted_yield": f"{predicted_yield:.2f}"}
