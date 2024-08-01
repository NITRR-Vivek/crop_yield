from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI()  # Make sure this is named 'app'

# Load the ONNX model
ort_session = ort.InferenceSession('linear_regression_model.onnx')

class PredictionInput(BaseModel):
    soil: float
    seed: float
    fertilizer: float
    sunny: float
    rainfall: float
    irrigation: float

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        custom_sample = np.array([[data.soil, data.seed, data.fertilizer, data.sunny, data.rainfall, data.irrigation]], dtype=np.float32)
        inputs = {ort_session.get_inputs()[0].name: custom_sample}
        predictions = ort_session.run(None, inputs)
        predicted_yield = predictions[0].flatten()[0]
        return {"predicted_yield": predicted_yield}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
