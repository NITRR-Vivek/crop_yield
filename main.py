from fastapi import FastAPI
import numpy as np
import joblib
 
loaded_model = joblib.load('linear_regression_model.h5')
 
app = FastAPI()
 
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Define the prediction endpoint
@app.post("/predict/")
async def predict(features: list): 
    custom_sample = np.array([features], dtype=np.float32)
    predictions = loaded_model.predict(custom_sample)
    predicted_yield = predictions[0]
    return {"predicted_yield": f"{predicted_yield:.2f}"}
