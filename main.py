from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    result = model.predict(data)
    return {"prediction": int(result[0])}