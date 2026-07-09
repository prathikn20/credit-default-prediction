import joblib
import pandas as pd
from fastapi import FastAPI
from src.schema import CreditApplication

app = FastAPI()
model = joblib.load("models/credit_pipeline.joblib")

@app.post("/predict")
def predict(application: CreditApplication):

    data = application.model_dump()

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]
    prediction = ""
    if prob >= 0.4:
        prediction = "default"
    else:
        prediction = "Not Default"

    return {"prediction": prediction, "probability": float(prob)}
