from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agri Forecast API (L0)")

# Load once at startup
MODEL = joblib.load("models/model.pkl")

FEATURES = [
    "lag_1","lag_2","lag_3","lag_5","lag_10",
    "roll_ret_mean_5","roll_ret_mean_10","roll_ret_mean_20",
    "roll_ret_std_5","roll_ret_std_10","roll_ret_std_20",
]

# Allow local Dash (http://127.0.0.1:8050) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8050", "http://localhost:8050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    # Provide latest feature vector (e.g., computed offline)
    features: dict

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict with feature dict."}

@app.post("/predict")
def predict(req: PredictRequest):
    x = [[req.features.get(k, 0.0) for k in FEATURES]]
    yhat = MODEL.predict(x)[0]
    return {"prediction_next_day_close": float(yhat)}
