from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import xgboost as xgb
from typing import List, Dict
from .prediction_pipeline import OutageLSTM, predict_outages_ensemble, preprocess_data, load_data_sources

app = FastAPI(title="Noc-netIntel API", description="AI-Powered Network Operations Intelligence Assistant")

# Pydantic model for chat input
class ChatRequest(BaseModel):
    prompt: str
    region: str = None
    horizon: str = "day"

# Pydantic model for forecast output
class ForecastResponse(BaseModel):
    site_code: str
    prediction_time: str
    outage_probability: float
    predicted_outage_type: str

# Placeholder for LLM (to be implemented)
def llm_generate_response(predictions: pd.DataFrame, prompt: str) -> str:
    # TODO: Integrate DeepSeek or similar LLM
    return f"Predicted outages:\n{predictions.to_string()}"

# Load models (assume pre-trained)
lstm_model = None  # Load OutageLSTM model
xgb_model = None   # Load XGBoost model
scaler = None
le = None
SEQ_LENGTH = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_models():
    global lstm_model, xgb_model, scaler, le
    # Placeholder: Load pre-trained models and preprocessing objects
    data_sources = load_data_sources()
    processed_data, scaler, le = preprocess_data(data_sources)
    lstm_model = OutageLSTM(input_size=7, hidden_size=64, num_layers=2, num_classes=len(le.classes_)).to(DEVICE)
    lstm_model.load_state_dict(torch.load("lstm_model.pth"))
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_model.json")

@app.post("/chat", response_model=Dict)
async def chat(request: ChatRequest):
    """
    Handle chat queries and return AI-generated insights.
    """
    data_sources = load_data_sources()
    processed_data, _, _ = preprocess_data(data_sources)
    
    predictions = predict_outages_ensemble(
        lstm_model, xgb_model, processed_data, scaler, le, SEQ_LENGTH, DEVICE, request.horizon
    )
    
    if request.region:
        predictions = predictions[predictions["site_code"].str.contains(request.region)]
    
    if predictions.empty:
        raise HTTPException(status_code=404, detail="No predictions found for the specified criteria")
    
    response_text = llm_generate_response(predictions, request.prompt)
    return {"response": response_text, "predictions": predictions.to_dict(orient="records")}

@app.get("/forecast", response_model=List[ForecastResponse])
async def get_forecast(horizon: str = "day"):
    """
    Return raw outage predictions.
    """
    data_sources = load_data_sources()
    processed_data, _, _ = preprocess_data(data_sources)
    
    predictions = predict_outages_ensemble(
        lstm_model, xgb_model, processed_data, scaler, le, SEQ_LENGTH, DEVICE, horizon
    )
    
    return predictions.to_dict(orient="records")

# TODO: Implement /schedule and /logs endpoints