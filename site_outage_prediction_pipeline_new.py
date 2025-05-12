import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import aiohttp
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from transformers import DistilBertTokenizer, DistilBertModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

# FastAPI App
app = FastAPI(title="Noc-netIntel API", description="AI-Powered Network Operations Intelligence Assistant")

# Pydantic Models
class ChatRequest(BaseModel):
    prompt: str
    region: str | None = None
    horizon: str = "day"

class ChatResponse(BaseModel):
    response: str
    predictions: list[dict]

class ForecastResponse(BaseModel):
    site_code: str
    region: str
    prediction_time: str
    outage_probability: float
    predicted_outage_type: str
    predicted_outage_duration: float
    rca: str
    explanation: str

# Global Models and Objects
lstm_model = None
xgb_model_class = None
xgb_model_reg = None
scaler = None
le = None
rca_embeddings = None
# SEQ_LENGTH = 24
SEQ_LENGTH = 2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HF_API_KEY = os.getenv("HF_API_KEY", "your-huggingface-api-key")  # Replace with DeepSeek API key if available

# 1. Data Source Outline and Loading
def load_data_sources():
    try:
        data_sources = {
            "alarms": pd.read_csv("data/alarms.csv"),
            "alarm_classifications": pd.read_csv("data/alarm_classifications.csv"),
            "site_details": pd.read_csv("data/site_details.csv"),
            "site_data_availability": pd.read_csv("data/site_data_availability.csv"),
            "ticket_data": pd.read_csv("data/ticket_data.csv")
        }
        # Debug: Print columns and shapes
        # for key, df in data_sources.items():
            # print(f"{key}: {df.shape}, Columns: {df.columns.tolist()}")
        return data_sources
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    # return data_sources

# 2. Data Preprocessing


def preprocess_data(data_sources):
    # Standardize alarms
    alarms = data_sources["alarms"].copy()
    # Rename columns
    alarms.rename(columns={
        'site_id': 'site_code',
        'first_occurred_on': 'timestamp'
    }, inplace=True)
    if 'timestamp' not in alarms.columns:
        raise KeyError("alarms DataFrame is missing 'timestamp' column")

    # Convert timestamp to datetime, coerce errors
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"], errors="coerce")
    invalid_timestamp_count = alarms["timestamp"].isna().sum()
    
    # Filter out rows with invalid timestamps
    if invalid_timestamp_count > 0:
        # print(f"Filtering out {invalid_timestamp_count} rows with invalid timestamps in alarms")
        invalid_rows = alarms[alarms["timestamp"].isna()][["site_code", "timestamp", "alarm_name"]]
        # print("Sample invalid rows in alarms:", invalid_rows.head().to_string())
        alarms = alarms[alarms["timestamp"].notna()]

    # Create merge_timestamp for date-only merging
    alarms["merge_timestamp"] = alarms["timestamp"].dt.date

    # Ensure alarm_name is string
    alarms["alarm_name"] = alarms["alarm_name"].astype(str)

    # Standardize alarm_classifications
    alarm_classifications = data_sources["alarm_classifications"].copy()
    alarm_classifications.rename(columns={'event_category': 'classification'}, inplace=True)
    alarm_classifications["alarm_name"] = alarm_classifications["alarm_name"].astype(str)

    # Standardize site_details
    site_details = data_sources["site_details"].copy()
    site_details.rename(columns={
        'site_id': 'site_code',
        'region_name': 'region'
    }, inplace=True)

    # Standardize site_data_availability
    site_data_availability = data_sources["site_data_availability"].copy()
    site_data_availability.rename(columns={
        'site_id': 'site_code',
        'availability': 'availability_percentage'
    }, inplace=True)

    # Debug: Inspect site_data_availability timestamps
    site_data_availability["timestamp"] = pd.to_datetime(site_data_availability["timestamp"], errors="coerce")
    invalid_timestamp_count = site_data_availability["timestamp"].isna().sum()
    
    # Filter out rows with invalid timestamps
    if invalid_timestamp_count > 0:
        # print(f"Filtering out {invalid_timestamp_count} rows with invalid timestamps in site_data_availability")
        invalid_rows = site_data_availability[site_data_availability["timestamp"].isna()][["site_code", "timestamp"]]
        # print("Sample invalid rows in site_data_availability:", invalid_rows.head().to_string())
        site_data_availability = site_data_availability[site_data_availability["timestamp"].notna()]

    # Normalize site_data_availability timestamp to date-only
    site_data_availability["merge_timestamp"] = site_data_availability["timestamp"].dt.date

    # Check for duplicates in merge keys
    alarms_duplicates = alarms.duplicated(subset=["site_code", "merge_timestamp"]).sum()
    site_data_duplicates = site_data_availability.duplicated(subset=["site_code", "merge_timestamp"]).sum()
    
    # Standardize ticket_data
    ticket_data = data_sources["ticket_data"].copy()

    ticket_data.rename(columns={
        'site_id': 'site_code',
        'business_rca': 'rca',
        'ticket_service_interruption_time': 'outage_duration'
    }, inplace=True)
    # Debug: Inspect ticket_data alarm_name and outage_duration
    
    # Ensure alarm_name is string
    ticket_data["alarm_name"] = ticket_data["alarm_name"].astype(str)

    # Convert outage_duration to numeric, coerce errors to NaN
    ticket_data["outage_duration"] = pd.to_numeric(ticket_data["outage_duration"], errors="coerce")

    # Convert from minutes to hours, replace NaN with 0
    ticket_data["outage_duration"] = ticket_data["outage_duration"] / 60.0
    
    ticket_data["outage_duration"] = ticket_data["outage_duration"].fillna(0.0)
    
    # Perform merges with minimal columns
    alarms = alarms[["site_code", "timestamp", "merge_timestamp", "alarm_name", "alarm_id"]].merge(
        alarm_classifications[["alarm_name", "classification"]], on="alarm_name", how="left"
    ).merge(
        site_details[["site_code", "region"]], on="site_code", how="left"
    ).merge(
        site_data_availability[["site_code", "merge_timestamp", "availability_percentage"]],
        on=["site_code", "merge_timestamp"],
        how="left"
    ).merge(
        ticket_data[["site_code", "alarm_name", "rca", "outage_duration"]],
        on=["site_code", "alarm_name"],
        how="left"
    )
    

    # Debug: Inspect merge result
    
    # Fill missing values
    alarms.fillna({
        "availability_percentage": 100.0,
        "rca": "unknown",
        "classification": "unknown",
        "outage_duration": 0.0,
        "region": "unknown"
    }, inplace=True)

    # Add derived features
    # Ensure timestamp is datetime64 for rolling
    
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"], errors="coerce")

    # Sort by site_code and timestamp for rolling
    alarms = alarms.sort_values(["site_code", "timestamp"])

    # print(alarms)
    
    alarms["alarm_count_24h"] = (
    alarms
    .set_index("timestamp")
    .groupby("site_code")["site_code"]
    .rolling("24h", closed="both")
    .count()
    .reset_index(level=0, drop=True)
    .sort_index()
    .values  # convert back to array to assign as new column
)
    alarms["mean_availability_24h"] = (
    alarms
    .set_index("timestamp")
    .groupby("site_code")["availability_percentage"]
    .rolling("24h", closed="both")
    .mean()
    .reset_index(level=0, drop=True)
    .sort_index()
    .values
)
    

    alarms["outage_frequency_7d"] = (
    alarms
    .assign(is_outage=alarms["classification"] == "site_outage")
    .set_index("timestamp")
    .groupby("site_code")["is_outage"]
    .rolling("7d", closed="both")
    .sum()
    .reset_index(level=0, drop=True)
    .sort_index()
    .values
)
    # Drop merge_timestamp
    alarms = alarms.drop(columns=["merge_timestamp"])



    # Add derived features from timestamp
    alarms["hour"] = alarms["timestamp"].dt.hour
    alarms["day_of_week"] = alarms["timestamp"].dt.dayofweek  # Monday=0, Sunday=6

    # Define holidays (example: you can expand this list or use a library like `holidays`)
    from datetime import date
    holiday_list = [
        date(2025, 1, 1),  # New Year's Day
        date(2025, 12, 25),  # Christmas
        # Add more holidays as needed
    ]
    alarms["is_holiday"] = alarms["timestamp"].dt.date.isin(holiday_list).astype(int)

    # Encode categorical variables
    le = LabelEncoder()

    alarms["classification_encoded"] = le.fit_transform(alarms["classification"])
    
    print("Unique classification values:", alarms["classification"].unique())
    print("Encoded classification values:", np.unique(alarms["classification_encoded"]))
    print("LabelEncoder classes:", le.classes_)
    print("Number of classes:", len(le.classes_))


    alarms["rca_encoded"] = le.fit_transform(alarms["rca"])
    alarms["region_encoded"] = le.fit_transform(alarms["region"])

    # Scale numerical features
    scaler = StandardScaler()

    numerical_cols = ["availability_percentage", "alarm_count_24h", "mean_availability_24h", 
                     "outage_frequency_7d", "outage_duration", "hour", "day_of_week", "is_holiday"]
    alarms[numerical_cols] = scaler.fit_transform(alarms[numerical_cols])

    return alarms, scaler, le

def generate_rca_embeddings(rca_values, device="cpu"):

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    try:
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    except OSError as e:
        print(f"PyTorch model load failed: {e}")
        print("Trying to load TensorFlow weights with from_tf=True...")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased", from_tf=True).to(device)

    # print(":Model loaded successfully:")
    model.eval()

    embeddings = {}
    unique_rcas = np.unique(rca_values)

    for rca in unique_rcas:
        inputs = tokenizer(rca, return_tensors="pt", padding=True, truncation=True, max_length=16).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        embeddings[rca] = embedding

    return embeddings


# 4. LLM Integration for Root Cause Explanations
def generate_llm_explanation(prediction, api_key=HF_API_KEY):
    """
    Generate root cause explanation using an LLM via Hugging Face Inference API.
    """
    site_code = prediction["site_code"]
    outage_prob = prediction["outage_probability"]
    outage_type = prediction["predicted_outage_type"]
    duration = prediction["predicted_outage_duration"]
    rca = prediction["rca"]
    
    prompt = (
        f"You are an AI assistant for a network operations center. Given the following outage prediction, provide a concise explanation of the root cause and recommend an action:\n"
        f"- Site: {site_code}\n"
        f"- Outage Type: {outage_type}\n"
        f"- Probability: {outage_prob:.2%}\n"
        f"- Estimated Duration: {duration:.2f} hours\n"
        f"- Root Cause Analysis (RCA): {rca}\n"
        f"Format the response as:\n"
        f"Site {site_code}: {outage_type} predicted with {outage_prob:.2%} probability.\n"
        f"Estimated duration: {duration:.2f} hours.\n"
        f"Root Cause: [Explain the cause based on RCA]\n"
        f"Recommended Action: [Specific action to mitigate or resolve]"
    )
    
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 200, "temperature": 0.7, "top_p": 0.9}
    }
    
    # Placeholder for Hugging Face Inference API (replace with DeepSeek API if available)
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-8b"
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            explanation = response.json()[0]["generated_text"].split(prompt)[-1].strip()
            return explanation
        else:
            return f"Error: LLM API returned status {response.status_code}"
    except Exception as e:
        return f"Error: Failed to query LLM - {str(e)}"

# 5. Asynchronous Data Release
async def release_data_to_service(data: pd.DataFrame, endpoint_url: str):
    async with aiohttp.ClientSession() as session:
        data_json = data.to_dict(orient="records")
        try:
            async with session.post(endpoint_url, json={"data": data_json}) as response:
                if response.status == 200:
                    print(f"Data successfully sent to {endpoint_url}")
                    return await response.json()
                else:
                    print(f"Failed to send data: {response.status}")
                    return None
        except Exception as e:
            print(f"Error sending data: {e}")
            return None

# 6. Create Sequences for LSTM
# def create_sequences(data, seq_length=24, class_target="classification_encoded", reg_target="outage_duration"):
# def create_sequences(data, seq_length=2, class_target="classification_encoded", reg_target="outage_duration"):
#     X, y_class, y_reg = [], [], []
#     feature_cols = [
#         "availability_percentage", "alarm_count_24h", "classification_encoded",
#         "rca_encoded", "hour", "day_of_week", "is_holiday"
#     ]
    
#     for site in data["site_code"].unique():
#         site_data = data[data["site_code"] == site].sort_values("timestamp")
#         for i in range(len(site_data) - seq_length):
#             seq = site_data[feature_cols].iloc[i:i + seq_length].values
#             class_target_val = site_data[class_target].iloc[i + seq_length]
#             reg_target_val = site_data[reg_target].iloc[i + seq_length]
#             X.append(seq)
#             y_class.append(class_target_val)
#             y_reg.append(reg_target_val)
    
#     return np.array(X), np.array(y_class), np.array(y_reg)

def create_sequences(data, seq_length=2, class_target="classification_encoded", reg_target="outage_duration"):
    X, y_class, y_reg = [], [], []
    feature_cols = [
        "availability_percentage", "alarm_count_24h", "classification_encoded",
        "rca_encoded", "hour", "day_of_week", "is_holiday"
    ]
    
    for site in data["site_code"].unique():
        site_data = data[data["site_code"] == site].sort_values("timestamp")
        print(f"Columns for site {site}:", site_data.columns.tolist())  # Debug statement

        for i in range(len(site_data) - seq_length):
            seq = site_data[feature_cols].iloc[i:i + seq_length].values
            class_target_val = site_data[class_target].iloc[i + seq_length]
            reg_target_val = site_data[reg_target].iloc[i + seq_length]
            X.append(seq)
            y_class.append(class_target_val)
            y_reg.append(reg_target_val)
        
    y_class = np.array(y_class)
    print("Unique y_class values:", np.unique(y_class))
    print("Min y_class:", y_class.min(), "Max y_class:", y_class.max())
    
    
    return np.array(X), np.array(y_class), np.array(y_reg)
# 7. Create Tabular Data for XGBoost
def create_tabular_data(data, class_target="classification_encoded", reg_target="outage_duration"):
    feature_cols = [
        "availability_percentage", "alarm_count_24h", "mean_availability_24h",
        "outage_frequency_7d", "rca_encoded", "region_encoded", "hour", "day_of_week", "is_holiday"
    ]
    latest_data = data.groupby("site_code").last().reset_index()
    X = latest_data[feature_cols].values
    y_class = latest_data[class_target].values
    y_reg = latest_data[reg_target].values
    return X, y_class, y_reg

# 8. LSTM Model
class OutageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(OutageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_class = nn.Linear(hidden_size, num_classes)
        self.fc_reg = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        class_output = self.fc_class(out)
        reg_output = self.fc_reg(out)
        return class_output, reg_output

# 9. Custom Dataset for LSTM
class OutageDataset(Dataset):
    def __init__(self, X, y_class, y_reg):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y_class)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_reg[idx]

# 10. Train LSTM
def train_lstm(model, train_loader, num_epochs, device):

    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_class_batch, y_reg_batch in train_loader:
            X_batch, y_class_batch, y_reg_batch = X_batch.to(device), y_class_batch.to(device), y_reg_batch.to(device)
            class_output, reg_output = model(X_batch)
            
            class_loss = criterion_class(class_output, y_class_batch)
            reg_loss = criterion_reg(reg_output.squeeze(), y_reg_batch)
            loss = 0.5 * class_loss + 0.5 * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"LSTM Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 11. Train XGBoost
def train_xgboost(X_train, y_train_class, y_train_reg, num_classes):
    params_class = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss"
    }
    params_reg = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    dtrain_class = xgb.DMatrix(X_train, label=y_train_class)
    dtrain_reg = xgb.DMatrix(X_train, label=y_train_reg)
    model_class = xgb.train(params_class, dtrain_class, num_boost_round=100)
    model_reg = xgb.train(params_reg, dtrain_reg, num_boost_round=100)
    return model_class, model_reg

# 12. Ensemble Prediction
def predict_outages_ensemble(lstm_model, xgb_model_class, xgb_model_reg, data, scaler, le, seq_length, device, rca_embeddings, prediction_horizon="day"):
    lstm_model.eval()
    predictions = []
    current_time = datetime.now()
    
    if prediction_horizon == "day":
        end_time = current_time.replace(hour=23, minute=59, second=59)
    elif prediction_horizon == "tomorrow":
        current_time = current_time + timedelta(days=1)
        end_time = current_time.replace(hour=23, minute=59, second=59)
    else:
        end_time = current_time + timedelta(days=7)
    
    X_tabular, _, _ = create_tabular_data(data)
    dmatrix = xgb.DMatrix(X_tabular)
    xgb_class_probs = xgb_model_class.predict(dmatrix)
    xgb_reg_preds = xgb_model_reg.predict(dmatrix)
    
    with torch.no_grad():
        for idx, site in enumerate(data["site_code"].unique()):
            site_data = data[data["site_code"] == site].sort_values("timestamp")
            if len(site_data) < seq_length:
                continue
            
            seq = site_data.tail(seq_length)[[
                "availability_percentage", "alarm_count_24h", "classification_encoded",
                "rca_encoded", "hour", "day_of_week", "is_holiday"
            ]].values
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            class_output, reg_output = lstm_model(seq)
            lstm_class_prob = torch.softmax(class_output, dim=1).cpu().numpy()[0]
            lstm_reg_pred = reg_output.cpu().numpy()[0, 0]
            
            ensemble_class_prob = 0.6 * lstm_class_prob + 0.4 * xgb_class_probs[idx]
            ensemble_reg_pred = 0.5 * lstm_reg_pred + 0.5 * xgb_reg_preds[idx]
            predicted_class = le.inverse_transform([np.argmax(ensemble_class_prob)])[0]
            
            rca = site_data["rca"].iloc[-1]
            rca_embedding = rca_embeddings.get(rca, np.zeros(768))
            
            prediction = {
                "site_code": site,
                "region": site_data["region"].iloc[-1],
                "prediction_time": current_time.isoformat(),
                "outage_probability": ensemble_class_prob[le.transform(["site_outage"])[0]] if "site_outage" in le.classes_ else 0.0,
                "predicted_outage_type": predicted_class,
                "predicted_outage_duration": float(ensemble_reg_pred),
                "rca": rca,
                "rca_embedding": rca_embedding.tolist()
            }
            prediction["explanation"] = generate_llm_explanation(prediction)
            predictions.append(prediction)
    
    predictions_df = pd.DataFrame(predictions)
    
    region_summary = predictions_df.groupby("region").agg({
        "outage_probability": "mean",
        "site_code": "count"
    }).rename(columns={"site_code": "affected_sites"}).reset_index()
    region_summary["prediction_time"] = current_time.isoformat()
    region_summary["horizon"] = prediction_horizon
    
    return predictions_df, region_summary

# 13. Evaluate Model
def evaluate_model(lstm_model, xgb_model_class, xgb_model_reg, X_val, y_val_class, y_val_reg, data, scaler, le, seq_length, device, rca_embeddings):
    predictions_df, _ = predict_outages_ensemble(
        lstm_model, xgb_model_class, xgb_model_reg, data, scaler, le, seq_length, device, rca_embeddings, "day"
    )
    
    f1 = f1_score(y_val_class, le.transform(predictions_df["predicted_outage_type"]), average="weighted")
    auc = roc_auc_score(pd.get_dummies(y_val_class), pd.get_dummies(le.transform(predictions_df["predicted_outage_type"])), multi_class="ovr")
    rmse = mean_squared_error(y_val_reg, predictions_df["predicted_outage_duration"], squared=False)
    
    print(f"Validation Metrics: F1={f1:.4f}, AUC={auc:.4f}, RMSE={rmse:.4f}")
    return f1, auc, rmse

# 14. FastAPI Startup Event

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global lstm_model, xgb_model_class, xgb_model_reg, scaler, le, rca_embeddings
    data_sources = load_data_sources()
    processed_data, scaler, le = preprocess_data(data_sources)
    rca_embeddings = generate_rca_embeddings(processed_data["rca"].values, device=DEVICE)
    
    # Placeholder: Load pre-trained models (in production, load from files)
    X_lstm, y_class, y_reg = create_sequences(processed_data, seq_length=SEQ_LENGTH)
    lstm_model = OutageLSTM(X_lstm.shape[2], HIDDEN_SIZE, NUM_LAYERS, len(le.classes_)).to(DEVICE)
    # lstm_model.load_state_dict(torch.load("lstm_model.pth"))
    xgb_model_class = xgb.Booster()
    xgb_model_reg = xgb.Booster()
    # xgb_model_class.load_model("xgb_model_class.json")
    # xgb_model_reg.load_model("xgb_model_reg.json")
    
    yield  # Application runs here
    # Optional: Add shutdown logic here
    print("Shutting down...")

# Update FastAPI app initialization
app = FastAPI(title="Noc-netIntel API", description="AI-Powered Network Operations Intelligence Assistant", lifespan=lifespan)


# 15. FastAPI Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    data_sources = load_data_sources()
    processed_data, _, _ = preprocess_data(data_sources)
    
    predictions_df, region_summary = predict_outages_ensemble(
        lstm_model, xgb_model_class, xgb_model_reg, processed_data, scaler, le, SEQ_LENGTH, DEVICE, rca_embeddings, request.horizon
    )
    
    if request.region:
        predictions_df = predictions_df[predictions_df["region"] == request.region]
    
    if predictions_df.empty:
        raise HTTPException(status_code=404, detail="No predictions found for the specified criteria")
    
    # Format response based on prompt
    response_text = f"Outages for {request.horizon} in {request.region or 'all regions'}:\n"
    for _, pred in predictions_df.iterrows():
        response_text += f"\n{pred['explanation']}\n"
    
    return ChatResponse(
        response=response_text,
        predictions=predictions_df.to_dict(orient="records")
    )

@app.get("/forecast", response_model=list[ForecastResponse])
async def get_forecast(horizon: str = "day"):
    data_sources = load_data_sources()
    processed_data, _, _ = preprocess_data(data_sources)
    
    predictions_df, _ = predict_outages_ensemble(
        lstm_model, xgb_model_class, xgb_model_reg, processed_data, scaler, le, SEQ_LENGTH, DEVICE, rca_embeddings, horizon
    )
    return predictions_df.to_dict(orient="records")

@app.get("/schedule")
async def get_schedule():
    return {"message": "FME schedule not implemented yet"}

@app.get("/logs")
async def get_logs():
    return {"message": "Logs not implemented yet"}



# 16. Main Execution (for Training)
async def main():
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    
    data_sources = load_data_sources()

    processed_data, scaler, le = preprocess_data(data_sources)


    rca_embeddings = generate_rca_embeddings(processed_data["rca"].values, device=DEVICE)
    
    # UPLOAD_ENDPOINT = "http://localhost:8000/upload-data"
    # await release_data_to_service(processed_data, UPLOAD_ENDPOINT)
    
    X_lstm, y_class, y_reg = create_sequences(processed_data, seq_length=SEQ_LENGTH)
    
    print("Number of classes for LSTM:", len(le.classes_))


    X_train_lstm, X_temp_lstm, y_train_class, y_temp_class, y_train_reg, y_temp_reg = train_test_split(
        X_lstm, y_class, y_reg, test_size=0.3, random_state=42
    )
    X_val_lstm, X_test_lstm, y_val_class, y_test_class, y_val_reg, y_test_reg = train_test_split(
        X_temp_lstm, y_temp_class, y_temp_reg, test_size=0.5, random_state=42
    )
    
    X_tabular, y_tabular_class, y_tabular_reg = create_tabular_data(processed_data)
    X_train_xgb, X_temp_xgb, y_train_class_xgb, y_temp_class_xgb, y_train_reg_xgb, y_temp_reg_xgb = train_test_split(
        X_tabular, y_tabular_class, y_tabular_reg, test_size=0.3, random_state=42
    )
    X_val_xgb, X_test_xgb, y_val_class_xgb, y_test_class_xgb, y_val_reg_xgb, y_test_reg_xgb = train_test_split(
        X_temp_xgb, y_temp_class_xgb, y_temp_reg_xgb, test_size=0.5, random_state=42
    )
    
    train_dataset = OutageDataset(X_train_lstm, y_train_class, y_train_reg)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    lstm_model = OutageLSTM(X_lstm.shape[2], HIDDEN_SIZE, NUM_LAYERS, len(le.classes_)).to(DEVICE)
    train_lstm(lstm_model, train_loader, NUM_EPOCHS, DEVICE)
    
    xgb_model_class, xgb_model_reg = train_xgboost(X_train_xgb, y_train_class_xgb, y_train_reg_xgb, len(le.classes_))
    
    f1, auc, rmse = evaluate_model(
        lstm_model, xgb_model_class, xgb_model_reg, X_val_lstm, y_val_class, y_val_reg, 
        processed_data, scaler, le, SEQ_LENGTH, DEVICE, rca_embeddings
    )
    
    for horizon in ["day", "tomorrow", "week"]:
        predictions_df, region_summary = predict_outages_ensemble(
            lstm_model, xgb_model_class, xgb_model_reg, processed_data, scaler, le, SEQ_LENGTH, DEVICE, rca_embeddings, horizon
        )
        print(f"Predictions for {horizon}:")
        print(predictions_df[["site_code", "outage_probability", "predicted_outage_type", "predicted_outage_duration", "explanation"]])
        print(f"Region Summary for {horizon}:")
        print(region_summary)

if __name__ == "__main__":
    asyncio.run(main())