import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import aiohttp
import asyncio
from datetime import datetime, timedelta
import uuid

# 1. Data Source Outline and Loading (Unchanged)
def load_data_sources():
    data_sources = {
        "alarms": pd.DataFrame(),
        "alarm_classifications": pd.DataFrame(),
        "site_details": pd.DataFrame(),
        "site_data_availability": pd.DataFrame(),
        "ticket_data": pd.DataFrame()
    }
    return data_sources

# 2. Data Preprocessing (Extended for XGBoost)
def preprocess_data(data_sources):
    """
    Clean, merge, and preprocess data for LSTM and XGBoost.
    Returns: Processed DataFrame, scaler, and label encoder.
    """
    # Merge data sources (same as before)
    alarms = data_sources["alarms"].merge(
        data_sources["alarm_classifications"], on="alarm_name", how="left"
    ).merge(
        data_sources["site_details"], on="site_code", how="left"
    ).merge(
        data_sources["site_data_availability"], on=["site_code", "timestamp"], how="left"
    ).merge(
        data_sources["ticket_data"][["site_code", "alarm_id", "rca"]],
        on=["site_code", "alarm_id"],
        how="left"
    )
    
    # Handle missing values
    alarms.fillna({
        "availability_percentage": 100.0,
        "rca": "unknown",
        "classification": "unknown"
    }, inplace=True)
    
    # Convert timestamp
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"])
    
    # Feature Engineering for LSTM
    alarms["hour"] = alarms["timestamp"].dt.hour
    alarms["day_of_week"] = alarms["timestamp"].dt.dayofweek
    alarms["is_holiday"] = alarms["timestamp"].dt.date.isin([]).astype(int)
    alarms["alarm_count_24h"] = alarms.groupby("site_code")["timestamp"].transform(
        lambda x: x.rolling("24h", closed="both").count()
    )
    
    # Additional Features for XGBoost (tabular)
    alarms["mean_availability_24h"] = alarms.groupby("site_code")["availability_percentage"].transform(
        lambda x: x.rolling("24h", closed="both").mean()
    )
    alarms["outage_frequency_7d"] = alarms.groupby("site_code")["classification"].transform(
        lambda x: (x == "site_outage").rolling("7d", closed="both").sum()
    )
    
    # Encode categorical variables
    le = LabelEncoder()
    alarms["classification_encoded"] = le.fit_transform(alarms["classification"])
    alarms["rca_encoded"] = le.fit_transform(alarms["rca"])
    alarms["region_encoded"] = le.fit_transform(alarms["region"])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["availability_percentage", "alarm_count_24h", "mean_availability_24h", "outage_frequency_7d"]
    alarms[numerical_cols] = scaler.fit_transform(alarms[numerical_cols])
    
    return alarms, scaler, le

# 3. Asynchronous Data Release (Unchanged)
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

# 4. Create Sequences for LSTM
def create_sequences(data, seq_length=24, target_col="classification_encoded"):
    X, y = [], []
    feature_cols = [
        "availability_percentage", "alarm_count_24h", "classification_encoded",
        "rca_encoded", "hour", "day_of_week", "is_holiday"
    ]
    
    for site in data["site_code"].unique():
        site_data = data[data["site_code"] == site].sort_values("timestamp")
        for i in range(len(site_data) - seq_length):
            seq = site_data[feature_cols].iloc[i:i + seq_length].values
            target = site_data[target_col].iloc[i + seq_length]
            X.append(seq)
            y.append(target)
    
    return np.array(X), np.array(y)

# 5. Create Tabular Data for XGBoost
def create_tabular_data(data, target_col="classification_encoded"):
    """
    Prepare tabular data for XGBoost.
    Returns: Feature matrix (X) and targets (y).
    """
    feature_cols = [
        "availability_percentage", "alarm_count_24h", "mean_availability_24h",
        "outage_frequency_7d", "rca_encoded", "region_encoded", "hour", "day_of_week", "is_holiday"
    ]
    latest_data = data.groupby("site_code").last().reset_index()
    X = latest_data[feature_cols].values
    y = latest_data[target_col].values
    return X, y

# 6. LSTM Model
class OutageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(OutageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 7. Custom Dataset for LSTM
class OutageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 8. Train LSTM
def train_lstm(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"LSTM Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 9. Train XGBoost
def train_xgboost(X_train, y_train, num_classes):
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss"
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

# 10. Ensemble Prediction
def predict_outages_ensemble(lstm_model, xgb_model, data, scaler, le, seq_length, device, prediction_horizon="day"):
    """
    Predict outages using LSTM and XGBoost ensemble.
    """
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
    
    # Prepare tabular data for XGBoost
    X_tabular, _ = create_tabular_data(data)
    dmatrix = xgb.DMatrix(X_tabular)
    xgb_probs = xgb_model.predict(dmatrix)
    
    # LSTM predictions
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
            
            lstm_output = lstm_model(seq)
            lstm_prob = torch.softmax(lstm_output, dim=1).cpu().numpy()[0]
            
            # Ensemble: Weighted average (0.6 LSTM, 0.4 XGBoost)
            ensemble_prob = 0.6 * lstm_prob + 0.4 * xgb_probs[idx]
            predicted_class = le.inverse_transform([np.argmax(ensemble_prob)])[0]
            
            predictions.append({
                "site_code": site,
                "prediction_time": current_time,
                "outage_probability": ensemble_prob[le.transform(["site_outage"])[0]] if "site_outage" in le.classes_ else 0.0,
                "predicted_outage_type": predicted_class
            })
    
    return pd.DataFrame(predictions)

# 11. Main Execution
async def main():
    # Configuration
    SEQ_LENGTH = 24
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    UPLOAD_ENDPOINT = "http://localhost:8000/upload-data"
    
    # Load and preprocess data
    data_sources = load_data_sources()
    processed_data, scaler, le = preprocess_data(data_sources)
    
    # Release data asynchronously
    await release_data_to_service(processed_data, UPLOAD_ENDPOINT)
    
    # Prepare data for LSTM
    X_lstm, y_lstm = create_sequences(processed_data, seq_length=SEQ_LENGTH)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
    
    # Prepare data for XGBoost
    X_tabular, y_tabular = create_tabular_data(processed_data)
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_tabular, y_tabular, test_size=0.2, random_state=42)
    
    # Train LSTM
    train_dataset = OutageDataset(X_train_lstm, y_train_lstm)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    lstm_model = OutageLSTM(X_lstm.shape[2], HIDDEN_SIZE, NUM_LAYERS, len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    train_lstm(lstm_model, train_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train_xgb, y_train_xgb, len(le.classes_))
    
    # Evaluate models
    lstm_pred = predict_outages_ensemble(lstm_model, xgb_model, processed_data, scaler, le, SEQ_LENGTH, DEVICE, "day")
    lstm_f1 = f1_score(y_test_xgb, le.transform(lstm_pred["predicted_outage_type"]), average="weighted")
    print(f"Ensemble F1 Score: {lstm_f1:.4f}")
    
    # Predict outages
    for horizon in ["day", "tomorrow", "week"]:
        predictions = predict_outages_ensemble(lstm_model, xgb_model, processed_data, scaler, le, SEQ_LENGTH, DEVICE, horizon)
        print(f"Predictions for {horizon}:")
        print(predictions)

if __name__ == "__main__":
    asyncio.run(main())