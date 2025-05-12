import joblib
import pandas as pd
from sqlalchemy import create_engine

# Load your trained model
model = joblib.load("outage_predictor.pkl")

# Connect to PostgreSQL
engine = create_engine("postgresql://user:pass@host:port/dbname")

# Load latest 1 hour of data
df = pd.read_sql("""
    SELECT * FROM rca_tickets_processed 
    WHERE timestamp >= now() - interval '1 hour'
""", engine)

# Preprocess into model input
df["site_id"] = df["site_id"].astype(str)
X = df[["hour", "weekday", "duration_minutes", "site_id"]]  # Adjust to match your trained features

# Predict
predictions = model.predict(X)

# You can store or alert based on predictions
df["outage_risk"] = predictions
print(df[["site_id", "timestamp", "outage_risk"]])
