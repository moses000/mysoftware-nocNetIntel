import joblib
import pandas as pd
from sqlalchemy import create_engine

# Load the trained model
model = joblib.load("outage_predictor.pkl")

# === CONFIGURATION ===
DB_USER = "your_pg_user"
DB_PASS = "your_pg_password"
DB_HOST = "your.pg.server.com"
DB_PORT = "5432"
DB_NAME = "your_database"
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# === Real-time Prediction Function ===
def predict_outage():
    # Query for the latest data (e.g., last 1 hour of RCA ticket data)
    query = """
        SELECT * FROM rca_tickets_processed 
        WHERE timestamp >= now() - interval '1 hour'
    """
    df = pd.read_sql(query, engine)

    # Preprocess the data (similar to training step)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["duration_minutes"] = df["duration_minutes"].fillna(df["duration_minutes"].mean())

    le_site = LabelEncoder()
    df["site_id_encoded"] = le_site.fit_transform(df["site_id"])

    le_weekday = LabelEncoder()
    df["weekday_encoded"] = le_weekday.fit_transform(df["weekday"])

    # Prepare features for prediction
    X_new = df[["site_id_encoded", "hour", "weekday_encoded", "duration_minutes"]]

    # Predict outages
    predictions = model.predict(X_new)

    # Add predictions to dataframe
    df["outage_risk"] = predictions
    print(df[["site_id", "timestamp", "outage_risk"]])  # Or save/send as needed

if __name__ == "__main__":
    predict_outage()
