import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine

# === CONFIGURATION ===
API_URL = "https://your-api.com/api/rca-tickets"
USERNAME = "your_api_user"
PASSWORD = "your_api_password"

# PostgreSQL config
DB_USER = "your_pg_user"
DB_PASS = "your_pg_password"
DB_HOST = "your.pg.server.com"
DB_PORT = "5432"
DB_NAME = "your_database"
TABLE_NAME = "rca_tickets_processed"

# SQLAlchemy connection string
db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

# Optional query parameters
params = {
    "from_date": "2025-05-06T00:00:00Z",
    "to_date": "2025-05-07T00:00:00Z"
}

# === STEP 1: Fetch from API
def fetch_rca_data():
    response = requests.get(API_URL, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params)
    response.raise_for_status()
    return pd.DataFrame(response.json())

# === STEP 2: Clean and process, keeping unknown fields
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Define the expected fields
    known_columns = {
        "site_id", "timestamp", "ticket_id", "issue_type",
        "resolved", "resolved_at", "cause_code", "duration_minutes"
    }

    # Ensure required fields exist
    if not {"site_id", "timestamp"}.issubset(df.columns):
        raise ValueError("Missing required columns: 'site_id' and/or 'timestamp'.")

    # Normalize and clean known columns
    df["site_id"] = df["site_id"].str.upper().str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    # Identify unknown/dynamic fields
    extra_cols = list(set(df.columns) - known_columns - {"hour", "weekday"})
    df["extra_data"] = df[extra_cols].to_dict(orient="records") if extra_cols else [{}]

    # Keep only the structured fields + metadata + JSONB
    final_columns = list(known_columns) + ["hour", "weekday", "extra_data"]
    return df[final_columns]

# === STEP 3: Store in PostgreSQL
def store_to_postgres(df: pd.DataFrame):
    df.to_sql(TABLE_NAME, engine, if_exists="append", index=False, method="multi")
    print(f"✅ Stored {len(df)} records into '{TABLE_NAME}'.")

# === Run pipeline
if __name__ == "__main__":
    raw_df = fetch_rca_data()
    processed_df = process_data(raw_df)
    store_to_postgres(processed_df)
