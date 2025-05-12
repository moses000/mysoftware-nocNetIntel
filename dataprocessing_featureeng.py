import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sqlalchemy import create_engine

# === CONFIGURATION ===
DB_USER = "your_pg_user"
DB_PASS = "your_pg_password"
DB_HOST = "your.pg.server.com"
DB_PORT = "5432"
DB_NAME = "your_database"

# Connect to PostgreSQL
db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

# === STEP 1: Load and Preprocess Data ===
def load_and_preprocess_data():
    # Load the data from PostgreSQL (this is just an example, adjust query as needed)
    query = """
        SELECT * FROM rca_tickets_processed 
        WHERE timestamp >= now() - interval '1 month'
    """
    df = pd.read_sql(query, engine)

    # Drop rows with essential missing values (like site_id, timestamp)
    df.dropna(subset=["site_id", "timestamp"], inplace=True)

    # Feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["duration_minutes"] = df["duration_minutes"].fillna(df["duration_minutes"].mean())

    # Convert categorical variables to numerical (e.g., site_id, weekday)
    le_site = LabelEncoder()
    df["site_id_encoded"] = le_site.fit_transform(df["site_id"])

    le_weekday = LabelEncoder()
    df["weekday_encoded"] = le_weekday.fit_transform(df["weekday"])

    # Add more features as needed (e.g., rolling averages, lag features)

    # Prepare features and target
    X = df[["site_id_encoded", "hour", "weekday_encoded", "duration_minutes"]]  # Features
    y = df["outage"]  # Target variable (adjust the column if needed)

    return X, y

# === STEP 2: Train Model ===
def train_model():
    X, y = load_and_preprocess_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model for future predictions
    joblib.dump(model, "outage_predictor.pkl")
    print("Model saved as 'outage_predictor.pkl'")

# === STEP 3: Run the model training ===
if __name__ == "__main__":
    train_model()
