import pytest
import pandas as pd
import numpy as np
from site_outage_prediction_pipeline_new import app, preprocess_data, generate_rca_embeddings, create_sequences, create_tabular_data
from fastapi.testclient import TestClient
# from your_script import 
# Mock data for testing
@pytest.fixture
def mock_data_sources():
    alarms = pd.DataFrame({
        "alarm_name": ["alarm1", "alarm2"],
        "site_code": ["site1", "site2"],
        "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "alarm_id": [1, 2]
    })
    alarm_classifications = pd.DataFrame({
        "alarm_name": ["alarm1", "alarm2"],
        "classification": ["site_outage", "minor"]
    })
    site_details = pd.DataFrame({
        "site_code": ["site1", "site2"],
        "region": ["region1", "region2"]
    })
    site_data_availability = pd.DataFrame({
        "site_code": ["site1", "site2"],
        "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "availability_percentage": [99.0, 98.0]
    })
    ticket_data = pd.DataFrame({
        "site_code": ["site1", "site2"],
        "alarm_id": [1, 2],
        "rca": ["power_failure", "unknown"],
        "outage_duration": [2.0, 0.0]
    })
    return {
        "alarms": alarms,
        "alarm_classifications": alarm_classifications,
        "site_details": site_details,
        "site_data_availability": site_data_availability,
        "ticket_data": ticket_data
    }

def test_preprocess_data(mock_data_sources):
    processed_data, scaler, le = preprocess_data(mock_data_sources)
    assert not processed_data.empty
    assert "classification_encoded" in processed_data.columns
    assert "rca_encoded" in processed_data.columns
    assert processed_data["timestamp"].dtype == "datetime64[ns]"

def test_generate_rca_embeddings(mock_data_sources):
    processed_data, _, _ = preprocess_data(mock_data_sources)
    embeddings = generate_rca_embeddings(processed_data["rca"].values)
    assert len(embeddings) == len(np.unique(processed_data["rca"]))
    assert "power_failure" in embeddings
    assert embeddings["power_failure"].shape == (768,)

def test_create_sequences(mock_data_sources):
    processed_data, _, _ = preprocess_data(mock_data_sources)
    X, y_class, y_reg = create_sequences(processed_data, seq_length=1)
    assert X.shape[0] == y_class.shape[0] == y_reg.shape[0]
    assert X.shape[1] == 1  # seq_length
    assert X.shape[2] == 7  # number of features

def test_create_tabular_data(mock_data_sources):
    processed_data, _, _ = preprocess_data(mock_data_sources)
    X, y_class, y_reg = create_tabular_data(processed_data)
    assert X.shape[0] == y_class.shape[0] == y_reg.shape[0]
    assert X.shape[1] == 9  # number of features

@pytest.fixture
def client():
    return TestClient(app)

def test_chat_endpoint(client, mock_data_sources):
    response = client.post("/chat", json={"prompt": "test", "horizon": "day"})
    assert response.status_code in [200, 404]  # 404 if no predictions due to mock data
    if response.status_code == 200:
        assert "response" in response.json()
        assert "predictions" in response.json()

def test_forecast_endpoint(client):
    response = client.get("/forecast?horizon=day")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert isinstance(response.json(), list)