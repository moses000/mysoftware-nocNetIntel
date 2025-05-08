# 🧠 Noc-netIntel – AI-Powered Network Operations Intelligence Assistant

**Noc-netIntel** is an intelligent chat-driven platform designed to forecast critical network outages, provide possible root cause explanations, suggest proactive FME (Field Maintenance Engineer) deployment, and recommend resolutions — all powered by advanced AI, NLP, and time-series forecasting.


## 💡 What It Does

- 🔮 **Predicts outages** today, tomorrow, and over the week
- 📉 **Identifies root causes** using LSTM + LLM reasoning
- 📅 **Schedules field engineers** proactively
- 🧠 **Suggests resolutions** from a growing knowledge base
- 💬 **Conversational interface** with LLM (DeepSeek/OpenAI-compatible)


## 🛠 Technology Stack

| Layer                  | Tech                                |
|------------------------|-------------------------------------|
| **AI/NLP**             | DeepSeek / Custom LLM               |
| **ML/Forecasting**     | PyTorch + Custom LSTM               |
| **Backend**            | Python (FastAPI preferred)          |
| **Frontend**           | JavaScript (React recommended)      |
| **Database**           | PostgreSQL                          |
| **Data Pipeline**      | Python Scripts / Celery Tasks       |
| **Deployment**         | Docker & Docker Compose             |
| **Scheduler (optional)**| Celery + Redis for task management |


## 🧬 End-to-End Workflow

### 1. 🔗 Data Collection
- Sources: Sensor logs, BTS data, voltage/current levels, historical tickets, alarms
- Stored in PostgreSQL (structured) and optional object storage (raw logs)

### 2. 🧹 Data Preprocessing
- Cleansing missing/nulls, noise filtering
- Timestamp alignment, interpolation
- Scaling, encoding categorical signals (battery status, alarm type)

### 3. 🔧 Feature Engineering
- Temporal signals: time of day, day of week, holiday
- Environmental: power metrics, weather (optional)
- Historical: frequency of past outages, lag features
- Rolling stats: moving average, rate of failure

### 4. 📊 ML Forecasting (PyTorch + LSTM)
- Input: Sequence of multivariate time series
- Architecture: Multi-head LSTM → Dense heads (classification + regression)
- Outputs:
  - Outage probability
  - Affected region/site
  - Possible root cause embeddings
- Metrics: F1, AUC for classification; RMSE for regression

### 5. 🧠 NLP Reasoning Layer (DeepSeek / LLM)
- Converts ML output into readable advice
- Enhances with historical patterns and predefined rules
- Formats chat response: outage + root cause + FME plan + resolution

### 6. 📅 Proactive FME Scheduler
- Ranks urgency and location clustering
- Optimizes FME routing using heuristic or ML-based dispatch
- Integrates with external calendars/ticketing if needed


## 💬 Sample Chat Interaction

**User**: "What outages are expected tomorrow in the North East zone?"  
**Noc-netIntel**:

🛑 Predicted 3 possible outages:

* Site BGH-29 (Power drain) – 87% chance
* Site TMT-02 (Overload) – 72% chance
* Site JAK-10 (Backhaul degradation) – 55% chance

📌 Root Causes: Battery degradation, high load demand, backhaul link instability
🛠 Recommended Actions: Pre-deploy backup power units, initiate remote checks
👷 FME Suggestion: Team Alpha, report at 06:30 AM



## 🚀 API Overview

- `POST /chat` – Accepts user prompt, returns AI-generated insight
- `GET /forecast` – Returns raw model prediction
- `GET /schedule` – Lists recommended FME deployments
- `GET /logs` – Access recent outage logs (if allowed)

> Full Swagger UI at: `http://localhost:8000/docs`


## 🗃 Sample PostgreSQL Schema

CREATE TABLE outage_forecasts (
  id SERIAL PRIMARY KEY,
  site_code TEXT,
  prediction_date TIMESTAMP,
  outage_probability FLOAT,
  root_cause TEXT,
  fme_plan TEXT,
  resolution TEXT
);

## 🐳 Setup and Deployment

### ✅ Prerequisites

* Docker & Docker Compose
* Python 3.9+
* Node.js (for frontend)

### 📦 Running Locally

bash
git clone https://github.com/your-org/noc-netintel.git
cd noc-netintel

# Run with Docker Compose
docker-compose up --build

> Services:
>
> * `backend`: FastAPI ML/NLP engine
> * `frontend`: React chat UI (optional)
> * `ml_worker`: PyTorch + model runner
> * `postgres`: SQL data store


## 🔐 Auth & Roles

* JWT-based auth
* Roles: Admin, Analyst, Engineer
* Granular data access policies


## 📊 Monitoring & Logging

* Optional: Add Grafana for real-time alert visualization
* Backend logs all predictions and user queries
* Alerts for model drift / threshold breaches

## ✍️ Wiki & Docs

* 📘 `docs/data-pipeline.md`: Ingestion, ETL, transformations
* 📘 `docs/model.md`: LSTM architecture, training notes
* 📘 \`docs
