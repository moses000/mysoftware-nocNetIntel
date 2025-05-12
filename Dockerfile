FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV HF_API_KEY=your_huggingface_api_key
CMD ["uvicorn", "site_outage_prediction_pipeline_new:app", "--host", "0.0.0.0", "--port", "8000"]