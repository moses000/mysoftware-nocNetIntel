from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd

app = FastAPI()
engine = create_engine("postgresql://user:password@localhost:5432/dbname")

@app.post("/upload-data")
async def upload_data(payload: dict):
    data = pd.DataFrame(payload["data"])
    data.to_sql("preprocessed_data", engine, if_exists="append", index=False)
    return {"status": "success"}