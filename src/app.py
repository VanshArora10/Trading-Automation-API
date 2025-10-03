from fastapi import FastAPI
from .daily_recommender_once import run_once

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Trading API is live"}

@app.get("/run-trade")
def run_trade():
    results = run_once()
    if not results:
        return {"Trade": "No", "message": "No signals found"}
    return results
