from fastapi import FastAPI
from .daily_recommender_once import run_once

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Trading Automation API is running!"}

@app.get("/run-trade")
def run_trade():
    results = run_once()
    return results
