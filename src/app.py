from fastapi import FastAPI
from .daily_recommender_once import run_once

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Trading Automation API is running 🚀"}

@app.get("/run-trade")
def run_trade():
    return run_once()
