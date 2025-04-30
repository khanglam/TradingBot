from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import importlib.util

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class BacktestRequest(BaseModel):
    parameters: Dict[str, Any]

class OptimizeRequest(BaseModel):
    parameters: Dict[str, Any]
    optimize_params: Optional[List[str]] = None

class Trade(BaseModel):
    date: str
    action: str
    price: float
    pnl: float

class BacktestResult(BaseModel):
    profit_loss: float
    trades: List[Trade]
    stats: Dict[str, Any]

class ParameterInfo(BaseModel):
    name: str
    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None

# --- Dummy logic for now ---
from tests.backtest_lorentzian import run_lorentzian_backtest, run_lorentzian_optimize

@app.post("/backtest", response_model=BacktestResult)
def run_backtest(req: BacktestRequest):
    result = run_lorentzian_backtest(req.parameters)
    trades = [Trade(**trade) for trade in result['trades']]
    return BacktestResult(
        profit_loss=result['profit_loss'],
        trades=trades,
        stats=result['stats']
    )

@app.post("/optimize", response_model=BacktestResult)
def run_optimize(req: OptimizeRequest):
    result = run_lorentzian_optimize(req.parameters)
    trades = [Trade(**trade) for trade in result['trades']]
    # Optionally, you can add best_params to stats or return separately
    stats = result['stats']
    if 'best_params' in result:
        stats['best_params'] = result['best_params']
    return BacktestResult(
        profit_loss=result['profit_loss'],
        trades=trades,
        stats=stats
    )

@app.get("/parameters", response_model=List[ParameterInfo])
def get_parameters():
    # TODO: Dynamically extract from your strategy
    return [
        ParameterInfo(name="emaPeriod", type="int", default=20, min=5, max=200),
        ParameterInfo(name="adxThreshold", type="float", default=20.0, min=0.0, max=50.0),
        ParameterInfo(name="useAdxFilter", type="bool", default=True, choices=[True, False]),
    ]

@app.get("/stats", response_model=BacktestResult)
def get_stats():
    # Dummy stats and trade history
    return BacktestResult(
        profit_loss=1500.0,
        trades=[
            Trade(date="2024-01-01", action="BUY", price=100.0, pnl=10.0),
            Trade(date="2024-01-02", action="SELL", price=110.0, pnl=10.0),
            Trade(date="2024-01-03", action="BUY", price=105.0, pnl=-5.0),
        ],
        stats={"sharpe": 1.5, "max_drawdown": -0.12}
    )
def get_parameters():
    # TODO: Dynamically extract from your strategy
    return [
        ParameterInfo(name="emaPeriod", type="int", default=20, min=5, max=200),
        ParameterInfo(name="adxThreshold", type="float", default=20.0, min=0.0, max=50.0),
        ParameterInfo(name="useAdxFilter", type="bool", default=True, choices=[True, False]),
    ]

@app.post("/parameters/add", response_model=List[ParameterInfo])
def add_parameter(param: ParameterInfo):
    # TODO: Add parameter to strategy config
    return get_parameters()

@app.post("/parameters/remove", response_model=List[ParameterInfo])
def remove_parameter(param: ParameterInfo):
    # TODO: Remove parameter from strategy config
    return get_parameters()

@app.get("/health")
def health():
    return {"status": "ok"}
