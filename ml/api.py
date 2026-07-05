"""
RecoPulse — FastAPI serving layer
=================================
Serves the trained hybrid model behind a small, explainable HTTP API that the
Next.js product calls.

  GET  /health              liveness + whether the model is loaded
  GET  /metrics             the honest metrics.json (powers the monitoring cockpit)
  GET  /users?limit=…       sample of known user_ids (to populate the console dropdown)
  POST /predict             {user_id, product_name} -> rating + tier + reason + similar items
  POST /explain             alias of /predict with explain always on

Run:  uvicorn ml.api:app --reload --port 8000
"""
from __future__ import annotations
import os
import json
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HybridPredictor          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "artifacts", "model.joblib")
METRICS_PATH = os.path.join(HERE, "artifacts", "metrics.json")

app = FastAPI(title="RecoPulse API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_model: HybridPredictor | None = None
_metrics: dict | None = None
_user_ids: list = []


@app.on_event("startup")
def _load():
    global _model, _metrics, _user_ids
    if os.path.exists(MODEL_PATH):
        _model = HybridPredictor.load(MODEL_PATH)
        _user_ids = sorted(_model.user_stats.keys())[:2000]
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)


class PredictRequest(BaseModel):
    user_id: int
    product_name: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None,
            "metrics_loaded": _metrics is not None}


@app.get("/metrics")
def metrics():
    if _metrics is None:
        raise HTTPException(404, "metrics.json not found — run ml/train.py first")
    return _metrics


@app.get("/users")
def users(limit: int = 50):
    return {"user_ids": _user_ids[:limit], "total_known": len(_user_ids)}


@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "model not loaded — run ml/train.py first")
    result = _model.predict(req.user_id, req.product_name, explain=True)
    payload = result.as_dict()
    payload["user_id"] = req.user_id
    payload["product_name"] = req.product_name
    payload["user_known"] = req.user_id in _model.user_stats
    return payload


@app.post("/explain")
def explain(req: PredictRequest):
    return predict(req)
