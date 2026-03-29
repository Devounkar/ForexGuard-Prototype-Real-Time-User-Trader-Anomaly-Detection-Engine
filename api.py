"""
api.py
------
FastAPI REST layer for the Forex Anomaly Detection System.

Endpoints
---------
GET  /health          — liveness check, returns model load status
POST /score           — score a single user by user_id (uses saved features)
POST /predict         — score a raw feature vector (no prior data needed)
POST /score/batch     — score multiple user_ids in one call
GET  /alerts          — retrieve generated alerts (paginated, filterable)
GET  /alerts/{alert_id}  — retrieve a single alert by ID
GET  /users/{user_id}/risk  — full risk profile for one user

Run
---
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker-compose up

Notes
-----
- Models are loaded once at startup (lifespan event) to avoid cold-start
  latency on each request.
- All endpoints return structured JSON with a consistent envelope:
    { "status": "ok"|"error", "data": ..., "message": str }
- Feature vector inputs are validated with Pydantic v2 models.
- The /predict endpoint accepts raw feature values so the API can be
  called without the user having historical data in the system.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Path setup — resolve project root so imports work when api.py lives in src/
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.alert_system import AlertSystem
from src.autoencoder import AutoencoderDetector
from src.explainability import RuleBasedExplainer
from src.model import AnomalyDetector, MODEL_FEATURES

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = str(ROOT / "configs" / "config.yaml")
DATA_PATH   = ROOT / "data"
LABELS_PATH = DATA_PATH / "user_labels.parquet"


# ---------------------------------------------------------------------------
# Application state — models loaded once at startup
# ---------------------------------------------------------------------------

class AppState:
    detector:    Optional[AnomalyDetector]     = None
    ae_detector: Optional[AutoencoderDetector] = None
    explainer:   Optional[RuleBasedExplainer]  = None
    alert_sys:   Optional[AlertSystem]         = None
    features_df: Optional[pd.DataFrame]        = None
    config:      dict                          = {}


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data at startup, release at shutdown."""
    logger.info("Loading models …")
    try:
        state.config = yaml.safe_load(open(CONFIG_PATH))

        state.detector = AnomalyDetector(CONFIG_PATH).load()
        logger.info("Isolation Forest loaded ✓")

        ae_path = DATA_PATH / "autoencoder.pkl"
        if ae_path.exists():
            state.ae_detector = AutoencoderDetector(hidden_dim=32, encoding_dim=16).load()
            logger.info("Autoencoder loaded ✓")
        else:
            logger.warning("Autoencoder not found — single-model mode.")

        features_path = DATA_PATH / "train_features.parquet"
        if features_path.exists():
            train_df = pd.read_parquet(features_path)
            state.explainer = RuleBasedExplainer(top_k=3).fit(train_df)
            logger.info("Explainer fitted on %d training users ✓", len(train_df))

        full_features = DATA_PATH / "features.parquet"
        if full_features.exists():
            state.features_df = pd.read_parquet(full_features)
            logger.info("Feature store loaded: %d users ✓", len(state.features_df))

        state.alert_sys = AlertSystem(CONFIG_PATH)
        logger.info("AlertSystem ready ✓")

    except Exception as exc:
        logger.error("Startup error: %s", exc)

    yield

    logger.info("Shutting down API …")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ForexGuard Anomaly Detection API",
    description=(
        "Real-time anomaly scoring for forex user and trader behaviour. "
        "Powered by Isolation Forest + Bottleneck Autoencoder ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FeatureVector(BaseModel):
    """Raw feature vector for a user. All fields optional — missing values
    are filled with 0 (same behaviour as the training pipeline)."""

    user_id: str = Field(..., description="Opaque user identifier")

    # Portal — rates
    login_count_per_day:            float = 0.0
    deposit_count_per_day:          float = 0.0
    withdrawal_count_per_day:       float = 0.0
    support_ticket_count_per_day:   float = 0.0
    document_upload_count_per_day:  float = 0.0
    account_mod_count_per_day:      float = 0.0
    sensitive_mod_count_per_day:    float = 0.0

    # Portal — ratios
    unique_ips:                     float = 0.0
    unique_regions:                 float = 0.0
    geo_change_rate:                float = 0.0
    withdrawal_deposit_ratio:       float = 0.0
    avg_session_duration:           float = 0.0
    high_priority_ticket_rate:      float = 0.0
    unverified_doc_rate:            float = 0.0

    # Portal — new signals
    night_login_ratio:              float = 0.0
    avg_inter_event_s:              float = 0.0
    min_inter_event_s:              float = 0.0
    withdrawal_surge_ratio:         float = 1.0

    # Trading — core
    avg_trade_size:                 float = 0.0
    max_trade_size:                 float = 0.0
    std_trade_size:                 float = 0.0
    avg_leverage:                   float = 0.0
    max_leverage:                   float = 0.0
    avg_margin_usage:               float = 0.0
    max_margin_usage:               float = 0.0
    margin_spike_rate:              float = 0.0
    trade_frequency_per_hour:       float = 0.0
    max_trades_5min:                float = 0.0
    unique_pairs:                   float = 0.0
    total_pnl:                      float = 0.0

    # Trading — new signals
    std_pnl:                        float = 0.0
    pnl_volatility_ratio:           float = 0.0
    herfindahl_pairs:               float = 1.0
    last_trade_size_zscore:         float = 0.0
    last_margin_zscore:             float = 0.0
    last_leverage_zscore:           float = 0.0

    # Cross-domain — new signals
    churn_financial_ratio:          float = 0.0
    dormancy_withdrawal_score:      float = 0.0
    days_since_last_trade:          float = 0.0

    @field_validator("user_id")
    @classmethod
    def user_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id must not be empty")
        return v.strip()

    def to_series(self) -> pd.Series:
        return pd.Series(self.model_dump())


class ScoreRequest(BaseModel):
    user_id: str = Field(..., description="User ID to look up in the feature store")


class BatchScoreRequest(BaseModel):
    user_ids: list[str] = Field(..., min_length=1, max_length=500)


class ScoreResponse(BaseModel):
    status:           str
    user_id:          str
    anomaly_score:    float
    is_anomaly:       bool
    severity:         str
    top_features:     list[dict]
    narrative:        str
    ensemble_score:   Optional[float] = None
    ae_score:         Optional[float] = None
    message:          str = ""


class HealthResponse(BaseModel):
    status:           str
    models_loaded:    dict[str, bool]
    feature_store:    int
    version:          str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.75:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"


def _score_and_explain(features_df: pd.DataFrame) -> list[dict]:
    """Score a features DataFrame with IF (+ AE if available) and explain."""
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    if_results = state.detector.score(features_df)

    ae_results = None
    if state.ae_detector is not None:
        try:
            ae_results = state.ae_detector.score(features_df)
        except Exception as exc:
            logger.warning("AE scoring failed: %s", exc)

    output = []
    for _, row in if_results.iterrows():
        uid        = row["user_id"]
        if_score   = float(row["anomaly_score"])
        is_anomaly = bool(row["is_anomaly"])

        ae_score       = None
        ensemble_score = if_score
        if ae_results is not None:
            ae_row         = ae_results[ae_results["user_id"] == uid]
            if not ae_row.empty:
                ae_score       = float(ae_row.iloc[0]["anomaly_score"])
                ensemble_score = round(0.5 * if_score + 0.5 * ae_score, 4)
                # OR-logic: anomaly if EITHER model flags
                is_anomaly = is_anomaly or bool(ae_row.iloc[0]["is_anomaly"])

        # Explanation
        top_features = []
        narrative    = "Anomaly score elevated — no specific rule triggered."
        if state.explainer is not None:
            feat_row = features_df[features_df["user_id"] == uid]
            if not feat_row.empty:
                expl         = state.explainer.explain(feat_row.iloc[0])
                top_features = expl["triggered_rules"]
                narrative    = expl["narrative"]

        output.append({
            "user_id":        uid,
            "anomaly_score":  round(ensemble_score, 4),
            "is_anomaly":     is_anomaly,
            "severity":       _severity(ensemble_score),
            "top_features":   top_features,
            "narrative":      narrative,
            "ensemble_score": ensemble_score,
            "ae_score":       ae_score,
        })

    return output


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check. Returns model availability and feature store size."""
    return HealthResponse(
        status="ok",
        models_loaded={
            "isolation_forest": state.detector is not None,
            "autoencoder":      state.ae_detector is not None,
            "explainer":        state.explainer is not None,
        },
        feature_store=len(state.features_df) if state.features_df is not None else 0,
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score_by_user_id(req: ScoreRequest) -> ScoreResponse:
    """
    Score a single user by user_id, looking up their features from the
    pre-computed feature store (data/features.parquet).

    Use this endpoint when the user already has historical data in the system.
    Use /predict when you want to supply raw feature values directly.
    """
    if state.features_df is None:
        raise HTTPException(status_code=503, detail="Feature store not loaded.")

    user_feats = state.features_df[state.features_df["user_id"] == req.user_id]
    if user_feats.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User '{req.user_id}' not found in feature store.",
        )

    results = _score_and_explain(user_feats)
    r = results[0]
    return ScoreResponse(
        status="ok",
        **{k: r[k] for k in ScoreResponse.model_fields if k in r},
        message="Scored from feature store.",
    )


@app.post("/predict", response_model=ScoreResponse, tags=["scoring"])
def predict(feature_vec: FeatureVector) -> ScoreResponse:
    """
    Score a user from a raw feature vector supplied in the request body.

    This endpoint bypasses the feature store — useful for:
      - Scoring a new user in real-time before their data is persisted.
      - API testing with synthetic feature values.
      - Integration with external pipelines that compute features elsewhere.

    All feature fields are optional and default to 0 (same behaviour as the
    training pipeline's fill_value=0 strategy for missing columns).
    """
    features_df = pd.DataFrame([feature_vec.to_series()])
    results     = _score_and_explain(features_df)
    r           = results[0]
    return ScoreResponse(
        status="ok",
        **{k: r[k] for k in ScoreResponse.model_fields if k in r},
        message="Scored from provided feature vector.",
    )


@app.post("/score/batch", tags=["scoring"])
def score_batch(req: BatchScoreRequest) -> dict[str, Any]:
    """
    Score multiple users by user_id in a single request.
    Returns a list of ScoreResponse-shaped dicts plus a summary.
    Max 500 users per call.
    """
    if state.features_df is None:
        raise HTTPException(status_code=503, detail="Feature store not loaded.")

    found = state.features_df[state.features_df["user_id"].isin(req.user_ids)]
    missing = list(set(req.user_ids) - set(found["user_id"].tolist()))

    if found.empty:
        return {
            "status":  "ok",
            "results": [],
            "missing": missing,
            "message": "None of the supplied user_ids were found in the feature store.",
        }

    results = _score_and_explain(found)
    return {
        "status":   "ok",
        "results":  results,
        "missing":  missing,
        "summary": {
            "total_scored":   len(results),
            "total_anomalous": sum(1 for r in results if r["is_anomaly"]),
            "not_found":      len(missing),
        },
    }


@app.get("/alerts", tags=["alerts"])
def get_alerts(
    severity:  Optional[str] = Query(None, description="Filter by severity: LOW/MEDIUM/HIGH/CRITICAL"),
    limit:     int            = Query(100, ge=1, le=1000),
    offset:    int            = Query(0, ge=0),
    min_score: float          = Query(0.0, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """
    Retrieve generated alerts with optional severity and score filters.
    Alerts are loaded from data/alerts.jsonl (batch mode).
    """
    if state.alert_sys is None:
        raise HTTPException(status_code=503, detail="AlertSystem not initialised.")

    alerts = state.alert_sys.load_alerts()
    if not alerts:
        return {"status": "ok", "total": 0, "alerts": [], "message": "No alerts found."}

    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity.upper()]
    if min_score > 0:
        alerts = [a for a in alerts if a.get("risk_score", 0) >= min_score]

    alerts.sort(key=lambda a: a.get("risk_score", 0), reverse=True)
    page = alerts[offset: offset + limit]

    return {
        "status":  "ok",
        "total":   len(alerts),
        "offset":  offset,
        "limit":   limit,
        "alerts":  page,
    }


@app.get("/alerts/{alert_id}", tags=["alerts"])
def get_alert(alert_id: str) -> dict[str, Any]:
    """Retrieve a single alert by its alert_id."""
    if state.alert_sys is None:
        raise HTTPException(status_code=503, detail="AlertSystem not initialised.")

    alerts = state.alert_sys.load_alerts()
    match  = next((a for a in alerts if a.get("alert_id") == alert_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")

    return {"status": "ok", "alert": match}


@app.get("/users/{user_id}/risk", tags=["users"])
def user_risk_profile(user_id: str) -> dict[str, Any]:
    """
    Full risk profile for a single user:
      - Current anomaly score from both models
      - All triggered rules with excess ratios
      - All alerts ever generated for this user
      - Key feature values
    """
    if state.features_df is None:
        raise HTTPException(status_code=503, detail="Feature store not loaded.")

    user_feats = state.features_df[state.features_df["user_id"] == user_id]
    if user_feats.empty:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    scores  = _score_and_explain(user_feats)
    s       = scores[0]

    # All alerts for this user
    user_alerts: list[dict] = []
    if state.alert_sys is not None:
        user_alerts = [
            a for a in state.alert_sys.load_alerts()
            if a.get("user_id") == user_id
        ]
        user_alerts.sort(key=lambda a: a.get("risk_score", 0), reverse=True)

    # Key feature snapshot (only MODEL_FEATURES that exist)
    row          = user_feats.iloc[0]
    feature_snap = {
        feat: round(float(row[feat]), 4)
        for feat in MODEL_FEATURES
        if feat in row.index and not pd.isna(row[feat])
    }

    return {
        "status":        "ok",
        "user_id":       user_id,
        "anomaly_score": s["anomaly_score"],
        "is_anomaly":    s["is_anomaly"],
        "severity":      s["severity"],
        "narrative":     s["narrative"],
        "top_features":  s["top_features"],
        "ae_score":      s.get("ae_score"),
        "alerts":        user_alerts,
        "features":      feature_snap,
    }


@app.get("/", tags=["system"])
def root() -> dict:
    """API root — basic info."""
    return {
        "name":    "ForexGuard Anomaly Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": ["/score", "/predict", "/score/batch", "/alerts", "/users/{user_id}/risk"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
