"""
model.py  (v2 — extended MODEL_FEATURES)
-----------------------------------------
Unsupervised anomaly detection using Isolation Forest.

Changes in v2
-------------
MODEL_FEATURES expanded to include all features added in
feature_engineering.py v3:
  - night_login_ratio
  - min_inter_event_s
  - avg_inter_event_s
  - std_pnl
  - pnl_volatility_ratio
  - herfindahl_pairs
  - last_trade_size_zscore
  - last_margin_zscore
  - last_leverage_zscore
  - churn_financial_ratio
  - dormancy_withdrawal_score
  - withdrawal_surge_ratio
  - days_since_last_trade

Bug fixed (v1, retained)
------------------------
BUG 4 — contamination param was set to 0.08, exactly matching the true
anomaly fraction. Fix: contamination is now read from config as "auto",
decoupling the IF decision boundary from the true anomaly fraction.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# Features used for model input (v2 — extended)
MODEL_FEATURES = [
    # Portal — rate-normalised counts (per-day)
    "login_count_per_day",
    "deposit_count_per_day",
    "withdrawal_count_per_day",
    "support_ticket_count_per_day",
    "document_upload_count_per_day",
    "account_mod_count_per_day",
    "sensitive_mod_count_per_day",
    # Portal — ratio/rate features (scale-invariant)
    "unique_ips",
    "unique_regions",
    "geo_change_rate",
    "withdrawal_deposit_ratio",
    "avg_session_duration",
    "high_priority_ticket_rate",
    "unverified_doc_rate",
    # Portal — NEW: off-hours access
    "night_login_ratio",
    # Portal — NEW: inter-event timing (bot detection)
    "avg_inter_event_s",
    "min_inter_event_s",
    # Portal — NEW: withdrawal surge
    "withdrawal_surge_ratio",
    # Trading — scale-invariant or already rate-based
    "avg_trade_size",
    "max_trade_size",
    "std_trade_size",
    "avg_leverage",
    "max_leverage",
    "avg_margin_usage",
    "max_margin_usage",
    "margin_spike_rate",
    "trade_frequency_per_hour",
    "max_trades_5min",
    "unique_pairs",
    "total_pnl",
    # Trading — NEW: PnL volatility
    "std_pnl",
    "pnl_volatility_ratio",
    # Trading — NEW: instrument concentration
    "herfindahl_pairs",
    # Trading — NEW: within-user behavioural z-scores
    "last_trade_size_zscore",
    "last_margin_zscore",
    "last_leverage_zscore",
    # Cross-domain — NEW: churn / dormancy patterns
    "churn_financial_ratio",
    "dormancy_withdrawal_score",
    "days_since_last_trade",
]


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector with RobustScaler pre-processing.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["model"]

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[RobustScaler]   = None
        self.feature_names: list[str]         = MODEL_FEATURES
        self.model_path  = Path(self.cfg["model_path"])
        self.scaler_path = Path(self.cfg["scaler_path"])

    def _prepare_X(self, features_df: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.feature_names if c in features_df.columns]
        missing   = set(self.feature_names) - set(available)
        if missing:
            logger.warning("Missing features (filled with 0): %s", missing)
        X = features_df.reindex(columns=self.feature_names, fill_value=0).values
        return X.astype(float)

    def train(self, features_df: pd.DataFrame) -> "AnomalyDetector":
        X = self._prepare_X(features_df)
        logger.info("Training Isolation Forest on feature matrix %s …", X.shape)

        self.scaler  = RobustScaler()
        X_scaled     = self.scaler.fit_transform(X)

        # BUG 4 FIX: contamination comes from config (set to "auto")
        contamination = self.cfg["contamination"]

        self.model = IsolationForest(
            n_estimators=self.cfg["n_estimators"],
            contamination=contamination,
            max_samples=self.cfg["max_samples"],
            random_state=self.cfg["random_state"],
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        logger.info(
            "Isolation Forest training complete. contamination=%s", contamination
        )
        return self

    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X           = self._prepare_X(features_df)
        X_scaled    = self.scaler.transform(X)
        raw_scores  = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)

        score_min, score_max = raw_scores.min(), raw_scores.max()
        if score_max > score_min:
            normalised = 1 - (raw_scores - score_min) / (score_max - score_min)
        else:
            normalised = np.zeros_like(raw_scores)

        results = features_df[["user_id"]].copy()
        results["anomaly_score"] = np.round(normalised, 4)
        results["is_anomaly"]    = predictions == -1
        results["raw_if_score"]  = np.round(raw_scores, 4)

        logger.info(
            "IF scored %d users — anomalies detected: %d",
            len(results), results["is_anomaly"].sum()
        )
        return results

    def score_single(self, feature_row: pd.Series) -> dict:
        df     = pd.DataFrame([feature_row])
        result = self.score(df)
        return result.iloc[0].to_dict()

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info("IF Model saved → %s", self.model_path)

    def load(self) -> "AnomalyDetector":
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info("IF Model loaded from %s", self.model_path)
        return self
