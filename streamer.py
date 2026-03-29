"""
streamer.py
-----------
Simulates a real-time event stream from the synthetic dataset.

Uses Python generators + time-delay simulation (Option B from the spec).
Can be swapped for a Kafka consumer by replacing `EventStreamer` with a
Kafka-backed implementation that shares the same interface.

Architecture
------------
  raw events (parquet)
       ↓
  EventStreamer  (generator, yields individual events in chrono order)
       ↓
  StreamingPipeline  (buffers events → triggers feature re-computation)
       ↓
  AnomalyDetector.score_single()
       ↓
  RuleBasedExplainer.explain()
       ↓
  AlertSystem.generate_single_alert()
"""

import logging
import time
from datetime import datetime
from typing import Generator, Optional

import pandas as pd
import yaml

from src.alert_system import AlertSystem
from src.explainability import RuleBasedExplainer
from src.feature_engineering import FeatureEngineer
from src.model import AnomalyDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level event streamer
# ---------------------------------------------------------------------------

class EventStreamer:
    """
    Iterates over pre-sorted portal + trading events in chronological order,
    simulating a real-time stream.

    Parameters
    ----------
    portal_df : pd.DataFrame
    trades_df : pd.DataFrame
    delay_ms  : int
        Artificial delay between events (milliseconds).  Set to 0 for
        maximum throughput (batch scoring mode).
    """

    def __init__(
        self,
        portal_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        delay_ms: int = 0,
    ):
        # Tag source
        portal_df = portal_df.copy()
        portal_df["_source"] = "portal"
        trades_df = trades_df.copy()
        trades_df["_source"] = "trade"

        # Merge and sort
        common = ["user_id", "timestamp", "_source"]
        self._stream = (
            pd.concat(
                [portal_df.assign(**{c: None for c in trades_df.columns if c not in portal_df}),
                 trades_df.assign(**{c: None for c in portal_df.columns if c not in trades_df})],
                ignore_index=True,
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self.delay_s = delay_ms / 1000.0
        self._total = len(self._stream)

    def stream(self) -> Generator[dict, None, None]:
        """Yield events one at a time."""
        for _, row in self._stream.iterrows():
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            yield row.to_dict()

    def __len__(self) -> int:
        return self._total


# ---------------------------------------------------------------------------
# High-level streaming pipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """
    Orchestrates the end-to-end stream: ingest → feature buffer →
    model scoring → explanation → alert generation.

    The pipeline maintains a rolling buffer of events per user.
    After every `score_every_n` events, it re-scores all users
    whose data changed since the last scoring cycle.

    Parameters
    ----------
    config_path : str
    detector    : AnomalyDetector  (pre-trained)
    explainer   : RuleBasedExplainer  (pre-fitted)
    alert_sys   : AlertSystem
    score_every_n : int
        How many events to ingest before triggering a scoring round.
    delay_ms    : int
        Delay between streamed events (0 = as fast as possible).
    max_events  : int, optional
        Stop after this many events (useful for demos / tests).
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        detector: Optional[AnomalyDetector] = None,
        explainer: Optional[RuleBasedExplainer] = None,
        alert_sys: Optional[AlertSystem] = None,
        score_every_n: int = 500,
        delay_ms: int = 0,
        max_events: Optional[int] = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.score_every_n = score_every_n
        self.delay_ms = delay_ms
        self.max_events = max_events

        self.detector = detector or AnomalyDetector(config_path).load()
        self.explainer = explainer
        self.alert_sys = alert_sys or AlertSystem(config_path)
        self._fe = FeatureEngineer()

        # Rolling buffers
        self._portal_buf: list[dict] = []
        self._trade_buf: list[dict] = []
        self._dirty_users: set[str] = set()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _ingest(self, event: dict) -> None:
        source = event.get("_source", "")
        uid = event.get("user_id", "UNKNOWN")
        self._dirty_users.add(uid)

        if source == "portal":
            self._portal_buf.append(event)
        elif source == "trade":
            self._trade_buf.append(event)

    # ------------------------------------------------------------------
    # Scoring cycle
    # ------------------------------------------------------------------

    def _score_cycle(self) -> list[dict]:
        """Compute features + scores for all buffered users."""
        if not self._portal_buf and not self._trade_buf:
            return []

        portal_df = pd.DataFrame(self._portal_buf) if self._portal_buf else pd.DataFrame()
        trade_df = pd.DataFrame(self._trade_buf) if self._trade_buf else pd.DataFrame()

        # Fill missing columns
        if portal_df.empty:
            portal_df = pd.DataFrame(columns=[
                "user_id", "timestamp", "event_type", "ip_address",
                "device_type", "region", "session_duration_s", "amount_usd",
            ])
        if trade_df.empty:
            trade_df = pd.DataFrame(columns=[
                "trade_id", "user_id", "timestamp", "currency_pair", "trade_size",
                "lot_size", "leverage", "margin_used", "margin_usage_pct",
                "trade_duration_s", "pnl_usd", "is_anomalous",
            ])

        features_df = self._fe.build_features(portal_df, trade_df)
        scores_df = self.detector.score(features_df)

        alerts_generated = []
        if self.explainer:
            explained = self.explainer.explain_batch(features_df, scores_df)
            for _, row in explained.iterrows():
                alert = self.alert_sys.generate_single_alert(
                    user_id=row["user_id"],
                    risk_score=row["anomaly_score"],
                    explanation={
                        "narrative": row["narrative"],
                        "features_triggered": row["features_triggered"],
                        "triggered_rules": row["triggered_rules"],
                    },
                )
                if alert:
                    alerts_generated.append(alert)

        self._dirty_users.clear()
        return alerts_generated

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        portal_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> list[dict]:
        """
        Run the streaming pipeline over the provided datasets.

        Parameters
        ----------
        portal_df : pd.DataFrame
        trades_df : pd.DataFrame

        Returns
        -------
        all_alerts : list[dict]
        """
        streamer = EventStreamer(portal_df, trades_df, delay_ms=self.delay_ms)
        total = len(streamer)
        all_alerts: list[dict] = []
        n = 0

        logger.info("Starting stream — %d events total …", total)

        for event in streamer.stream():
            self._ingest(event)
            n += 1

            if n % self.score_every_n == 0:
                logger.info("  [%d / %d events] — scoring cycle …", n, total)
                alerts = self._score_cycle()
                all_alerts.extend(alerts)

            if self.max_events and n >= self.max_events:
                logger.info("Reached max_events=%d — stopping.", self.max_events)
                break

        # Final scoring pass
        logger.info("Final scoring pass …")
        alerts = self._score_cycle()
        all_alerts.extend(alerts)

        logger.info("Stream complete. Total alerts: %d", len(all_alerts))
        return all_alerts
