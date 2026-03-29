"""
alert_system.py
---------------
Generates, formats, and persists structured anomaly alerts.

Alert schema
------------
{
    "alert_id"          : str,
    "user_id"           : str,
    "risk_score"        : float,
    "alert"             : str,          # human-readable summary
    "features_triggered": list[str],
    "triggered_rules"   : list[dict],
    "timestamp"         : str (ISO 8601),
    "severity"          : str           # LOW / MEDIUM / HIGH / CRITICAL
}
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _severity(risk_score: float) -> str:
    if risk_score >= 0.90:
        return "CRITICAL"
    if risk_score >= 0.75:
        return "HIGH"
    if risk_score >= 0.60:
        return "MEDIUM"
    return "LOW"


class AlertSystem:
    """
    Converts anomaly detection results into structured alerts and persists them.

    Parameters
    ----------
    config_path : str
        Path to YAML config.
    output_path : str, optional
        Override output file path (defaults to config value).
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        output_path: Optional[str] = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.threshold: float = cfg["alerts"]["risk_score_threshold"]
        self.output_path = Path(output_path or cfg["alerts"]["output_path"])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear previous alerts file
        self.output_path.write_text("")

    # ------------------------------------------------------------------
    # Single alert
    # ------------------------------------------------------------------

    def _build_alert(self, explanation_row: dict) -> dict:
        """Construct one alert dict from an explanation row."""
        risk_score = float(explanation_row.get("anomaly_score", 0.0))
        narrative = explanation_row.get("narrative", "Anomaly detected.")
        features = explanation_row.get("features_triggered", [])
        rules = explanation_row.get("triggered_rules", [])

        return {
            "alert_id": str(uuid.uuid4()),
            "user_id": explanation_row["user_id"],
            "risk_score": round(risk_score, 4),
            "severity": _severity(risk_score),
            "alert": narrative,
            "features_triggered": features,
            "triggered_rules": rules,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def generate_alerts(self, explained_df: pd.DataFrame) -> list[dict]:
        """
        Generate alerts for all users above the risk threshold.

        Parameters
        ----------
        explained_df : pd.DataFrame
            Output of RuleBasedExplainer.explain_batch()

        Returns
        -------
        alerts : list[dict]
        """
        filtered = explained_df[explained_df["anomaly_score"] >= self.threshold]
        filtered = filtered.sort_values("anomaly_score", ascending=False)

        alerts = []
        for _, row in filtered.iterrows():
            alert = self._build_alert(row.to_dict())
            alerts.append(alert)
            self._write_alert(alert)
            self._log_alert(alert)

        logger.info("Generated %d alerts (threshold ≥ %.2f).", len(alerts), self.threshold)
        return alerts

    # ------------------------------------------------------------------
    # Single-event alert (streaming mode)
    # ------------------------------------------------------------------

    def generate_single_alert(
        self,
        user_id: str,
        risk_score: float,
        explanation: dict,
    ) -> Optional[dict]:
        """
        Create and emit one alert for a streaming event.

        Returns None if score is below threshold.
        """
        if risk_score < self.threshold:
            return None

        row = {
            "user_id": user_id,
            "anomaly_score": risk_score,
            **explanation,
        }
        alert = self._build_alert(row)
        self._write_alert(alert)
        self._log_alert(alert)
        return alert

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _write_alert(self, alert: dict) -> None:
        """Append alert as a JSON line."""
        with open(self.output_path, "a") as f:
            f.write(json.dumps(alert) + "\n")

    @staticmethod
    def _log_alert(alert: dict) -> None:
        sev = alert["severity"]
        uid = alert["user_id"]
        score = alert["risk_score"]
        msg = alert["alert"]
        logger.warning("[%s] user=%s score=%.4f | %s", sev, uid, score, msg)

    # ------------------------------------------------------------------
    # Reading alerts back
    # ------------------------------------------------------------------

    def load_alerts(self) -> list[dict]:
        """Load all alerts from the output file."""
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return []
        alerts = []
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return alerts

    def alerts_to_dataframe(self) -> pd.DataFrame:
        """Return alerts as a DataFrame."""
        alerts = self.load_alerts()
        if not alerts:
            return pd.DataFrame()
        df = pd.DataFrame(alerts)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("risk_score", ascending=False).reset_index(drop=True)
