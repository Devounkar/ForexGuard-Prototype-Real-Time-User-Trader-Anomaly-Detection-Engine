"""
explainability.py  (v2 — extended rule set)
--------------------------------------------
Rule-based explanation engine for anomaly alerts.

New in v2
---------
Added rule definitions for all features introduced in feature_engineering.py v3:
  - night_login_ratio          (off-hours access pattern)
  - min_inter_event_s          (bot-like rapid-fire events)
  - std_pnl                    (PnL volatility)
  - pnl_volatility_ratio       (scale-invariant PnL vol)
  - herfindahl_pairs           (single-instrument concentration)
  - last_trade_size_zscore     (within-user sudden trade spike)
  - last_margin_zscore         (within-user sudden margin spike)
  - churn_financial_ratio      (deposit→no trade→withdrawal)
  - dormancy_withdrawal_score  (dormancy + large withdrawal)
  - withdrawal_surge_ratio     (sudden large withdrawal vs baseline)

For each anomalous user, compares their feature values against
population percentiles to generate a human-readable explanation
without requiring SHAP.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

RULE_DEFINITIONS = [
    # ── Original rules ────────────────────────────────────────────────────
    {
        "feature":    "max_trades_5min",
        "percentile": 95,
        "label":      "high-frequency trading burst",
        "direction":  "high",
    },
    {
        "feature":    "max_margin_usage",
        "percentile": 90,
        "label":      "critically high margin usage",
        "direction":  "high",
    },
    {
        "feature":    "avg_margin_usage",
        "percentile": 90,
        "label":      "sustained elevated margin usage",
        "direction":  "high",
    },
    {
        "feature":    "margin_spike_rate",
        "percentile": 90,
        "label":      "frequent margin spikes",
        "direction":  "high",
    },
    {
        "feature":    "max_trade_size",
        "percentile": 95,
        "label":      "unusually large trade size",
        "direction":  "high",
    },
    {
        "feature":    "std_trade_size",
        "percentile": 95,
        "label":      "highly volatile trade sizing",
        "direction":  "high",
    },
    {
        "feature":    "trade_frequency_per_hour",
        "percentile": 95,
        "label":      "abnormally high trade frequency",
        "direction":  "high",
    },
    {
        "feature":    "unique_regions",
        "percentile": 90,
        "label":      "logins from multiple geographic regions",
        "direction":  "high",
    },
    {
        "feature":    "geo_change_rate",
        "percentile": 90,
        "label":      "rapid geographic location changes",
        "direction":  "high",
    },
    {
        "feature":    "withdrawal_deposit_ratio",
        "percentile": 90,
        "label":      "high withdrawal-to-deposit ratio",
        "direction":  "high",
    },
    {
        "feature":    "max_leverage",
        "percentile": 90,
        "label":      "extreme leverage usage",
        "direction":  "high",
    },
    {
        "feature":    "unique_ips",
        "percentile": 90,
        "label":      "access from many distinct IP addresses",
        "direction":  "high",
    },

    # ── NEW: Off-hours login ──────────────────────────────────────────────
    {
        "feature":    "night_login_ratio",
        "percentile": 90,
        "label":      "unusually high rate of off-hours logins (00:00–06:00 UTC)",
        "direction":  "high",
    },

    # ── NEW: Bot-like inter-event timing ──────────────────────────────────
    # LOW min_inter_event_s is the anomaly — machine-speed clicks
    {
        "feature":    "min_inter_event_s",
        "percentile": 5,
        "label":      "near-instant inter-event timing suggesting automated activity",
        "direction":  "low",
    },

    # ── NEW: PnL volatility ───────────────────────────────────────────────
    {
        "feature":    "std_pnl",
        "percentile": 95,
        "label":      "extreme PnL volatility across trades",
        "direction":  "high",
    },
    {
        "feature":    "pnl_volatility_ratio",
        "percentile": 95,
        "label":      "high PnL variance relative to trade volume",
        "direction":  "high",
    },

    # ── NEW: Single-instrument concentration ─────────────────────────────
    {
        "feature":    "herfindahl_pairs",
        "percentile": 90,
        "label":      "single-instrument trading concentration",
        "direction":  "high",
    },

    # ── NEW: Within-user behavioural drift (z-scores) ────────────────────
    {
        "feature":    "last_trade_size_zscore",
        "percentile": 95,
        "label":      "sudden spike in trade size vs user's own baseline",
        "direction":  "high",
    },
    {
        "feature":    "last_margin_zscore",
        "percentile": 95,
        "label":      "sudden spike in margin usage vs user's own baseline",
        "direction":  "high",
    },
    {
        "feature":    "last_leverage_zscore",
        "percentile": 95,
        "label":      "sudden spike in leverage vs user's own baseline",
        "direction":  "high",
    },

    # ── NEW: Churn-abuse pattern ──────────────────────────────────────────
    {
        "feature":    "churn_financial_ratio",
        "percentile": 95,
        "label":      "deposit/withdrawal activity disproportionate to trading volume",
        "direction":  "high",
    },

    # ── NEW: Dormancy + sudden large withdrawal ───────────────────────────
    {
        "feature":    "dormancy_withdrawal_score",
        "percentile": 90,
        "label":      "prolonged trading dormancy followed by large withdrawal",
        "direction":  "high",
    },
    {
        "feature":    "withdrawal_surge_ratio",
        "percentile": 90,
        "label":      "withdrawal amount far exceeds user's own historical average",
        "direction":  "high",
    },
]


class RuleBasedExplainer:
    """
    Generates human-readable explanations for anomalous users.

    Parameters
    ----------
    top_k : int
        Maximum number of triggered rules to include in the explanation.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self._thresholds: dict[str, float] = {}
        self._population_stats: Optional[pd.DataFrame] = None

    def fit(self, features_df: pd.DataFrame) -> "RuleBasedExplainer":
        """
        Compute population percentile thresholds from the full feature set.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full user-level feature matrix (normal + anomalous users).
        """
        self._population_stats = features_df.describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.90, 0.95]
        )
        for rule in RULE_DEFINITIONS:
            feat = rule["feature"]
            pct  = rule["percentile"]
            if feat in features_df.columns:
                self._thresholds[feat] = float(
                    np.percentile(features_df[feat].fillna(0), pct)
                )
        logger.info("Explainer fitted on %d users.", len(features_df))
        return self

    def explain(self, user_row: pd.Series) -> dict:
        """
        Generate an explanation for a single user.

        Parameters
        ----------
        user_row : pd.Series
            One row from the features DataFrame (with user_id).

        Returns
        -------
        explanation : dict
            {
                "triggered_rules": [...],
                "features_triggered": [...],
                "narrative": "..."
            }
        """
        triggered = []
        for rule in RULE_DEFINITIONS:
            feat = rule["feature"]
            if feat not in user_row.index:
                continue
            threshold = self._thresholds.get(feat)
            if threshold is None:
                continue
            val = float(user_row[feat]) if not pd.isna(user_row[feat]) else 0.0

            direction = rule["direction"]
            if direction == "high" and val > threshold:
                excess_ratio = (val - threshold) / max(abs(threshold), 1e-9)
                triggered.append({
                    "feature":      feat,
                    "value":        round(val, 4),
                    "threshold":    round(threshold, 4),
                    "label":        rule["label"],
                    "excess_ratio": round(excess_ratio, 3),
                })
            elif direction == "low" and val < threshold:
                # For "low" rules: excess_ratio = how far BELOW the threshold
                excess_ratio = (threshold - val) / max(abs(threshold), 1e-9)
                triggered.append({
                    "feature":      feat,
                    "value":        round(val, 4),
                    "threshold":    round(threshold, 4),
                    "label":        rule["label"],
                    "excess_ratio": round(excess_ratio, 3),
                })

        # Sort by how far from threshold the value is
        triggered.sort(key=lambda x: x["excess_ratio"], reverse=True)
        top = triggered[: self.top_k]

        if top:
            labels    = " and ".join(r["label"] for r in top)
            narrative = f"Anomalous behaviour detected: {labels}."
        else:
            narrative = (
                "Anomaly score elevated — pattern does not match normal user baseline."
            )

        return {
            "triggered_rules":    top,
            "features_triggered": [r["feature"] for r in top],
            "narrative":          narrative,
        }

    def explain_batch(
        self,
        features_df: pd.DataFrame,
        scores_df:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach explanations to all anomalous users.

        Parameters
        ----------
        features_df : pd.DataFrame
        scores_df   : pd.DataFrame  (output of AnomalyDetector.score)

        Returns
        -------
        explained : pd.DataFrame
        """
        anomalous = scores_df[scores_df["is_anomaly"]].copy()
        merged    = anomalous.merge(features_df, on="user_id", how="left")

        results = []
        for _, row in merged.iterrows():
            expl = self.explain(row)
            results.append({
                "user_id":      row["user_id"],
                "anomaly_score": row["anomaly_score"],
                **expl,
            })

        return pd.DataFrame(results)
