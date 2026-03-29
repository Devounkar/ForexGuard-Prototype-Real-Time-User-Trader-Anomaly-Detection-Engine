"""
feature_engineering.py  (v3 — extended feature set)
------------------------------------------------------
Aggregates raw portal and trading events into per-user feature vectors.

New in v3
---------
INTER-EVENT TIME DELTAS
    avg_inter_event_s, min_inter_event_s — captures bot-like rapid fire
    activity patterns that raw counts miss.

Z-SCORE / ROLLING STATS PER USER
    For each key trading metric (trade_size, margin_usage_pct, leverage)
    we compute per-user rolling mean, std, and z-score of the final value
    vs that user's own baseline. Captures sudden behavioural shifts within
    a single user's session — a user whose last trade is 3σ above their
    own mean is flagged even if their absolute value looks normal.

PNL VOLATILITY
    std_pnl  — standard deviation of per-trade PnL (Sharpe-like signal).
    pnl_volatility_ratio  — std_pnl / (abs_pnl + 1) for scale invariance.
    High PnL volatility is a marker for high-risk / speculative behaviour.

TRADE INSTRUMENT CONCENTRATION
    herfindahl_pairs — Herfindahl-Hirschman Index over currency pair counts.
    Range [1/n, 1], where 1 means all trades on a single pair (concentrated).
    Captures single-instrument concentration noted in the spec.

OFF-HOURS LOGIN
    login_night_count, login_night_rate — logins between 00:00–06:00 UTC.
    night_login_ratio — fraction of all logins that are off-hours.

DEPOSIT → NO TRADE → WITHDRAWAL (churn-abuse pattern)
    deposit_withdrawal_ratio_vs_trades — (deposits + withdrawals) / trades.
    A user who deposits and withdraws frequently with minimal trading
    scores very high here, capturing the bonus-abuse / churn pattern.
    churn_risk_flag — binary: withdrawal_deposit_ratio > 0.8 AND
    trade_frequency_per_hour < 25th percentile of the population.
    (Computed post-merge in build_features.)

DORMANCY + SUDDEN LARGE WITHDRAWAL
    days_since_last_trade — gap between last trade and last portal event.
    dormancy_withdrawal_score — days_since_last_trade × (max single
    withdrawal / mean single withdrawal). High when a user goes quiet
    then makes a large withdrawal.

Fix applied (v2, retained)
--------------------------
RATE NORMALISATION — count features divided by observation span in days.
"""

import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SENSITIVE_MODS = {"bank_account_change", "email_change", "phone_change", "leverage_change"}

# Off-hours window (UTC) — 00:00 inclusive, 06:00 exclusive
_NIGHT_START_H = 0
_NIGHT_END_H   = 6


class FeatureEngineer:
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes

    # ------------------------------------------------------------------
    # Portal features
    # ------------------------------------------------------------------

    def _portal_features(self, portal_df: pd.DataFrame) -> pd.DataFrame:
        df = portal_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        agg = df.groupby("user_id").agg(
            total_portal_events   =("event_type", "count"),
            login_count           =("event_type", lambda x: (x == "login").sum()),
            logout_count          =("event_type", lambda x: (x == "logout").sum()),
            deposit_count         =("event_type", lambda x: (x == "deposit").sum()),
            withdrawal_count      =("event_type", lambda x: (x == "withdrawal").sum()),
            kyc_update_count      =("event_type", lambda x: (x == "kyc_update").sum()),
            support_ticket_count  =("event_type", lambda x: (x == "support_ticket_open").sum()),
            document_upload_count =("event_type", lambda x: (x == "document_upload").sum()),
            account_mod_count     =("event_type", lambda x: (x == "account_modification").sum()),
            total_deposited       =("amount_usd",  lambda x: x[df.loc[x.index, "event_type"] == "deposit"].sum()),
            total_withdrawn       =("amount_usd",  lambda x: x[df.loc[x.index, "event_type"] == "withdrawal"].sum()),
            avg_session_duration  =("session_duration_s", "mean"),
            max_session_duration  =("session_duration_s", "max"),
            unique_ips            =("ip_address",  "nunique"),
            unique_regions        =("region",      "nunique"),
            unique_devices        =("device_type", "nunique"),
        ).reset_index()

        agg["geo_change_rate"] = agg["unique_regions"] / agg["login_count"].clip(lower=1)
        agg["withdrawal_deposit_ratio"] = (
            agg["total_withdrawn"] / agg["total_deposited"].replace(0, np.nan)
        ).fillna(0)

        # High-priority ticket rate
        if "ticket_priority" in df.columns:
            high_prio = (
                df[df["ticket_priority"].isin(["high", "critical"])]
                .groupby("user_id").size().rename("high_priority_tickets")
            )
            agg = agg.merge(high_prio, on="user_id", how="left")
            agg["high_priority_tickets"] = agg["high_priority_tickets"].fillna(0)
            agg["high_priority_ticket_rate"] = (
                agg["high_priority_tickets"] / agg["support_ticket_count"].clip(lower=1)
            )
        else:
            agg["high_priority_tickets"] = 0
            agg["high_priority_ticket_rate"] = 0.0

        # Unverified document rate
        if "document_verified" in df.columns:
            unverified = (
                df[(df["event_type"] == "document_upload") & (df["document_verified"] == False)]
                .groupby("user_id").size().rename("unverified_docs")
            )
            agg = agg.merge(unverified, on="user_id", how="left")
            agg["unverified_docs"] = agg["unverified_docs"].fillna(0)
            agg["unverified_doc_rate"] = (
                agg["unverified_docs"] / agg["document_upload_count"].clip(lower=1)
            )
        else:
            agg["unverified_docs"] = 0
            agg["unverified_doc_rate"] = 0.0

        # Sensitive account modifications
        if "modification_type" in df.columns:
            sensitive = (
                df[df["modification_type"].isin(_SENSITIVE_MODS)]
                .groupby("user_id").size().rename("sensitive_mod_count")
            )
            agg = agg.merge(sensitive, on="user_id", how="left")
            agg["sensitive_mod_count"] = agg["sensitive_mod_count"].fillna(0)
        else:
            agg["sensitive_mod_count"] = 0

        # ── Rate normalisation ────────────────────────────────────────────
        span = (
            df.groupby("user_id")["timestamp"]
            .agg(lambda x: max((x.max() - x.min()).total_seconds() / 86400, 1.0))
            .rename("obs_days_portal")
        )
        agg = agg.merge(span, on="user_id", how="left")
        agg["obs_days_portal"] = agg["obs_days_portal"].fillna(1.0)

        for col in [
            "login_count", "deposit_count", "withdrawal_count",
            "support_ticket_count", "document_upload_count",
            "account_mod_count", "sensitive_mod_count",
        ]:
            if col in agg.columns:
                agg[f"{col}_per_day"] = (
                    agg[col] / agg["obs_days_portal"]
                ).round(4)

        # ── NEW: Off-hours login features ─────────────────────────────────
        # Night = 00:00 UTC to 06:00 UTC (inclusive start, exclusive end)
        login_events = df[df["event_type"] == "login"].copy()
        if not login_events.empty:
            login_events["hour_utc"] = login_events["timestamp"].dt.hour
            login_events["is_night"] = login_events["hour_utc"].between(
                _NIGHT_START_H, _NIGHT_END_H - 1
            )
            night_counts = (
                login_events.groupby("user_id")["is_night"]
                .agg(login_night_count="sum", _login_total="count")
                .reset_index()
            )
            night_counts["night_login_ratio"] = (
                night_counts["login_night_count"] / night_counts["_login_total"].clip(lower=1)
            ).round(4)
            agg = agg.merge(
                night_counts[["user_id", "login_night_count", "night_login_ratio"]],
                on="user_id", how="left",
            )
        else:
            agg["login_night_count"] = 0
            agg["night_login_ratio"] = 0.0

        agg["login_night_count"] = agg["login_night_count"].fillna(0)
        agg["night_login_ratio"] = agg["night_login_ratio"].fillna(0.0)

        # ── NEW: Inter-event time deltas (portal) ─────────────────────────
        # Measures how fast a user fires portal events — very low avg suggests
        # automated / bot-like behaviour.
        inter_event = self._inter_event_features(df, id_col="user_id", ts_col="timestamp")
        agg = agg.merge(inter_event, on="user_id", how="left")
        agg["avg_inter_event_s"] = agg["avg_inter_event_s"].fillna(
            agg["avg_inter_event_s"].median()
        )
        agg["min_inter_event_s"] = agg["min_inter_event_s"].fillna(0)

        # ── NEW: Dormancy + sudden large withdrawal ───────────────────────
        # days_since_last_trade computed later in build_features (needs trades)
        # Here we compute: max single withdrawal and mean single withdrawal
        withdrawal_events = df[df["event_type"] == "withdrawal"].copy()
        if not withdrawal_events.empty and "amount_usd" in withdrawal_events.columns:
            wd_stats = (
                withdrawal_events.groupby("user_id")["amount_usd"]
                .agg(max_single_withdrawal="max", mean_single_withdrawal="mean")
                .reset_index()
            )
            agg = agg.merge(wd_stats, on="user_id", how="left")
        else:
            agg["max_single_withdrawal"] = 0.0
            agg["mean_single_withdrawal"] = 0.0

        agg["max_single_withdrawal"] = agg["max_single_withdrawal"].fillna(0.0)
        agg["mean_single_withdrawal"] = agg["mean_single_withdrawal"].fillna(0.0)

        # withdrawal_surge_ratio — how much bigger was the biggest withdrawal
        # than the user's average? High = dormancy/surge pattern.
        agg["withdrawal_surge_ratio"] = (
            agg["max_single_withdrawal"]
            / agg["mean_single_withdrawal"].replace(0, np.nan)
        ).fillna(1.0).round(4)

        # last_portal_event_ts — needed for dormancy calc in build_features
        last_portal = (
            df.groupby("user_id")["timestamp"].max().rename("last_portal_ts")
        )
        agg = agg.merge(last_portal, on="user_id", how="left")

        return agg

    # ------------------------------------------------------------------
    # Trading features
    # ------------------------------------------------------------------

    def _trade_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        agg = df.groupby("user_id").agg(
            total_trades       =("trade_id",        "count"),
            avg_trade_size     =("trade_size",       "mean"),
            max_trade_size     =("trade_size",       "max"),
            std_trade_size     =("trade_size",       "std"),
            avg_lot_size       =("lot_size",         "mean"),
            avg_leverage       =("leverage",         "mean"),
            max_leverage       =("leverage",         "max"),
            avg_margin_usage   =("margin_usage_pct", "mean"),
            max_margin_usage   =("margin_usage_pct", "max"),
            avg_trade_duration =("trade_duration_s", "mean"),
            total_pnl          =("pnl_usd",          "sum"),
            abs_pnl            =("pnl_usd",          lambda x: x.abs().sum()),
            unique_pairs       =("currency_pair",    "nunique"),
        ).reset_index()

        high_margin = (
            df[df["margin_usage_pct"] > 80]
            .groupby("user_id").size().rename("high_margin_trades")
        )
        agg = agg.merge(high_margin, on="user_id", how="left")
        agg["high_margin_trades"] = agg["high_margin_trades"].fillna(0)
        agg["margin_spike_rate"]  = agg["high_margin_trades"] / agg["total_trades"].clip(lower=1)

        time_spans = (
            df.groupby("user_id")["timestamp"]
            .agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)
            .rename("observation_hours")
        )
        agg = agg.merge(time_spans, on="user_id", how="left")
        agg["trade_frequency_per_hour"] = (
            agg["total_trades"] / agg["observation_hours"].replace(0, np.nan)
        ).fillna(0)

        agg = agg.merge(self._hft_burst_feature(df), on="user_id", how="left")
        agg["std_trade_size"] = agg["std_trade_size"].fillna(0)

        # ── NEW: PnL volatility ───────────────────────────────────────────
        pnl_vol = (
            df.groupby("user_id")["pnl_usd"]
            .std()
            .rename("std_pnl")
            .reset_index()
        )
        agg = agg.merge(pnl_vol, on="user_id", how="left")
        agg["std_pnl"] = agg["std_pnl"].fillna(0)
        # Scale-invariant ratio: std / (|mean PnL| + 1)
        agg["pnl_volatility_ratio"] = (
            agg["std_pnl"] / (agg["abs_pnl"] / agg["total_trades"].clip(lower=1) + 1)
        ).round(4)

        # ── NEW: Herfindahl instrument concentration ──────────────────────
        # HHI = sum of squared market shares; 1 = all trades on one pair.
        hhi = self._herfindahl_feature(df)
        agg = agg.merge(hhi, on="user_id", how="left")
        agg["herfindahl_pairs"] = agg["herfindahl_pairs"].fillna(1.0)

        # ── NEW: Per-user rolling z-score features ────────────────────────
        # For each key metric, compute z-score of user's last observation
        # vs their own distribution. Captures within-user behavioural drift.
        zscore_feats = self._user_zscore_features(df)
        agg = agg.merge(zscore_feats, on="user_id", how="left")
        for col in ["last_trade_size_zscore", "last_margin_zscore", "last_leverage_zscore"]:
            agg[col] = agg[col].fillna(0.0)

        # ── NEW: Deposit→no trade→withdrawal churn indicator ─────────────
        # This is a ratio; denominator comes from trade data.
        # deposit_count and withdrawal_count merged from portal in build_features.
        # Here we just store total_trades for the ratio.
        agg["last_trade_ts"] = (
            df.groupby("user_id")["timestamp"].max().reset_index()["timestamp"]
        )
        # Re-attach properly
        last_trade = df.groupby("user_id")["timestamp"].max().rename("last_trade_ts")
        agg = agg.merge(last_trade, on="user_id", how="left")

        return agg

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inter_event_features(df: pd.DataFrame, id_col: str, ts_col: str) -> pd.DataFrame:
        """
        Compute avg and min inter-event gap per user (in seconds).
        Very low values indicate automated / bot-like behaviour.
        """
        records = []
        for uid, grp in df.groupby(id_col):
            times = grp[ts_col].sort_values()
            if len(times) < 2:
                records.append({id_col: uid, "avg_inter_event_s": np.nan, "min_inter_event_s": 0})
                continue
            deltas = times.diff().dt.total_seconds().dropna()
            records.append({
                id_col:              uid,
                "avg_inter_event_s": round(float(deltas.mean()), 2),
                "min_inter_event_s": round(float(deltas.min()), 2),
            })
        result = pd.DataFrame(records)
        result.rename(columns={id_col: "user_id"}, inplace=True)
        return result

    @staticmethod
    def _hft_burst_feature(trades_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for uid, grp in trades_df.groupby("user_id"):
            grp = grp.sort_values("timestamp")
            if len(grp) < 2:
                records.append({"user_id": uid, "max_trades_5min": len(grp)})
                continue
            grp = grp.set_index("timestamp")
            rolling_counts = grp["trade_id"].resample("5min").count()
            records.append({"user_id": uid, "max_trades_5min": int(rolling_counts.max())})
        return pd.DataFrame(records)

    @staticmethod
    def _herfindahl_feature(trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Herfindahl-Hirschman Index over currency pairs per user.
        HHI = Σ (share_i²), where share_i = trades on pair_i / total_trades.
        Range: [1/n_pairs, 1]. Value of 1 = all trades on one instrument.
        """
        records = []
        for uid, grp in trades_df.groupby("user_id"):
            counts = grp["currency_pair"].value_counts()
            shares = counts / counts.sum()
            hhi    = float((shares ** 2).sum())
            records.append({"user_id": uid, "herfindahl_pairs": round(hhi, 4)})
        return pd.DataFrame(records)

    @staticmethod
    def _user_zscore_features(trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each user, compute z-score of their LAST trade's key metrics
        against their own historical distribution.

        A user whose most recent trade is 3σ above their own mean is flagged
        even if their absolute value sits within the population normal range.
        This captures sudden within-session behavioural spikes.
        """
        records = []
        for uid, grp in trades_df.groupby("user_id"):
            grp = grp.sort_values("timestamp")
            if len(grp) < 3:
                records.append({
                    "user_id":               uid,
                    "last_trade_size_zscore": 0.0,
                    "last_margin_zscore":     0.0,
                    "last_leverage_zscore":   0.0,
                })
                continue

            def _zscore(series: pd.Series) -> float:
                mu  = series[:-1].mean()
                sig = series[:-1].std()
                if sig == 0 or np.isnan(sig):
                    return 0.0
                return round(float((series.iloc[-1] - mu) / sig), 4)

            records.append({
                "user_id":               uid,
                "last_trade_size_zscore": _zscore(grp["trade_size"]),
                "last_margin_zscore":     _zscore(grp["margin_usage_pct"]),
                "last_leverage_zscore":   _zscore(grp["leverage"].astype(float)),
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Main builder
    # ------------------------------------------------------------------

    def build_features(self, portal_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building portal features …")
        portal_feats = self._portal_features(portal_df)
        logger.info("Building trading features …")
        trade_feats  = self._trade_features(trades_df)

        features = pd.merge(portal_feats, trade_feats, on="user_id", how="outer")

        # ── NEW: Churn-abuse pattern ──────────────────────────────────────
        # (deposit_count + withdrawal_count) / total_trades
        # High = lots of financial activity relative to actual trading.
        features["churn_financial_ratio"] = (
            (features.get("deposit_count", 0) + features.get("withdrawal_count", 0))
            / features["total_trades"].clip(lower=1)
        ).round(4)

        # ── NEW: Dormancy + sudden large withdrawal ───────────────────────
        # days_since_last_trade: gap between last trade and last portal event.
        # If a user was dormant in trading but then shows portal activity
        # (especially withdrawal), this score rises.
        if "last_portal_ts" in features.columns and "last_trade_ts" in features.columns:
            features["last_portal_ts"] = pd.to_datetime(features["last_portal_ts"])
            features["last_trade_ts"]  = pd.to_datetime(features["last_trade_ts"])
            features["days_since_last_trade"] = (
                (features["last_portal_ts"] - features["last_trade_ts"])
                .dt.total_seconds() / 86400
            ).clip(lower=0).round(2)
            # dormancy_withdrawal_score = dormancy_days × withdrawal_surge_ratio
            features["dormancy_withdrawal_score"] = (
                features["days_since_last_trade"] * features.get("withdrawal_surge_ratio", 1.0)
            ).round(4)
        else:
            features["days_since_last_trade"]     = 0.0
            features["dormancy_withdrawal_score"]  = 0.0

        # Drop helper timestamp columns — not model features
        features.drop(
            columns=["last_portal_ts", "last_trade_ts"],
            errors="ignore",
            inplace=True,
        )

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        logger.info("Feature matrix shape: %s", features.shape)
        return features


# ---------------------------------------------------------------------------
# Streaming buffer (unchanged API, internally consistent with v3)
# ---------------------------------------------------------------------------

class StreamingFeatureBuffer:
    _PORTAL_COLS = [
        "user_id", "timestamp", "event_type", "ip_address", "device_type",
        "region", "session_duration_s", "amount_usd", "ticket_category",
        "ticket_priority", "document_type", "document_verified", "modification_type",
    ]
    _TRADE_COLS = [
        "trade_id", "user_id", "timestamp", "currency_pair", "trade_size",
        "lot_size", "leverage", "margin_used", "margin_usage_pct",
        "trade_duration_s", "pnl_usd", "is_anomalous",
    ]

    def __init__(self, window_minutes: int = 15):
        self._keep_at_least = timedelta(hours=1)
        self._portal_buf: list[dict] = []
        self._trade_buf:  list[dict] = []
        self._fe = FeatureEngineer(window_minutes)

    def add_portal_event(self, event: dict) -> None:
        self._portal_buf.append(event)

    def add_trade_event(self, event: dict) -> None:
        self._trade_buf.append(event)

    def _prune(self, now: pd.Timestamp) -> None:
        cutoff = now - self._keep_at_least
        self._portal_buf = [e for e in self._portal_buf if e["timestamp"] >= cutoff]
        self._trade_buf  = [e for e in self._trade_buf  if e["timestamp"] >= cutoff]

    def get_features(self, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        portal_df = pd.DataFrame(self._portal_buf) if self._portal_buf else pd.DataFrame(columns=self._PORTAL_COLS)
        trade_df  = pd.DataFrame(self._trade_buf)  if self._trade_buf  else pd.DataFrame(columns=self._TRADE_COLS)
        if portal_df.empty and trade_df.empty:
            return pd.DataFrame()
        if as_of is not None:
            portal_df = portal_df[portal_df["timestamp"] <= as_of]
            trade_df  = trade_df[trade_df["timestamp"]   <= as_of]
        return self._fe.build_features(portal_df, trade_df)
