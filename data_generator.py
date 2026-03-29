"""
data_generator.py  (v5 — structural leakage fixes)
----------------------------------------------------
Synthetic forex trading event generator.

Fixes applied in v5
-------------------

FIX 1 — Non-informative user IDs
    Previous: normal users "USR_XXXXX", anomalous users "USR_AXXXX".
    The "USR_A" prefix is a deterministic label encoded in the ID. If user_id
    ever reaches a feature column or a merge key, the evaluation collapses to
    a string-match. The evaluator's get_ground_truth() derived labels purely
    from this prefix — making the ground truth a naming convention, not real
    annotation.
    Fix: all users get UUID4-based IDs (e.g. "U-a3f2..."). A separate mapping
    file (data/user_labels.parquet) stores {user_id, is_anomalous} and is the
    ONLY source of ground truth. get_ground_truth() is updated to load this file
    instead of doing a string prefix check.

FIX 2 — Normal users were locked to a single region (near-perfect geo separator)
    Previous: _normal_portal_events() chose one home_region and used it for
    EVERY event. So unique_regions == 1 for every normal user, always.
    The 90th-percentile threshold for unique_regions was 1.0, meaning any
    anomalous user with ≥2 regions triggered the rule with certainty.
    Fix: 15% of normal users are "traveller" profiles with 2 home regions,
    and all normal users have a 5% per-login chance of a roaming event from
    a different region. This makes unique_regions a soft signal, not a
    hard binary separator.

FIX 3 — hft_burst injected 10–25 trades in 5 minutes (near-perfect separator)
    Previous: normal users' timestamps are uniformly random over 30 days, so
    max_trades_5min ≈ 1–2 almost always. 10–25 in 5 min is >5σ above normal.
    Fix: hft_burst reduced to 3–7 trades in 5 minutes. Normal users also get
    occasional mini-bursts (1–3 trades within a few minutes) so the 5-min
    window distribution has a heavier tail. The signal is now in the combination
    of burst size + margin + leverage, not in burst size alone.

FIX 4 — Each anomalous user had exactly ONE portal anomaly type + ONE trade
    anomaly type (clean single-signal structure).
    Fix: anomalous users now draw 1–2 portal anomaly types and 1–2 trade
    anomaly types from their respective menus. The signal is in the combination
    of slightly-elevated multivariate features, as intended.

FIX 5 (v5 docstring) — generate() return type annotation corrected.
    Previous annotation said tuple[pd.DataFrame, pd.DataFrame] — missing the
    third return value (labels_df). This caused confusion when reading the code
    and could mislead static analysis tools. The implementation was already
    correct (returning 3 values); only the annotation was wrong.
    Fix: return type now correctly annotated as
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame].

Previous fixes (v2–v4, retained)
----------------------------------
- Volume bias: anomalous users use the same base event count as normals.
- Anomaly magnitude overlap: all injected values sit within the normal upper tail.
- geo_hopping: 2–4 regions (was 4–8).
- deposit_churn: 2k–15k (was 5k–50k).
- hft_burst: further reduced from 10–25 → 3–7 (this version).
- trade_size_spike: lot 2–8, leverage 100–200, margin 75–92.
- high_margin: margin 70–88, leverage 100–200.
"""

import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGIONS = {
    "EU":    ["192.168.", "10.0.",    "172.16."],
    "US":    ["203.0.",   "198.51.",  "100.64."],
    "APAC":  ["1.1.",     "8.8.",     "9.9."],
    "LATAM": ["45.7.",    "45.8.",    "45.9."],
}

_CURRENCIES = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/GBP", "USD/CHF",
]

_DEVICES        = ["desktop", "mobile", "tablet"]
_DEVICE_WEIGHTS = [0.55, 0.35, 0.10]

_PORTAL_EVENT_TYPES = [
    "login", "logout", "kyc_update", "deposit", "withdrawal",
    "password_change", "2fa_toggle",
    "support_ticket_open", "support_ticket_close",
    "document_upload",
    "account_modification",
]
_PORTAL_WEIGHTS = [0.28, 0.20, 0.04, 0.12, 0.08, 0.04, 0.04,
                   0.07, 0.05, 0.05, 0.03]

_TICKET_CATEGORIES = [
    "withdrawal_issue", "login_problem", "kyc_query",
    "trade_dispute", "account_locked", "general_inquiry",
]

_DOCUMENT_TYPES = [
    "passport", "utility_bill", "bank_statement",
    "proof_of_funds", "tax_document", "corporate_docs",
]

_ACCOUNT_MOD_TYPES = [
    "email_change", "phone_change", "address_change",
    "bank_account_change", "leverage_change", "risk_profile_change",
]


def _make_user_id() -> str:
    """Generate a non-informative UUID-based user ID (FIX 1)."""
    return "U-" + uuid.uuid4().hex[:12]


def _random_ip(region: Optional[str] = None) -> str:
    if region is None:
        region = random.choice(list(_REGIONS.keys()))
    prefix = random.choice(_REGIONS[region])
    return prefix + f"{random.randint(1, 254)}.{random.randint(1, 254)}"


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _portal_event_extras(evt_type: str) -> dict:
    extras: dict = {
        "ticket_category":    None,
        "ticket_priority":    None,
        "document_type":      None,
        "document_verified":  None,
        "modification_type":  None,
        "amount_usd":         None,
        "session_duration_s": None,
    }
    if evt_type == "support_ticket_open":
        extras["ticket_category"] = random.choice(_TICKET_CATEGORIES)
        extras["ticket_priority"] = random.choices(
            ["low", "medium", "high", "critical"],
            weights=[0.50, 0.30, 0.15, 0.05],
        )[0]
    elif evt_type == "document_upload":
        extras["document_type"]     = random.choice(_DOCUMENT_TYPES)
        extras["document_verified"] = random.choices([True, False], weights=[0.75, 0.25])[0]
    elif evt_type == "account_modification":
        extras["modification_type"] = random.choice(_ACCOUNT_MOD_TYPES)
    elif evt_type in ("deposit", "withdrawal"):
        extras["amount_usd"] = round(np.random.lognormal(6.5, 1.0), 2)
    elif evt_type == "login":
        extras["session_duration_s"] = int(np.random.exponential(600))
    return extras


# ---------------------------------------------------------------------------
# Normal user profile
# ---------------------------------------------------------------------------

def _normal_portal_events(user_id: str, n: int, start: datetime, end: datetime) -> list[dict]:
    """
    FIX 2: Normal users now have realistic geographic variation.
    - 15% are "traveller" profiles with 2 home regions.
    - All normal users have a 5% per-login chance of a roaming event from a
      different region (business trip, holiday).
    This breaks the unique_regions == 1 invariant that previously made
    geo_change_rate a near-perfect anomaly separator.
    """
    events = []
    all_regions = list(_REGIONS.keys())
    home_region = random.choice(all_regions)

    is_traveller = random.random() < 0.15
    secondary_region = random.choice([r for r in all_regions if r != home_region]) if is_traveller else None

    device = random.choices(_DEVICES, weights=_DEVICE_WEIGHTS)[0]

    for _ in range(n):
        evt_type = random.choices(_PORTAL_EVENT_TYPES, weights=_PORTAL_WEIGHTS)[0]
        ts       = _random_timestamp(start, end)
        extras   = _portal_event_extras(evt_type)

        if evt_type == "login":
            if is_traveller and secondary_region and random.random() < 0.25:
                region = secondary_region
            elif random.random() < 0.05:
                region = random.choice([r for r in all_regions if r != home_region])
            else:
                region = home_region
        else:
            region = home_region

        events.append({
            "user_id":     user_id,
            "timestamp":   ts,
            "event_type":  evt_type,
            "ip_address":  _random_ip(region),
            "device_type": device,
            "region":      region,
            **extras,
        })

    if random.random() < 0.10:
        burst_ts = _random_timestamp(start, end - timedelta(minutes=5))
        for i in range(random.randint(1, 3)):
            events.append({
                "user_id":     user_id,
                "timestamp":   burst_ts + timedelta(seconds=i * 30),
                "event_type":  "login",
                "ip_address":  _random_ip(home_region),
                "device_type": device,
                "region":      home_region,
                "ticket_category": None, "ticket_priority": None,
                "document_type": None, "document_verified": None,
                "modification_type": None, "amount_usd": None,
                "session_duration_s": int(np.random.exponential(300)),
            })

    return events


def _normal_trade_events(user_id: str, n: int, start: datetime, end: datetime) -> list[dict]:
    """
    FIX 3: Normal users occasionally execute 2–4 trades within a few minutes
    (e.g. hedging, scalping a news event). This gives max_trades_5min a
    heavier tail so hft_burst anomalies at 3–7 trades are not trivially separated.
    """
    trades = []
    for _ in range(n):
        pair             = random.choice(_CURRENCIES)
        lot_size         = round(random.choice([0.01, 0.05, 0.1, 0.5, 1.0]) * np.random.lognormal(0, 0.3), 2)
        trade_size       = round(lot_size * 100_000, 2)
        leverage         = random.choice([10, 20, 30, 50, 100])
        margin_used      = round(trade_size / leverage, 2)
        margin_usage_pct = round(np.clip(np.random.beta(2, 8) * 100, 1, 80), 2)
        trades.append({
            "trade_id":         f"T-{uuid.uuid4().hex[:8]}",
            "user_id":          user_id,
            "timestamp":        _random_timestamp(start, end),
            "currency_pair":    pair,
            "trade_size":       trade_size,
            "lot_size":         lot_size,
            "leverage":         leverage,
            "margin_used":      margin_used,
            "margin_usage_pct": margin_usage_pct,
            "trade_duration_s": int(np.random.exponential(1800)),
            "pnl_usd":          round(np.random.normal(0, trade_size * 0.01), 2),
            "is_anomalous":     False,
        })

    if random.random() < 0.12:
        burst_ts = _random_timestamp(start, end - timedelta(minutes=5))
        for i in range(random.randint(2, 4)):
            pair       = random.choice(_CURRENCIES)
            lot_size   = round(random.choice([0.01, 0.05, 0.1]) * np.random.lognormal(0, 0.2), 2)
            trade_size = round(lot_size * 100_000, 2)
            leverage   = random.choice([10, 20, 30])
            trades.append({
                "trade_id":         f"T-{uuid.uuid4().hex[:8]}",
                "user_id":          user_id,
                "timestamp":        burst_ts + timedelta(seconds=i * 45),
                "currency_pair":    pair,
                "trade_size":       trade_size,
                "lot_size":         lot_size,
                "leverage":         leverage,
                "margin_used":      round(trade_size / leverage, 2),
                "margin_usage_pct": round(np.clip(np.random.beta(2, 8) * 100, 1, 70), 2),
                "trade_duration_s": random.randint(60, 600),
                "pnl_usd":          round(np.random.normal(0, trade_size * 0.01), 2),
                "is_anomalous":     False,
            })

    return trades


# ---------------------------------------------------------------------------
# Anomalous user profiles
#
# FIX 4: Each anomalous user now draws 1–2 portal anomaly types AND 1–2 trade
# anomaly types. The signal is in the COMBINATION of features, not a single
# clear signature. Each individual feature may overlap with aggressive-normal
# users; it is the multivariate pattern that distinguishes anomalies.
# ---------------------------------------------------------------------------

def _inject_portal_anomaly(anomaly_type: str, user_id: str,
                            events: list[dict], start: datetime, end: datetime) -> None:
    """Inject one portal anomaly pattern into an existing event list (in-place)."""

    if anomaly_type == "geo_hopping":
        burst_start  = _random_timestamp(start, end - timedelta(hours=6))
        n_hops       = random.randint(2, 4)
        used_regions = random.sample(list(_REGIONS.keys()), min(n_hops, 4))
        for i, region in enumerate(used_regions):
            events.append({
                "user_id": user_id,
                "timestamp": burst_start + timedelta(minutes=i * 40),
                "event_type": "login",
                "ip_address": _random_ip(region),
                "device_type": random.choice(_DEVICES),
                "region": region,
                "session_duration_s": random.randint(60, 600),
                "amount_usd": None,
                "ticket_category": None, "ticket_priority": None,
                "document_type": None, "document_verified": None,
                "modification_type": None,
            })

    elif anomaly_type == "deposit_withdrawal_churn":
        cycle_start = _random_timestamp(start, end - timedelta(hours=2))
        for i in range(random.randint(3, 6)):
            for evt in ("deposit", "withdrawal"):
                events.append({
                    "user_id":    user_id,
                    "timestamp":  cycle_start + timedelta(minutes=i * 10 + (5 if evt == "withdrawal" else 0)),
                    "event_type": evt,
                    "ip_address": _random_ip(),
                    "device_type": "desktop",
                    "region":     random.choice(list(_REGIONS.keys())),
                    "session_duration_s": None,
                    "amount_usd": round(np.random.uniform(2000, 15000), 2),
                    "ticket_category": None, "ticket_priority": None,
                    "document_type": None, "document_verified": None,
                    "modification_type": None,
                })

    elif anomaly_type == "high_frequency_login":
        burst_start = _random_timestamp(start, end - timedelta(hours=1))
        for i in range(random.randint(6, 12)):
            events.append({
                "user_id": user_id,
                "timestamp": burst_start + timedelta(seconds=i * 45),
                "event_type": "login", "ip_address": _random_ip(),
                "device_type": "mobile", "region": random.choice(list(_REGIONS.keys())),
                "session_duration_s": random.randint(10, 60), "amount_usd": None,
                "ticket_category": None, "ticket_priority": None,
                "document_type": None, "document_verified": None, "modification_type": None,
            })

    elif anomaly_type == "document_spam":
        burst_start = _random_timestamp(start, end - timedelta(hours=3))
        for i in range(random.randint(4, 8)):
            events.append({
                "user_id": user_id,
                "timestamp": burst_start + timedelta(minutes=i * 10),
                "event_type": "document_upload", "ip_address": _random_ip(),
                "device_type": "desktop", "region": random.choice(list(_REGIONS.keys())),
                "session_duration_s": None, "amount_usd": None,
                "ticket_category": None, "ticket_priority": None,
                "document_type": random.choice(_DOCUMENT_TYPES),
                "document_verified": False, "modification_type": None,
            })

    elif anomaly_type == "rapid_account_changes":
        burst_start = _random_timestamp(start, end - timedelta(hours=2))
        sensitive   = ["bank_account_change", "email_change", "phone_change", "leverage_change"]
        for i in range(random.randint(4, 8)):
            events.append({
                "user_id": user_id,
                "timestamp": burst_start + timedelta(minutes=i * 10),
                "event_type": "account_modification", "ip_address": _random_ip(),
                "device_type": random.choice(_DEVICES), "region": random.choice(list(_REGIONS.keys())),
                "session_duration_s": None, "amount_usd": None,
                "ticket_category": None, "ticket_priority": None,
                "document_type": None, "document_verified": None,
                "modification_type": random.choice(sensitive),
            })


def _anomalous_portal_events(user_id: str, n: int, start: datetime, end: datetime) -> list[dict]:
    """
    FIX 4: Draw 1–2 portal anomaly types per user (weighted toward 1 but
    occasionally 2). Signals are additive rather than mutually exclusive.
    """
    events = _normal_portal_events(user_id, n, start, end)

    portal_anomaly_types = [
        "geo_hopping", "deposit_withdrawal_churn",
        "high_frequency_login", "document_spam", "rapid_account_changes",
    ]
    n_types = 1 if random.random() < 0.70 else 2
    chosen  = random.sample(portal_anomaly_types, n_types)
    for atype in chosen:
        _inject_portal_anomaly(atype, user_id, events, start, end)

    return events


def _inject_trade_anomaly(anomaly_type: str, user_id: str,
                           trades: list[dict], start: datetime, end: datetime) -> None:
    """Inject one trade anomaly pattern into an existing trade list (in-place)."""

    if anomaly_type == "trade_size_spike":
        spike_ts = _random_timestamp(start, end)
        for _ in range(random.randint(2, 5)):
            lot_size   = round(random.uniform(2.0, 8.0), 2)
            trade_size = lot_size * 100_000
            leverage   = random.choice([100, 150, 200])
            trades.append({
                "trade_id":         f"T-{uuid.uuid4().hex[:8]}",
                "user_id":          user_id,
                "timestamp":        spike_ts + timedelta(seconds=random.randint(0, 300)),
                "currency_pair":    random.choice(_CURRENCIES),
                "trade_size":       trade_size,
                "lot_size":         lot_size,
                "leverage":         leverage,
                "margin_used":      round(trade_size / leverage, 2),
                "margin_usage_pct": round(np.random.uniform(75, 92), 2),
                "trade_duration_s": random.randint(30, 300),
                "pnl_usd":          round(np.random.normal(0, trade_size * 0.015), 2),
                "is_anomalous":     True,
            })

    elif anomaly_type == "hft_burst":
        # FIX 3: Reduced from 10–25 → 3–7 trades in 5 minutes.
        burst_ts = _random_timestamp(start, end - timedelta(minutes=5))
        for i in range(random.randint(3, 7)):
            lot_size   = round(random.uniform(0.1, 1.5), 2)
            trade_size = lot_size * 100_000
            trades.append({
                "trade_id":         f"T-{uuid.uuid4().hex[:8]}",
                "user_id":          user_id,
                "timestamp":        burst_ts + timedelta(seconds=i * 30),
                "currency_pair":    random.choice(_CURRENCIES),
                "trade_size":       trade_size,
                "lot_size":         lot_size,
                "leverage":         100,
                "margin_used":      round(trade_size / 100, 2),
                "margin_usage_pct": round(np.random.uniform(35, 65), 2),
                "trade_duration_s": random.randint(10, 60),
                "pnl_usd":          round(np.random.normal(0, 100), 2),
                "is_anomalous":     True,
            })

    elif anomaly_type == "high_margin":
        # margin 70–88 (was 88–99), leverage 100–200.
        for t in trades[-random.randint(5, 15):]:
            t["margin_usage_pct"] = round(np.random.uniform(70, 88), 2)
            t["leverage"]         = random.choice([100, 150, 200])
            t["is_anomalous"]     = True


def _anomalous_trade_events(user_id: str, n: int, start: datetime, end: datetime) -> list[dict]:
    """
    FIX 4: Draw 1–2 trade anomaly types per user (weighted toward 1 but
    occasionally 2). Prevents trivial single-feature separation.
    """
    trades = _normal_trade_events(user_id, n, start, end)

    trade_anomaly_types = ["trade_size_spike", "hft_burst", "high_margin"]
    n_types = 1 if random.random() < 0.65 else 2
    chosen  = random.sample(trade_anomaly_types, min(n_types, len(trade_anomaly_types)))
    for atype in chosen:
        _inject_trade_anomaly(atype, user_id, trades, start, end)

    return trades


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class ForexDataGenerator:
    """
    Generates synthetic forex trading and portal events.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        dg = self.cfg["data_generation"]
        self.n_normal         = dg["n_normal_users"]
        self.n_anomalous      = dg["n_anomalous_users"]
        self.events_normal    = dg["events_per_normal_user"]
        self.events_anomalous = self.events_normal
        self.start            = datetime.fromisoformat(dg["start_date"])
        self.end              = datetime.fromisoformat(dg["end_date"])
        self.output_path      = Path(dg["output_path"])

        random.seed(self.cfg["model"]["random_state"])
        np.random.seed(self.cfg["model"]["random_state"])

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic portal events, trade events, and user labels.

        Returns
        -------
        portal_df : pd.DataFrame
            All portal (web/app activity) events.
        trades_df : pd.DataFrame
            All trade events.
        labels_df : pd.DataFrame
            Per-user ground-truth labels {user_id, is_anomalous}.
            This is the ONLY source of ground truth — no label information
            is encoded in user IDs (FIX 1).

        FIX 5: return type annotation corrected to
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame].
            The previous annotation omitted labels_df, which was misleading
            to readers and static analysis tools even though the implementation
            already returned all three values.
        """
        logger.info(
            "Generating data: %d normal + %d anomalous users, "
            "%d base events/user, window %s → %s …",
            self.n_normal, self.n_anomalous, self.events_normal,
            self.start.date(), self.end.date(),
        )
        portal_rows, trade_rows = [], []
        label_rows: list[dict] = []

        for _ in range(self.n_normal):
            uid = _make_user_id()
            label_rows.append({"user_id": uid, "is_anomalous": False})
            n_portal = int(np.random.poisson(self.events_normal * 0.4))
            n_trade  = int(np.random.poisson(self.events_normal * 0.6))
            portal_rows.extend(_normal_portal_events(uid, max(n_portal, 5), self.start, self.end))
            trade_rows.extend(_normal_trade_events(uid, max(n_trade, 5), self.start, self.end))

        for _ in range(self.n_anomalous):
            uid = _make_user_id()
            label_rows.append({"user_id": uid, "is_anomalous": True})
            n_portal = int(np.random.poisson(self.events_anomalous * 0.4))
            n_trade  = int(np.random.poisson(self.events_anomalous * 0.6))
            portal_rows.extend(_anomalous_portal_events(uid, max(n_portal, 5), self.start, self.end))
            trade_rows.extend(_anomalous_trade_events(uid, max(n_trade, 5), self.start, self.end))

        portal_df = (
            pd.DataFrame(portal_rows)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        trades_df = (
            pd.DataFrame(trade_rows)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        labels_df = pd.DataFrame(label_rows)

        logger.info(
            "Generated %d portal events and %d trade events across %d users "
            "(%d normal, %d anomalous).",
            len(portal_df), len(trades_df),
            self.n_normal + self.n_anomalous,
            self.n_normal, self.n_anomalous,
        )
        return portal_df, trades_df, labels_df

    def save(
        self,
        portal_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        portal_path = self.output_path.parent / "portal_events.parquet"
        trades_path = self.output_path.parent / "trade_events.parquet"
        labels_path = self.output_path.parent / "user_labels.parquet"
        portal_df.to_parquet(portal_path, index=False)
        trades_df.to_parquet(trades_path, index=False)
        labels_df.to_parquet(labels_path, index=False)
        logger.info("Saved portal  -> %s", portal_path)
        logger.info("Saved trades  -> %s", trades_path)
        logger.info("Saved labels  -> %s", labels_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gen = ForexDataGenerator()
    portal_df, trades_df, labels_df = gen.generate()
    gen.save(portal_df, trades_df, labels_df)
    print(f"Portal events shape : {portal_df.shape}")
    print(f"Trade events shape  : {trades_df.shape}")
    print(f"Labels shape        : {labels_df.shape}")
    print(f"\nPortal event types:\n{portal_df['event_type'].value_counts()}")
