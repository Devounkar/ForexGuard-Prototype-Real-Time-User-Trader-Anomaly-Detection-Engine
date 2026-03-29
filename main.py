"""
main.py  (v5 — UUID user IDs + label-file ground truth)
---------------------------------------------------------
Entry point for the Forex Anomaly Detection System.

Modes
-----
  python main.py generate   -- generate synthetic data
  python main.py train      -- feature engineering + train BOTH models
  python main.py evaluate   -- evaluate both models, print metrics
  python main.py stream     -- run simulated streaming pipeline
  python main.py run        -- generate + train + evaluate + stream in one shot

IMPORTANT — after replacing any source file you MUST run:
    python main.py generate   (to get new UUID-based IDs + label file)
    python main.py train
    python main.py evaluate
Old .pkl files and old parquets with "USR_A" IDs are incompatible.

Fixes applied (v5)
------------------
FIX 1 (data_generator.py) — User IDs are now opaque UUID-based strings.
    Labels are stored in data/user_labels.parquet instead of being encoded
    in the ID prefix. The evaluator loads this file for ground truth.

FIX 2 (data_generator.py) — Normal users now have realistic geographic
    variation (traveller profiles + 5% roaming chance per login). Breaks
    the unique_regions == 1 invariant that was a near-perfect separator.

FIX 3 (data_generator.py) — hft_burst reduced from 10–25 → 3–7 trades in
    5 minutes. Normal users also get occasional mini-bursts (2–4 trades)
    so max_trades_5min has a heavier tail.

FIX 4 (data_generator.py) — Anomalous users now draw 1–2 portal anomaly
    types AND 1–2 trade anomaly types (vs exactly 1 of each before).

FIX 5 (evaluator.py) — get_ground_truth() loads user_labels.parquet instead
    of checking user_id prefix. evaluate() accepts labels_path param.

Previous fixes (v2–v4, retained)
----------------------------------
FIX A (v3) — lr=0.0005 for AE training stability.
FIX B (v3) — 80/20 stratified train/test split for out-of-sample metrics.
FIX C (v4, Bug 7) — RuleBasedExplainer fitted on train_df only, matching
    the population used in scoring. Streaming mode does the same.
FIX D (v4, Bug 4) — contamination="auto" in config so IF doesn't get a
    free hint about the true anomaly fraction.
FIX E (v4, Bug 3) — AE reconstruction errors consistently clipped before
    computing MSE (autoencoder.py).
FIX F (v4, Bug 1+6) — Ensemble uses OR-logic flags and mean imputation
    for missing scores (evaluator.py).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from src.alert_system import AlertSystem
from src.autoencoder import AutoencoderDetector
from src.data_generator import ForexDataGenerator
from src.evaluator import compare_models, ensemble_scores, evaluate, print_report
from src.explainability import RuleBasedExplainer
from src.feature_engineering import FeatureEngineer
from src.model import AnomalyDetector
from src.streamer import StreamingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

CONFIG      = "configs/config.yaml"
LABELS_PATH = "data/user_labels.parquet"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def cmd_generate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("=== STEP 1 : Data Generation ===")
    gen = ForexDataGenerator(CONFIG)
    portal_df, trades_df, labels_df = gen.generate()
    gen.save(portal_df, trades_df, labels_df)
    n_anom = labels_df["is_anomalous"].sum()
    n_norm = (~labels_df["is_anomalous"]).sum()
    logger.info(
        "Portal events: %d  |  Trade events: %d  |  Users: normal=%d anomalous=%d",
        len(portal_df), len(trades_df), n_norm, n_anom,
    )
    return portal_df, trades_df, labels_df


def cmd_train(portal_df: pd.DataFrame, trades_df: pd.DataFrame, labels_df: pd.DataFrame):
    logger.info("=== STEP 2 : Feature Engineering & Model Training ===")

    fe = FeatureEngineer()
    features_df = fe.build_features(portal_df, trades_df)
    features_df.to_parquet("data/features.parquet", index=False)
    logger.info("Features saved  shape=%s", features_df.shape)

    # Merge labels for stratified split — labels not used as model features
    features_with_labels = features_df.merge(labels_df, on="user_id", how="left")
    label_col = features_with_labels["is_anomalous"].fillna(False).astype(int)

    train_df, test_df = train_test_split(
        features_df, test_size=0.2, random_state=42, stratify=label_col
    )
    train_df.to_parquet("data/train_features.parquet", index=False)
    test_df.to_parquet("data/test_features.parquet",  index=False)

    train_labels = labels_df[labels_df["user_id"].isin(train_df["user_id"])]
    test_labels  = labels_df[labels_df["user_id"].isin(test_df["user_id"])]
    logger.info(
        "Train/test split — train: %d (anomalous: %d)  test: %d (anomalous: %d)",
        len(train_df), train_labels["is_anomalous"].sum(),
        len(test_df),  test_labels["is_anomalous"].sum(),
    )

    # Isolation Forest — contamination="auto" (decoupled from true anomaly rate)
    logger.info("--- Training Isolation Forest ---")
    detector = AnomalyDetector(CONFIG)
    detector.train(train_df)
    detector.save()

    # Autoencoder v7 — Adam, wider network, normal-only threshold calibration
    logger.info("--- Training Autoencoder (v7: Adam + wider net + normal-only threshold) ---")
    ae_detector = AutoencoderDetector(
        hidden_dim=32, encoding_dim=16, epochs=300, lr=0.001,
        contamination=0.05, labels_path=Path(LABELS_PATH),
    )
    ae_detector.train(train_df, labels_path=Path(LABELS_PATH))
    ae_detector.save()

    # BUG 7 FIX (v4): explainer always fitted on train_df
    explainer = RuleBasedExplainer(top_k=3).fit(train_df)

    # Alert demo on training set (IF scores)
    if_scores_df = detector.score(train_df)
    explained_df = explainer.explain_batch(train_df, if_scores_df)
    alert_sys    = AlertSystem(CONFIG)
    alerts       = alert_sys.generate_alerts(explained_df)

    logger.info(
        "IF anomalies on train: %d / %d  |  Alerts: %d",
        if_scores_df["is_anomaly"].sum(), len(if_scores_df), len(alerts),
    )
    for a in alerts[:3]:
        logger.info("  [%s] %s -> %.3f | %s", a["severity"], a["user_id"], a["risk_score"], a["alert"])

    return detector, ae_detector, explainer, train_df, test_df


def cmd_evaluate(detector, ae_detector, train_df: pd.DataFrame, test_df: pd.DataFrame):
    logger.info("=== STEP 3 : Model Evaluation (held-out test set) ===")
    logger.info("Evaluating on %d held-out users (never seen during training)", len(test_df))

    if_scores_df = detector.score(test_df)
    ae_scores_df = ae_detector.score(test_df)

    if_scores_df.to_parquet("data/if_scores.parquet", index=False)
    ae_scores_df.to_parquet("data/ae_scores.parquet", index=False)

    # FIX 5: labels_path passed explicitly — no prefix-based ground truth
    if_metrics = evaluate(if_scores_df, test_df, model_name="Isolation Forest",
                          labels_path=LABELS_PATH)
    ae_metrics = evaluate(ae_scores_df, test_df, model_name="Autoencoder",
                          labels_path=LABELS_PATH)
    print_report(if_metrics)
    print_report(ae_metrics)

    comparison = compare_models(if_scores_df, ae_scores_df, test_df, labels_path=LABELS_PATH)
    comparison.to_csv("data/model_comparison.csv")
    logger.info("Comparison saved -> data/model_comparison.csv")

    # Ensemble with OR-logic (Bug 1 + Bug 6 fix in evaluator.py)
    ensemble_df = ensemble_scores(if_scores_df, ae_scores_df, test_df)
    ensemble_df.to_parquet("data/ensemble_scores.parquet", index=False)

    ensemble_metrics = evaluate(
        ensemble_df,
        test_df,
        model_name="Ensemble (IF + AE)",
        score_col="ensemble_score",
        labels_path=LABELS_PATH,
    )
    print_report(ensemble_metrics)

    return if_metrics, ae_metrics, ensemble_metrics


def cmd_stream(portal_df: pd.DataFrame, trades_df: pd.DataFrame, detector, explainer):
    logger.info("=== STEP 4 : Streaming Pipeline ===")
    alert_sys = AlertSystem(CONFIG, output_path="data/stream_alerts.jsonl")
    pipeline  = StreamingPipeline(
        config_path=CONFIG, detector=detector, explainer=explainer,
        alert_sys=alert_sys, score_every_n=1000, delay_ms=0,
    )
    alerts = pipeline.run(portal_df, trades_df)
    logger.info("Streaming complete. %d alerts -> data/stream_alerts.jsonl", len(alerts))
    return alerts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Forex Anomaly Detection System")
    parser.add_argument(
        "mode",
        choices=["generate", "train", "evaluate", "stream", "run"],
    )
    args = parser.parse_args()

    if args.mode == "generate":
        cmd_generate()

    elif args.mode == "train":
        portal_df = pd.read_parquet("data/portal_events.parquet")
        trades_df = pd.read_parquet("data/trade_events.parquet")
        labels_df = pd.read_parquet("data/user_labels.parquet")
        cmd_train(portal_df, trades_df, labels_df)

    elif args.mode == "evaluate":
        test_df     = pd.read_parquet("data/test_features.parquet")
        train_df    = pd.read_parquet("data/train_features.parquet")
        detector    = AnomalyDetector(CONFIG).load()
        ae_detector = AutoencoderDetector(hidden_dim=32, encoding_dim=16).load()
        cmd_evaluate(detector, ae_detector, train_df, test_df)

    elif args.mode == "stream":
        portal_df = pd.read_parquet("data/portal_events.parquet")
        trades_df = pd.read_parquet("data/trade_events.parquet")
        # BUG 7 FIX (v4): fit explainer on train_df only
        train_df  = pd.read_parquet("data/train_features.parquet")
        detector  = AnomalyDetector(CONFIG).load()
        explainer = RuleBasedExplainer(top_k=3).fit(train_df)
        cmd_stream(portal_df, trades_df, detector, explainer)

    elif args.mode == "run":
        portal_df, trades_df, labels_df = cmd_generate()
        detector, ae_detector, explainer, train_df, test_df = cmd_train(
            portal_df, trades_df, labels_df
        )
        cmd_evaluate(detector, ae_detector, train_df, test_df)
        cmd_stream(portal_df, trades_df, detector, explainer)
        logger.info("All stages complete. Launch: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
