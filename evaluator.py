"""
evaluator.py  (v4 — label-file ground truth + previous ensemble fixes)
-----------------------------------------------------------------------
Model evaluation for unsupervised anomaly detectors.

Fix in this version (v4)
-------------------------
BUG (label leakage via ID prefix) — get_ground_truth() previously derived
    ground truth by checking whether user_id starts with "USR_A". This made
    the label a deterministic function of the user ID string, meaning any
    accidental inclusion of user_id in features or a bad merge key could give
    a model perfect information for free. It also meant evaluation was testing
    a naming convention rather than real annotation.

    Fix: get_ground_truth() now loads data/user_labels.parquet (written by
    ForexDataGenerator.save()) and merges on user_id. The label file is the
    ONLY source of ground truth. User IDs are now opaque UUID-based strings
    (e.g. "U-a3f2c1...") that carry no label information.

Bugs fixed in previous versions (retained)
------------------------------------------
BUG 1 (v3) — Ensemble threshold 0.45 caused ensemble to perform worse than IF
    alone. Fix A: threshold lowered to 0.35. Fix B: OR-logic flags — a user is
    anomalous if EITHER model flags them, preserving ensemble recall benefit.

BUG 6 (v3) — ensemble_scores() filled missing user scores with 0 (zero
    imputation). Fix: fill with per-model mean score (neutral imputation).

BUG (v2) — ensemble_scores() returned 'ensemble_score' but not 'anomaly_score',
    so evaluate() used the wrong column. Fixed: both columns written.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LABELS_PATH = Path("data/user_labels.parquet")


def get_ground_truth(
    features_df: pd.DataFrame,
    labels_path: str | Path = _LABELS_PATH,
) -> pd.Series:
    """
    Return a binary Series (1 = anomalous, 0 = normal) aligned to features_df.

    Loads the label file written by ForexDataGenerator. User IDs are opaque
    UUID-based strings — no label is encoded in the ID itself (FIX v4).

    Parameters
    ----------
    features_df  : pd.DataFrame  — must contain 'user_id' column.
    labels_path  : path to user_labels.parquet produced by the data generator.
    """
    labels_df = pd.read_parquet(labels_path)[["user_id", "is_anomalous"]]
    merged    = features_df[["user_id"]].merge(labels_df, on="user_id", how="left")
    if merged["is_anomalous"].isna().any():
        n_missing = int(merged["is_anomalous"].isna().sum())
        logger.warning(
            "get_ground_truth: %d user_id(s) not found in label file — "
            "treating as normal (0). Check that labels were generated with "
            "the same ForexDataGenerator run.",
            n_missing,
        )
    return merged["is_anomalous"].fillna(False).astype(int)


def evaluate(
    scores_df: pd.DataFrame,
    features_df: pd.DataFrame,
    model_name: str = "Model",
    score_col: str = "anomaly_score",
    labels_path: str | Path = _LABELS_PATH,
) -> dict:
    from sklearn.metrics import (
        average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    merged = scores_df.merge(features_df[["user_id"]], on="user_id", how="left")
    y_true = get_ground_truth(merged, labels_path=labels_path).values
    y_pred = merged["is_anomaly"].astype(int).values

    if score_col not in merged.columns:
        logger.warning(
            "evaluate(): score_col='%s' missing. Available: %s. Falling back to is_anomaly.",
            score_col, list(merged.columns),
        )
        y_score = y_pred.astype(float)
    else:
        y_score = merged[score_col].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc  = average_precision_score(y_true, y_score)
    except ValueError as exc:
        logger.warning("AUC failed (%s) — defaulting to 0.5.", exc)
        roc_auc = 0.5
        pr_auc  = float(y_true.mean())

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "model":           model_name,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1_score":        round(f1, 4),
        "roc_auc":         round(roc_auc, 4),
        "pr_auc":          round(pr_auc, 4),
        "tp":              int(tp),
        "fp":              int(fp),
        "tn":              int(tn),
        "fn":              int(fn),
        "total_flagged":   int(y_pred.sum()),
        "total_anomalous": int(y_true.sum()),
    }
    logger.info(
        "[%s] P=%.3f R=%.3f F1=%.3f ROC-AUC=%.3f PR-AUC=%.3f | TP=%d FP=%d FN=%d",
        model_name, precision, recall, f1, roc_auc, pr_auc, tp, fp, fn,
    )
    return metrics


def compare_models(
    if_scores_df: pd.DataFrame,
    ae_scores_df: pd.DataFrame,
    features_df:  pd.DataFrame,
    labels_path:  str | Path = _LABELS_PATH,
) -> pd.DataFrame:
    if_metrics = evaluate(if_scores_df, features_df, model_name="Isolation Forest",
                          labels_path=labels_path)
    ae_metrics = evaluate(ae_scores_df, features_df, model_name="Autoencoder",
                          labels_path=labels_path)
    comparison = pd.DataFrame([if_metrics, ae_metrics]).set_index("model")
    logger.info("\n%s", comparison.to_string())
    return comparison


def ensemble_scores(
    if_scores_df: pd.DataFrame,
    ae_scores_df: pd.DataFrame,
    features_df:  pd.DataFrame,
    w_if: float = 0.5,
    w_ae: float = 0.5,
    threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Weighted ensemble of IF and AE scores with OR-logic anomaly flags.

    BUG 1 FIX (v3) — is_anomaly uses OR logic: flagged if EITHER model flags
    the user, restoring the ensemble's purpose of catching cases one model misses.

    BUG 6 FIX (v3) — Missing scores filled with per-model mean (neutral
    imputation) rather than 0 (which artificially suppressed borderline cases).

    Parameters
    ----------
    threshold : float
        Score threshold for ensemble_score-based flagging when both models score
        a user. The OR flag takes precedence, but threshold covers cases where
        both models are uncertain but their combined score crosses the line.
    """
    if_df = if_scores_df[["user_id", "anomaly_score", "is_anomaly"]].rename(
        columns={"anomaly_score": "if_score", "is_anomaly": "if_flag"}
    )
    ae_df = ae_scores_df[["user_id", "anomaly_score", "is_anomaly"]].rename(
        columns={"anomaly_score": "ae_score", "is_anomaly": "ae_flag"}
    )

    merged = if_df.merge(ae_df, on="user_id", how="outer")

    # BUG 6 FIX: fill missing scores with column mean, not 0
    merged["if_score"] = merged["if_score"].fillna(merged["if_score"].mean())
    merged["ae_score"] = merged["ae_score"].fillna(merged["ae_score"].mean())
    merged["if_flag"]  = merged["if_flag"].fillna(False)
    merged["ae_flag"]  = merged["ae_flag"].fillna(False)

    merged["ensemble_score"] = (w_if * merged["if_score"] + w_ae * merged["ae_score"]).round(4)
    merged["anomaly_score"]  = merged["ensemble_score"]  # alias for evaluate()

    # BUG 1 FIX: OR logic — flag if EITHER model flags, or score crosses threshold
    merged["is_anomaly"] = (
        merged["if_flag"] | merged["ae_flag"] |
        (merged["ensemble_score"] >= threshold)
    )

    n_flagged = int(merged["is_anomaly"].sum())
    logger.info(
        "Ensemble scored %d users — anomalies: %d (OR-logic + score threshold=%.2f)",
        len(merged), n_flagged, threshold,
    )
    return merged


def print_report(metrics: dict) -> None:
    print(f"\n{'─' * 45}")
    print(f"  Model : {metrics['model']}")
    print(f"{'─' * 45}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1 Score       : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC         : {metrics['pr_auc']:.4f}")
    print(f"{'─' * 45}")
    print(f"  True Positives : {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  True Negatives : {metrics['tn']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  Total Flagged  : {metrics['total_flagged']}")
    print(f"  Known Anomalous: {metrics['total_anomalous']}")
    print(f"{'─' * 45}\n")
