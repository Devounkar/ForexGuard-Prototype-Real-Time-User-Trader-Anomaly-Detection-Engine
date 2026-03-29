"""
autoencoder.py  (v8 — train on normals only + stable score normalisation)
--------------------------------------------------------------------------
Bottleneck Autoencoder anomaly detector.

Bugs fixed in this version (v8)
--------------------------------

BUG E — AE trained on the full dataset (normals + anomalies).
    The autoencoder's anomaly detection ability rests entirely on one
    assumption: the network learns the manifold of NORMAL behaviour, so
    anomalous inputs reconstruct poorly. When anomalous users are included
    in the training set, the AE learns to reconstruct their patterns too.
    The reconstruction error gap between normals and anomalies collapses,
    and the model loses its core detection ability.

    Even though BUG A (v7) fixed the *threshold* to be calibrated on
    normal-user errors only, the network *weights* were already contaminated
    by anomalous training examples — the threshold fix was treating a symptom,
    not the disease.

    Fix: _train_autoencoder() is called with X_normal only (users whose
    is_anomalous label is False). The labels file is loaded during train()
    solely for this filtering step. Label information never enters the network
    weights — it is only used to decide which rows to exclude from training.
    Threshold calibration (BUG A fix, retained) remains on normal-user errors.

BUG F — score() normalises reconstruction errors to [0, 1] using the
    min/max of the CURRENT SCORED BATCH, not the training distribution.
    This makes anomaly_score a relative, batch-dependent value:
      - A user with error=0.9 scores 1.0 if they are the worst in the batch,
        but 0.4 if a more extreme outlier is also in the batch.
      - The ensemble blends this drifting AE score equally with IF's stable,
        absolute decision_function score, degrading ensemble quality.
      - The is_anomaly flag uses the raw threshold (stable), but the ensemble
        uses anomaly_score — so the two signals are inconsistent.

    Fix: during train(), save the min and max of training reconstruction
    errors (train_error_min, train_error_max). score() normalises using
    these fixed bounds. Scores outside [0, 1] are clipped so the range
    stays valid for ensemble weighting. This makes AE anomaly_score
    comparable across batches and compatible with IF anomaly_score.

Bugs fixed in previous versions (v7, retained)
-----------------------------------------------
BUG A — Threshold calibrated on full mixed population (too loose).
    Fix: threshold set at (1-contamination) percentile of normal-user
    reconstruction errors only.
BUG B — Vanilla SGD too slow to converge.
    Fix: Adam optimizer with lr=0.001.
BUG C — Network capacity too small (27→16→8→16→27).
    Fix: wider network 27→32→16→32→27.
BUG D — Clip to [-5, 5] discards real anomaly signal.
    Fix: clip extended to [-10, 10].
BUG 3 (v6) — Reconstruction error measured against unclipped inputs.
    Fix: consistent clipping in both train and score.
EARLY STOPPING (v6) — validation split with best-weight restore. Patience=30.
BUG 1 (v3) — _forward() bypassed the bottleneck. Fixed.
BUG 2 (v4) — Exploding gradients. Fixed via gradient clipping + He init.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

from src.model import MODEL_FEATURES

AE_MODEL_PATH  = Path("data/autoencoder.pkl")
AE_SCALER_PATH = Path("data/ae_scaler.pkl")
_LABELS_PATH   = Path("data/user_labels.parquet")

_CLIP_RANGE = 10.0


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _he(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He initialisation — better for ReLU layers than Xavier."""
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))


def _build_autoencoder(input_dim: int, hidden_dim: int = 32, encoding_dim: int = 16) -> dict:
    """
    Wider bottleneck autoencoder: input -> hidden -> bottleneck -> hidden -> input.
    BUG C FIX: increased capacity from (16, 8) to (32, 16).
    Encoder: ReLU activations. Decoder hidden: ReLU. Output: linear.
    """
    rng = np.random.default_rng(42)
    return {
        "W1": _he(input_dim,    hidden_dim,   rng), "b1": np.zeros(hidden_dim),
        "W2": _he(hidden_dim,   encoding_dim, rng), "b2": np.zeros(encoding_dim),
        "W3": _he(encoding_dim, hidden_dim,   rng), "b3": np.zeros(hidden_dim),
        "W4": _he(hidden_dim,   input_dim,    rng), "b4": np.zeros(input_dim),
        "input_dim": input_dim, "hidden_dim": hidden_dim, "encoding_dim": encoding_dim,
    }


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _forward(w: dict, X: np.ndarray) -> dict:
    """Full encoder-decoder forward pass. All activations returned for backprop."""
    z1 = X  @ w["W1"] + w["b1"];  a1 = _relu(z1)
    z2 = a1 @ w["W2"] + w["b2"];  a2 = _relu(z2)
    z3 = a2 @ w["W3"] + w["b3"];  a3 = _relu(z3)
    z4 = a3 @ w["W4"] + w["b4"]
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2,
            "z3": z3, "a3": a3, "z4": z4}


def _copy_weights(w: dict) -> dict:
    """Deep copy of weight dict (for early stopping best-weight restore)."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}


def _init_adam(w: dict) -> dict:
    """Initialise Adam moment accumulators (m=first, v=second) for each weight."""
    adam = {}
    for key in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
        adam[f"m_{key}"] = np.zeros_like(w[key])
        adam[f"v_{key}"] = np.zeros_like(w[key])
    adam["t"] = 0
    return adam


def _adam_update(
    w: dict,
    grads: dict,
    adam: dict,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """In-place Adam update. BUG B FIX: replaces vanilla SGD."""
    adam["t"] += 1
    t = adam["t"]
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t

    for key in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
        g = grads[key]
        adam[f"m_{key}"] = beta1 * adam[f"m_{key}"] + (1.0 - beta1) * g
        adam[f"v_{key}"] = beta2 * adam[f"v_{key}"] + (1.0 - beta2) * g ** 2
        m_hat = adam[f"m_{key}"] / bc1
        v_hat = adam[f"v_{key}"] / bc2
        w[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _train_autoencoder(
    w: dict,
    X: np.ndarray,
    epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    patience: int = 30,
) -> list[float]:
    """
    Mini-batch Adam with MSE loss and early stopping.

    BUG E FIX: X must contain NORMAL users only (filtering done in train()).
    BUG B FIX: Adam optimizer replaces SGD.
    BUG D FIX: clip extended to ±10 (was ±5).
    Patience raised from 20 → 30 to avoid premature stopping.
    """
    X = np.clip(X, -_CLIP_RANGE, _CLIP_RANGE)

    n   = len(X)
    rng = np.random.default_rng(42)

    val_size = max(1, int(n * val_fraction))
    idx_all  = rng.permutation(n)
    val_idx  = idx_all[:val_size]
    trn_idx  = idx_all[val_size:]
    X_val    = X[val_idx]
    X_trn    = X[trn_idx]

    adam          = _init_adam(w)
    best_val_loss = float("inf")
    best_weights  = _copy_weights(w)
    no_improve    = 0
    train_losses  = []

    for epoch in range(epochs):
        perm = rng.permutation(len(X_trn))
        epoch_loss, batches = 0.0, 0

        for start in range(0, len(X_trn), batch_size):
            B = X_trn[perm[start: start + batch_size]]
            m = len(B)

            fwd  = _forward(w, B)
            err  = fwd["z4"] - B
            loss = (err ** 2).mean()

            if np.isnan(loss) or np.isinf(loss):
                logger.warning("NaN/Inf loss at epoch %d — skipping batch.", epoch + 1)
                continue

            epoch_loss += loss
            batches    += 1

            dz4 = 2.0 * err / m
            dW4 = fwd["a3"].T @ dz4;  db4 = dz4.sum(0)
            dz3 = (dz4 @ w["W4"].T) * (fwd["z3"] > 0)
            dW3 = fwd["a2"].T @ dz3;  db3 = dz3.sum(0)
            dz2 = (dz3 @ w["W3"].T) * (fwd["z2"] > 0)
            dW2 = fwd["a1"].T @ dz2;  db2 = dz2.sum(0)
            dz1 = (dz2 @ w["W2"].T) * (fwd["z1"] > 0)
            dW1 = B.T @ dz1;          db1 = dz1.sum(0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                     "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}
            _adam_update(w, grads, adam, lr=lr)

        avg_train = epoch_loss / max(batches, 1)
        train_losses.append(avg_train)

        val_fwd  = _forward(w, X_val)
        val_loss = ((val_fwd["z4"] - X_val) ** 2).mean()

        if (epoch + 1) % 50 == 0:
            logger.info(
                "  Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch + 1, epochs, avg_train, val_loss,
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights  = _copy_weights(w)
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.6f, patience=%d).",
                    epoch + 1, best_val_loss, patience,
                )
                break

    w.update(best_weights)
    logger.info("Training complete. Best val_loss=%.6f", best_val_loss)
    return train_losses


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AutoencoderDetector:
    """
    Deep bottleneck Autoencoder anomaly detector.

    Architecture (v8): input(27) -> 32 -> 16 -> 32 -> input(27)

    v8 changes vs v7:
    - Network trained on NORMAL users only (BUG E fix). The AE learns the
      normal manifold exclusively; anomalous users are excluded from weight
      updates. Labels are used only for this filtering — they never affect
      network weights.
    - score() normalises using training-error bounds (BUG F fix). The
      min/max of training reconstruction errors are saved during train() and
      used as fixed normalisation bounds in score(), making anomaly_score
      stable and comparable across batches and compatible with IF's score.

    v7 changes (retained):
    - Threshold calibrated on normal-user errors only (BUG A fix).
    - Adam optimizer replaces SGD (BUG B fix).
    - Wider network 32/16 replaces 16/8 (BUG C fix).
    - Clip range extended to ±10 from ±5 (BUG D fix).
    - Patience raised from 20 → 30.
    - Default lr = 0.001.
    """

    def __init__(
        self,
        hidden_dim:    int   = 32,
        encoding_dim:  int   = 16,
        epochs:        int   = 300,
        lr:            float = 0.001,
        contamination: float = 0.05,
        model_path:    Path  = AE_MODEL_PATH,
        scaler_path:   Path  = AE_SCALER_PATH,
        labels_path:   Path  = _LABELS_PATH,
    ):
        self.hidden_dim    = hidden_dim
        self.encoding_dim  = encoding_dim
        self.epochs        = epochs
        self.lr            = lr
        self.contamination = contamination
        self.model_path    = Path(model_path)
        self.scaler_path   = Path(scaler_path)
        self.labels_path   = Path(labels_path)

        self._w:                Optional[dict]        = None
        self._scaler:           Optional[RobustScaler] = None
        self._threshold:        float                 = 0.0
        # BUG F FIX: saved training-error bounds for stable normalisation
        self._train_error_min:  float                 = 0.0
        self._train_error_max:  float                 = 1.0
        self.feature_names:     list[str]             = MODEL_FEATURES
        self.training_losses:   list[float]           = []

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            logger.warning("AE — missing features (filled 0): %s", missing)
        return df.reindex(columns=self.feature_names, fill_value=0).values.astype(float)

    def _recon_errors(self, Xs: np.ndarray) -> np.ndarray:
        """
        Per-sample MSE between clipped scaled input and reconstructed output.
        BUG D FIX: clip extended to ±10 to preserve anomaly signal.
        """
        Xs_clipped = np.clip(Xs, -_CLIP_RANGE, _CLIP_RANGE)
        fwd = _forward(self._w, Xs_clipped)
        return ((Xs_clipped - fwd["z4"]) ** 2).mean(axis=1)

    def _get_normal_mask(
        self,
        features_df: pd.DataFrame,
        labels_path: Path,
    ) -> np.ndarray:
        """
        Return a boolean mask of shape (len(features_df),) that is True for
        normal users. Falls back to all-True if the label file is unavailable.

        BUG E FIX: used to filter training data to normal users only.
        """
        try:
            labels_df = pd.read_parquet(labels_path)[["user_id", "is_anomalous"]]
            merged    = features_df[["user_id"]].reset_index(drop=True).merge(
                labels_df, on="user_id", how="left"
            )
            is_normal = (~merged["is_anomalous"].fillna(False)).values
            n_normal  = int(is_normal.sum())
            n_total   = len(is_normal)
            logger.info(
                "Normal-user filter: %d / %d users selected for AE training.",
                n_normal, n_total,
            )
            if n_normal < 10:
                logger.warning(
                    "Too few normal users found (%d) — falling back to full "
                    "training set. Check that labels were generated with the "
                    "same ForexDataGenerator run.",
                    n_normal,
                )
                return np.ones(n_total, dtype=bool)
            return is_normal
        except FileNotFoundError:
            logger.warning(
                "Label file not found at '%s' — training on full dataset. "
                "Run 'python main.py generate' first.",
                labels_path,
            )
            return np.ones(len(features_df), dtype=bool)

    def _normal_only_errors(
        self,
        errors: np.ndarray,
        features_df: pd.DataFrame,
        labels_path: Path,
    ) -> np.ndarray:
        """
        Return reconstruction errors for normal users only.
        Falls back to full error array if labels are unavailable.
        BUG A FIX: threshold calibrated on normal population only.
        """
        is_normal = self._get_normal_mask(features_df, labels_path)
        normal_errors = errors[is_normal]
        if len(normal_errors) < 10:
            logger.warning(
                "Fewer than 10 normal errors available — using full error "
                "array for threshold calibration."
            )
            return errors
        return normal_errors

    def train(
        self,
        features_df: pd.DataFrame,
        labels_path: Optional[Path] = None,
    ) -> "AutoencoderDetector":
        """
        Train the autoencoder and calibrate the anomaly threshold.

        BUG E FIX: network weights are trained on NORMAL users only.
            The AE learns the manifold of normal behaviour. Anomalous users
            are excluded from _train_autoencoder() so their patterns never
            contaminate the weights. Labels are loaded here solely for this
            filtering step — they do not influence weight updates.

        BUG A FIX (retained): threshold is set at the (1 - contamination)
            percentile of NORMAL-user reconstruction errors only.

        BUG F FIX: training-error min/max are saved for use in score().

        Parameters
        ----------
        features_df : pd.DataFrame  — full training feature matrix.
        labels_path : Path, optional — override for user_labels.parquet path.
        """
        lpath = labels_path or self.labels_path
        X_all = self._prepare_X(features_df)
        logger.info("AE training on feature matrix %s …", X_all.shape)

        self._scaler = RobustScaler()
        Xs_all = self._scaler.fit_transform(X_all)

        # BUG E FIX: filter to normal users before training
        is_normal = self._get_normal_mask(features_df, lpath)
        Xs_normal = Xs_all[is_normal]
        logger.info(
            "BUG E FIX: training AE on %d normal users only (excluded %d anomalous).",
            int(is_normal.sum()), int((~is_normal).sum()),
        )

        # BUG C FIX: larger network capacity (32/16 instead of 16/8)
        self._w = _build_autoencoder(Xs_all.shape[1], self.hidden_dim, self.encoding_dim)
        self.training_losses = _train_autoencoder(
            self._w, Xs_normal,          # ← NORMAL users only (BUG E FIX)
            epochs=self.epochs,
            lr=self.lr,
            val_fraction=0.1,
            patience=30,
        )

        # Compute errors on the FULL training set (to set normalisation bounds
        # and threshold; anomalous users' errors should be high here now)
        all_errors = self._recon_errors(Xs_all)

        if np.isnan(all_errors).any():
            logger.error("Reconstruction errors contain NaN — training failed.")
            self._threshold        = 0.0
            self._train_error_min  = 0.0
            self._train_error_max  = 1.0
            return self

        # BUG F FIX: save training error bounds for stable score normalisation
        self._train_error_min = float(all_errors.min())
        self._train_error_max = float(all_errors.max())
        logger.info(
            "Training error bounds saved: min=%.6f  max=%.6f",
            self._train_error_min, self._train_error_max,
        )

        # BUG A FIX: calibrate threshold on normal-user errors only
        normal_errors = all_errors[is_normal]
        pct = (1.0 - self.contamination) * 100.0
        self._threshold = float(np.percentile(normal_errors, pct))

        n_flagged_train = int((all_errors > self._threshold).sum())
        logger.info(
            "AE trained. All-user recon error: mean=%.5f std=%.5f | "
            "Normal-only error: mean=%.5f std=%.5f | "
            "Threshold(%.0f%%ile of normals)=%.6f | "
            "Flagged on full train: %d / %d  (expected ≈ %d anomalous)",
            all_errors.mean(), all_errors.std(),
            normal_errors.mean(), normal_errors.std(),
            pct, self._threshold,
            n_flagged_train, len(all_errors),
            int((~is_normal).sum()),
        )
        return self

    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score users. Returns anomaly_score, reconstruction_error, is_anomaly.

        BUG F FIX: anomaly_score is normalised using training-error bounds
        (self._train_error_min / _train_error_max), not the current batch's
        min/max. This makes scores stable and comparable across batches, and
        compatible with the Isolation Forest anomaly_score in the ensemble.
        Scores outside [0, 1] are clipped — a score > 1.0 means the user's
        reconstruction error exceeds the worst training error, a strong signal.
        """
        if self._w is None or self._scaler is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X  = self._prepare_X(features_df)
        Xs = self._scaler.transform(X)

        errors = self._recon_errors(Xs)

        # BUG F FIX: use fixed training bounds, not batch min/max
        err_range = self._train_error_max - self._train_error_min
        if err_range > 0:
            normed = (errors - self._train_error_min) / err_range
            normed = np.clip(normed, 0.0, 1.0)
        else:
            normed = np.zeros_like(errors)

        out = features_df[["user_id"]].copy()
        out["anomaly_score"]        = np.round(normed, 4)
        out["reconstruction_error"] = np.round(errors, 6)
        out["is_anomaly"]           = errors > self._threshold

        n_anom = int(out["is_anomaly"].sum())
        logger.info(
            "AE scored %d users — flagged: %d (%.1f%%)",
            len(out), n_anom, 100.0 * n_anom / max(len(out), 1),
        )
        return out

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "weights":          self._w,
                "threshold":        self._threshold,
                "train_error_min":  self._train_error_min,
                "train_error_max":  self._train_error_max,
                "training_losses":  self.training_losses,
                "hidden_dim":       self.hidden_dim,
                "encoding_dim":     self.encoding_dim,
            }, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(
            "AE saved -> %s  (threshold=%.6f, error_range=[%.6f, %.6f])",
            self.model_path, self._threshold,
            self._train_error_min, self._train_error_max,
        )

    def load(self) -> "AutoencoderDetector":
        with open(self.model_path, "rb") as f:
            d = pickle.load(f)
        self._w              = d["weights"]
        self._threshold      = d["threshold"]
        # BUG F FIX: load saved training error bounds (backwards-compatible default)
        self._train_error_min = d.get("train_error_min", 0.0)
        self._train_error_max = d.get("train_error_max", 1.0)
        self.training_losses  = d.get("training_losses", [])
        self.hidden_dim       = d.get("hidden_dim", 32)
        self.encoding_dim     = d.get("encoding_dim", 16)
        with open(self.scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        logger.info(
            "AE loaded from %s  (threshold=%.6f, error_range=[%.6f, %.6f])",
            self.model_path, self._threshold,
            self._train_error_min, self._train_error_max,
        )
        return self
