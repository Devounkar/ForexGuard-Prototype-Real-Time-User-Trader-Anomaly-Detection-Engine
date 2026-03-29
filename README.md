# ForexGuard — Real-Time Forex Anomaly Detection Engine

> Detects suspicious user and trader behaviour across client portal and trading terminal activity, generates explainable risk alerts, and exposes results via a FastAPI REST layer.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Setup Instructions](#3-setup-instructions)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [API Reference](#5-api-reference)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Selection & Justification](#7-model-selection--justification)
8. [Evaluation Results](#8-evaluation-results)
9. [Explainability](#9-explainability)
10. [Streaming Architecture](#10-streaming-architecture)
11. [Assumptions, Trade-offs & Limitations](#11-assumptions-trade-offs--limitations)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  ForexDataGenerator → portal_events.parquet + trade_events.parquet   │
│                      + user_labels.parquet (ground truth, separate)  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                    FEATURE ENGINEERING                                │
│  FeatureEngineer.build_features()                                     │
│  Portal features + Trading features → 40+ per-user feature vectors   │
│  (rate-normalised, z-scored, inter-event timing, PnL volatility,     │
│   HHI concentration, dormancy + withdrawal patterns)                  │
└──────────────┬──────────────────────────────┬────────────────────────┘
               │                              │
┌──────────────▼──────────┐      ┌────────────▼────────────────────────┐
│   BASELINE MODEL         │      │   ADVANCED MODEL                    │
│   Isolation Forest       │      │   Bottleneck Autoencoder            │
│   (sklearn, RobustScaler)│      │   (NumPy/PyTorch, normal-only train) │
│   contamination="auto"   │      │   Adam, early stopping, stable norm  │
└──────────────┬──────────┘      └────────────┬────────────────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼────────────────────────┐
│                    ENSEMBLE LAYER  (evaluator.py)                     │
│  Weighted score blend (w_IF=0.5, w_AE=0.5)                           │
│  OR-logic flagging: anomalous if EITHER model flags                   │
│  Missing scores → per-model mean imputation (not zero)               │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                  EXPLAINABILITY  (explainability.py)                  │
│  RuleBasedExplainer — 25 population-percentile rules                 │
│  Triggers per user → narrative text (no SHAP required)               │
│  Fitted on train split only (no leakage)                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                    ALERT SYSTEM  (alert_system.py)                    │
│  Structured JSON alerts → alerts.jsonl                               │
│  Severity: LOW / MEDIUM / HIGH / CRITICAL  (risk score thresholds)  │
│  Schema: alert_id, user_id, risk_score, severity, narrative,         │
│          features_triggered, triggered_rules, timestamp              │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
       ┌───────────────────┴──────────────────────┐
       │                                          │
┌──────▼──────────────────┐         ┌─────────────▼──────────────────┐
│  REST API  (api.py)      │         │  STREAMING  (streamer.py)       │
│  FastAPI + uvicorn       │         │  EventStreamer (generator)       │
│  /health  /score         │         │  StreamingPipeline              │
│  /predict /alerts        │         │  score_every_n=1000 events      │
│  /users/{id}/risk        │         │  async batch simulation         │
└─────────────────────────┘         └────────────────────────────────┘
```

The system is designed so every component has a clean interface. The streaming pipeline, batch pipeline, and API all share the same `FeatureEngineer`, `AnomalyDetector`, `AutoencoderDetector`, `RuleBasedExplainer`, and `AlertSystem` instances.

---

## 2. Project Structure

```
forex_anomaly/
│
├── main.py                    # CLI entry point (generate/train/evaluate/stream/run)
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic event generation (v5 — UUID IDs, leakage-free)
│   ├── feature_engineering.py # 40+ per-user features (v3 — extended set)
│   ├── model.py               # Isolation Forest with RobustScaler (v2)
│   ├── autoencoder.py         # Bottleneck Autoencoder in NumPy (v8)
│   ├── evaluator.py           # Metrics, ensemble, ground-truth loading (v4)
│   ├── explainability.py      # Rule-based explainer, 25 rules (v2)
│   ├── alert_system.py        # Alert generation, JSONL persistence
│   ├── streamer.py            # EventStreamer + StreamingPipeline
│   └── api.py                 # FastAPI endpoints
│
├── configs/
│   └── config.yaml            # All hyperparameters and paths
│
├── data/                      # Generated at runtime (gitignored)
│   ├── portal_events.parquet
│   ├── trade_events.parquet
│   ├── user_labels.parquet    # ONLY source of ground truth
│   ├── features.parquet
│   ├── train_features.parquet
│   ├── test_features.parquet
│   ├── if_scores.parquet
│   ├── ae_scores.parquet
│   ├── ensemble_scores.parquet
│   ├── model_comparison.csv
│   ├── alerts.jsonl
│   └── stream_alerts.jsonl
│
├── models/                    # Saved model artefacts (gitignored)
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── autoencoder.pkl
│   └── ae_scaler.pkl
│
└── requirements.txt
```

---

## 3. Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd forex_anomaly
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
pyarrow>=12.0
pyyaml>=6.0
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.0
```

### Step 4 — Create the config file

Create `configs/config.yaml`:

```yaml
data_generation:
  n_normal_users: 1000
  n_anomalous_users: 80
  events_per_normal_user: 50
  start_date: "2024-01-01"
  end_date: "2024-03-31"
  output_path: "data/events.parquet"

model:
  model_path: "models/isolation_forest.pkl"
  scaler_path: "models/scaler.pkl"
  n_estimators: 200
  contamination: "auto"
  max_samples: "auto"
  random_state: 42

alerts:
  risk_score_threshold: 0.60
  output_path: "data/alerts.jsonl"
```

---

## 4. Running the Pipeline

All stages are controlled through `main.py`:

```bash
# Generate synthetic data (~50,000+ events)
python main.py generate

# Feature engineering + train Isolation Forest + Autoencoder
python main.py train

# Evaluate both models + ensemble on held-out test set
python main.py evaluate

# Run simulated streaming pipeline
python main.py stream

# Run ALL stages end-to-end in one command
python main.py run
```

> **Important:** Always re-run `generate → train → evaluate` if you replace any source file. Old `.pkl` files from previous runs are incompatible with new UUID-based user IDs.

### Launch the REST API

```bash
# Recommended
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# If uvicorn is on PATH
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/alerts
```

---

## 5. API Reference

All endpoints return a consistent JSON envelope:
```json
{ "status": "ok" | "error", "data": ..., "message": "..." }
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check, model load status |
| POST | `/score` | Score a user by `user_id` using saved features |
| POST | `/predict` | Score a raw feature vector (no historical data needed) |
| POST | `/score/batch` | Score multiple `user_id`s in one call |
| GET | `/alerts` | Retrieve generated alerts (paginated, filterable by severity) |
| GET | `/alerts/{alert_id}` | Retrieve a single alert by ID |
| GET | `/users/{user_id}/risk` | Full risk profile for one user |

Models are loaded once at startup via FastAPI's `lifespan` event to avoid cold-start latency per request.

---

## 6. Feature Engineering

`FeatureEngineer.build_features()` aggregates raw events into a **per-user feature vector** of 40+ features across five categories:

### 6.1 Portal / Access Features
| Feature | Description |
|---|---|
| `login_count_per_day` | Rate-normalised login volume |
| `unique_ips` | Distinct IP addresses used |
| `unique_regions` | Distinct geographic regions |
| `geo_change_rate` | Regions per login (rapid switching) |
| `night_login_ratio` | Fraction of logins between 00:00–06:00 UTC |
| `avg_inter_event_s` | Mean time between events (bot detection) |
| `min_inter_event_s` | Minimum inter-event gap (machine-speed clicks) |
| `withdrawal_deposit_ratio` | Total withdrawn / total deposited |
| `withdrawal_surge_ratio` | Latest withdrawal vs user's own baseline |
| `sensitive_mod_count_per_day` | Bank account / email / phone / leverage changes per day |
| `unverified_doc_rate` | Fraction of uploaded documents not verified |
| `high_priority_ticket_rate` | High/critical support tickets ratio |

### 6.2 Trading Features
| Feature | Description |
|---|---|
| `max_trades_5min` | Max trades in any 5-minute window (HFT burst) |
| `max_leverage` | Highest leverage used |
| `max_margin_usage` | Peak margin usage percentage |
| `margin_spike_rate` | Frequency of sudden margin spikes |
| `trade_frequency_per_hour` | Trades per hour of observed activity |
| `std_pnl` | Standard deviation of per-trade PnL (speculation signal) |
| `pnl_volatility_ratio` | std_pnl / (|total_pnl| + 1) — scale-invariant |
| `herfindahl_pairs` | HHI index over currency pairs (single-instrument concentration) |

### 6.3 Within-User Behavioural Drift (Z-Scores)
These capture **sudden shifts within a single user's own history**, independent of population norms:
| Feature | Description |
|---|---|
| `last_trade_size_zscore` | Last trade size vs user's own mean ± std |
| `last_margin_zscore` | Last margin usage vs user's own baseline |
| `last_leverage_zscore` | Last leverage vs user's own baseline |

### 6.4 Churn & Dormancy Patterns
| Feature | Description |
|---|---|
| `churn_financial_ratio` | (deposits + withdrawals) / total trades — captures deposit→no trade→withdrawal abuse |
| `days_since_last_trade` | Dormancy gap before latest portal activity |
| `dormancy_withdrawal_score` | days_since_last_trade × withdrawal_surge_ratio |

All count features are **rate-normalised** (divided by observation span in days) to prevent users with longer histories from appearing more anomalous simply due to more events.

---

## 7. Model Selection & Justification

### 7.1 Why Isolation Forest (Baseline)?

Isolation Forest was chosen as the baseline because:

- **Designed for unsupervised anomaly detection.** Unlike SVM or LOF, it does not need a distance metric across all pairs of points, making it efficient on high-dimensional feature spaces like this one (40+ features).
- **Scales linearly** with data size. LOF's O(n²) complexity would be prohibitive in a near-real-time streaming context.
- **Robust to irrelevant features.** IF randomly subsamples both data points and features in each tree, so features that carry no signal get averaged out rather than polluting the score.
- **`contamination="auto"`** was deliberately chosen over a fixed value. Setting contamination to the true anomaly fraction (e.g. 0.08) would give the model a free hint about the class balance and artificially inflate evaluation metrics. With `"auto"`, the IF decision boundary is learned independently of the true anomaly rate.
- **RobustScaler** pre-processing handles outlier-heavy financial data (extreme trade sizes, leverage spikes) better than StandardScaler because it uses medians and IQR rather than means and variance.

### 7.2 Why a Bottleneck Autoencoder (Advanced)?

A Bottleneck Autoencoder was chosen as the advanced model over LSTM Autoencoder, Transformer, or VAE because:

- **Complementary error profile to IF.** Isolation Forest scores anomalies based on how easy they are to isolate in feature space (structural outliers). The Autoencoder scores based on reconstruction error — how poorly the network can reconstruct an input from the compressed bottleneck representation. These two signals are genuinely different and the ensemble benefits from both.
- **Normal-only training.** The AE is trained exclusively on normal users' feature vectors. This is the core of its anomaly detection ability: the network learns the manifold of normal behaviour, so anomalous inputs that fall off that manifold reconstruct poorly. Training on the full dataset (including anomalies) would cause the AE to learn anomalous patterns too, collapsing the reconstruction error gap.
- **Stable score normalisation.** The AE normalises reconstruction errors against the min/max of the *training distribution*, not the current scored batch. This makes `anomaly_score` a consistent, absolute value across batches — essential for meaningful ensemble blending with IF's `decision_function` output.
- **Architecture:** `input(40) → Dense(32, ReLU) → Bottleneck(16, ReLU) → Dense(32, ReLU) → output(40, Linear)`. He initialisation, Adam optimiser (lr=0.001), gradient clipping, and early stopping with patience=30 on a 10% validation split ensure stable training.
- **Why not LSTM Autoencoder?** LSTM autoencoders are designed for sequential, time-ordered data (e.g., per-trade time series). The feature matrix here is per-user aggregated statistics — a static snapshot, not a sequence. Applying an LSTM to a feature vector would add sequence modelling overhead with no benefit and would require careful windowing that is not appropriate for this aggregated representation.
- **Why not VAE?** VAE would be appropriate if generative modelling or latent space sampling were needed (e.g., for synthetic data augmentation). For pure anomaly detection, the reconstruction error from a plain bottleneck AE is sufficient and more interpretable.
- **Why not Transformer?** Transformer attention would add significant computational cost with marginal benefit on a 40-feature flat vector. Transformers excel at positional relationships across long sequences, which is not the structure of this data.

### 7.3 Why an Ensemble?

| Model alone | Strength | Weakness |
|---|---|---|
| Isolation Forest | High precision (1.000) | Misses some anomalies (Recall=0.781) |
| Autoencoder | Perfect recall (1.000) | Too many false positives (Precision=0.516) |

The ensemble uses **OR-logic flagging**: a user is flagged as anomalous if *either* model flags them. This directly addresses the complementary weaknesses above — IF's misses (FN=7) are caught by the AE, while the ensemble score still provides a continuous risk measure for prioritisation. Mean imputation (not zero) is used for users scored by only one model, preventing score suppression at the boundary.

---

## 8. Evaluation Results

Evaluation is performed on a **held-out 20% test set** (stratified by label), never seen during training.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **Isolation Forest** | **1.0000** | 0.7812 | 0.8772 | 0.9920 | 0.9520 | 25 | 0 | 7 |
| Autoencoder | 0.5161 | **1.0000** | 0.6809 | 0.9926 | 0.9426 | 32 | 30 | 0 |
| **Ensemble (IF+AE)** | 0.5161 | **1.0000** | 0.6809 | **0.9937** | **0.9607** | 32 | 30 | 0 |

**Reading the results:**

- **Isolation Forest** achieves **perfect precision** (zero false positives) — every user it flags is genuinely anomalous. The cost is missing 7 anomalous users (recall 0.78). In a compliance context, this means IF alerts are high-confidence and can be acted on immediately, but should not be the only line of defence.

- **Autoencoder** achieves **perfect recall** (zero false negatives) — it catches every anomalous user. The cost is 30 false positives (precision 0.52). In a compliance context, AE alerts require a second-layer review but ensure no genuine fraud case is missed.

- **Ensemble** inherits the AE's perfect recall (the OR-logic preserves coverage) while gaining a slightly higher ROC-AUC (0.9937) and PR-AUC (0.9607) than either model alone. The ensemble's continuous `ensemble_score` allows analysts to triage by risk score even when both models flag the same user.

- **ROC-AUC ≥ 0.992 across all models** indicates that the ranking quality (the ability to sort users by anomalousness) is very high, regardless of the chosen binary threshold.

Ground truth is loaded from `data/user_labels.parquet` — a file written by the data generator that stores `{user_id, is_anomalous}`. User IDs are opaque UUID-based strings; no label information is encoded in the ID itself, preventing any form of ID-prefix leakage into features or evaluation logic.

---

## 9. Explainability

`RuleBasedExplainer` provides human-readable explanations without requiring SHAP:

- **25 rules** are defined, each mapping a feature to a population percentile threshold and a natural-language label (e.g., `"abnormally high trade frequency"`, `"near-instant inter-event timing suggesting automated activity"`).
- The explainer is **fitted on the training split only** — population percentile thresholds are computed from training users, not the full dataset, preventing leakage into the test evaluation.
- For each anomalous user, the top-3 most-triggered rules (by how far the feature value exceeds its threshold) are selected and assembled into a narrative.

**Example alert output:**
```json
{
  "alert_id": "f3a2...",
  "user_id": "U-9c3d7a...",
  "risk_score": 0.9341,
  "severity": "CRITICAL",
  "alert": "Anomalous behaviour detected: extreme PnL volatility across trades and deposit/withdrawal activity disproportionate to trading volume.",
  "features_triggered": ["std_pnl", "churn_financial_ratio"],
  "triggered_rules": [
    { "feature": "std_pnl", "value": 18420.3, "threshold": 4231.1, "label": "extreme PnL volatility across trades", "excess_ratio": 3.35 },
    { "feature": "churn_financial_ratio", "value": 12.4, "threshold": 3.1, "label": "deposit/withdrawal activity disproportionate to trading volume", "excess_ratio": 3.0 }
  ],
  "timestamp": "2024-06-01T23:31:16+00:00"
}
```

**Severity mapping:**
| Score Range | Severity |
|---|---|
| ≥ 0.90 | CRITICAL |
| 0.75 – 0.89 | HIGH |
| 0.60 – 0.74 | MEDIUM |
| < 0.60 | LOW (not alerted) |

---

## 10. Streaming Architecture

The streaming pipeline simulates real-time processing using Python generators (async batch simulation, as per spec option B):

```
EventStreamer (generator)
    ↓ yields events in chronological order
StreamingPipeline._ingest()
    ↓ buffers portal + trade events per user
Every score_every_n=1000 events:
    StreamingPipeline._score_cycle()
        ↓ FeatureEngineer.build_features() on current buffer
        ↓ AnomalyDetector.score_single()
        ↓ RuleBasedExplainer.explain()
        ↓ AlertSystem.generate_single_alert()
```

The design is interface-compatible with Kafka: `EventStreamer` can be replaced by a Kafka consumer that yields the same `dict` per event, with no changes to the downstream pipeline. The `StreamingFeatureBuffer` in `feature_engineering.py` maintains a rolling 1-hour window for latency-sensitive scoring.

---

## 11. Assumptions, Trade-offs & Limitations

### Assumptions

- **Static feature vectors per user.** Features are aggregated over the full observation window rather than computed on a sliding window. This is appropriate for batch scoring and for streaming where a reasonable buffer has accumulated, but it means very recent behavioural shifts may take `score_every_n` events to propagate.
- **Labels are withheld from models.** `user_labels.parquet` is used only for evaluation (ground truth comparison) and for Autoencoder training filtering (to ensure only normal users train the network). Labels are never passed as features.
- **Synthetic data approximates real forex behaviour.** Anomaly patterns (HFT burst, geo-hopping, deposit-churn, dormancy + withdrawal) are injected with realistic overlapping magnitudes to ensure model difficulty is representative, but real data would have more complex correlations.
- **Ground truth is binary.** In production, anomaly labels would have more granularity (e.g., fraud type, investigation outcome). The current binary label is a simplification.

### Trade-offs

| Decision | What was gained | What was traded off |
|---|---|---|
| Rule-based explainability over SHAP | No dependency on model internals; explainer is consistent across both models | Rules must be hand-authored; cannot discover new patterns automatically |
| OR-logic ensemble flagging | Perfect recall; no fraud case missed | 30 additional false positives; review queue is larger |
| Normal-only AE training | Clean anomaly detection signal | Requires a separate label file during training (semi-supervised dependency) |
| Async batch simulation over Kafka | No infrastructure dependency; runs out of the box | True sub-second latency requires a real message broker |
| `contamination="auto"` | No information leakage about true anomaly rate | IF threshold is less tuned; may miss borderline cases |

### Limitations

- **No graph / network-level anomaly detection.** The spec mentions IP hub detection, synchronized trading across accounts, and collusion rings (Section 8.5). These require a graph representation (e.g., shared-IP bipartite graph) and are not implemented in this version.
- **No temporal sliding windows.** Features are computed over the full user history. A production system would use rolling 1-hour, 24-hour, and 7-day windows simultaneously to catch both rapid spikes and slow drift.
- **No MLflow / W&B tracking.** Experiment metadata and model versioning are not tracked. The recommended next step would be wrapping `cmd_train()` in an MLflow run.
- **No real message broker.** The streaming layer uses Python generators. Kafka or Redpanda integration would require replacing `EventStreamer` with a consumer, which the architecture already supports by interface.
- **Autoencoder is NumPy-based.** The current implementation uses a hand-written forward/backward pass in NumPy for portability. Replacing it with a PyTorch `nn.Module` would improve training speed on GPU and simplify adding LSTM / VAE variants in future.
- **No hosted demo.** The system runs locally. Deployment to HuggingFace Spaces, Render, or AWS would require containerisation (a `Dockerfile` and `docker-compose.yml` can be added straightforwardly given the modular structure).
