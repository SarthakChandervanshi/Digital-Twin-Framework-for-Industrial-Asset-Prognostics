# Method

This document is stable narrative about **data, preprocessing, modeling, and explainability**. It does **not** repeat run-specific KPIs: **see `artifacts/data/metrics.json`** for RMSE, NASA score, R², coverage, interval width, and related fields on the agreed evaluation scope.

## Data

- **Dataset:** NASA C-MAPSS **FD001** (train: `train_FD001.txt`, test: `test_FD001.txt`, RUL labels for test: `RUL_FD001.txt`).
- **Train / validation split:** hold-out **engines** (not random rows)—training uses engines **1–80**, validation **81–100**, consistent with the FD001 notebook pipeline.
- **Test set:** FD001 test engines with piecewise RUL constructed from ground-truth remaining life at the last observed cycle.

## Preprocessing

- **Column drops:** sensors and settings with little or no variation are removed as in the notebook (same exclusions on train and test).
- **RUL target:** RUL is capped at **`RUL_CAP`** (fixed in the notebook) so the model focuses on the late-life regime used in the project.
- **Kalman smoothing:** per-engine, per-sensor 1D Kalman filter with **fixed** process noise **Q** and measurement noise **R** (documented in the notebook and chart metadata) to reduce measurement noise while preserving degradation trends; smoothed columns are named `*_kalman`.
- **Feature engineering:** rolling and trend-style features (e.g. `rs_*`, `tr_*`, `rc_*`) are applied consistently to train and test after Kalman.
- **Scaling:** **`StandardScaler`** is **fit on training rows only** and applied to train, validation, and test feature columns. Identifiers (`engine_id`, `time_in_cycles`) and target RUL are not scaled.

## Model

- **Architecture:** stacked **LSTM** with dense layers and a **three-output** head for quantiles (see exported Keras/ONNX artifacts under `notebooks/model/fd001/` or project model directory).
- **Training:** Adam with cosine learning-rate schedule, early stopping on validation loss, quantile (pinball) loss over the three heads.
- **Sequences:** sliding windows of length **`SEQ_LEN`** (30 cycles in the current setup) over scaled features; labels are RUL at the end of each window.
- **Inference:** predictions can be run from saved weights or **ONNX**; input rank is `(batch, SEQ_LEN, n_features)` matching training.

## Quantiles and evaluation scope

- **Quantiles:** **0.1, 0.5, 0.9**—interpreted as lower, median, and upper RUL under the learned distribution.
- **Paper-style metrics:** primary alignment with common C-MAPSS practice uses **one row per test engine**: the **last available window** per engine. That scope is recorded in **`metrics.json`** (`scope` field).

## Confidence score (dashboard)

For each engine in that summary:

- **`width`** = `rul_high − rul_low` (quantile band width).
- **`k`** = **median** of `width` over all engines in the same summary.
- **`confidence`** = **`exp(−width / k)`** so narrower bands receive higher scores (with a documented floor if `k` is degenerate).

Exact definitions and any exported metadata live alongside **`artifacts/data/predictions.json`** where applicable.

## SHAP (global importance)

- **Explained output:** **median quantile head only** (simplest and stable for dashboards)—not the low or high heads.
- **Model:** frozen trained network; same input tensor shape as inference.
- **Background:** random subset of **training** windows.
- **Explained set:** e.g. last window per test engine or a fixed random subset of test windows—must match what is saved in **`artifacts/data/shap_global.json`**.
- **Aggregation:** mean **absolute** SHAP values over explained samples and time steps, then **rank** features **1…K**.

## Where numbers live

| Artifact | Role |
|----------|------|
| `artifacts/data/metrics.json` | Point and interval metrics on the chosen scope |
| `artifacts/data/predictions.json` | Per-engine quantiles, width, confidence, coverage flags |
| `artifacts/data/shap_global.json` | Global SHAP ranking |
| `model/fd001/config.json` (or equivalent) | Training hyperparameters and shape metadata |
