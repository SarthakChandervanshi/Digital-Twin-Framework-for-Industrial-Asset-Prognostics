# Digital Twin Framework for Industrial Asset Prognostics

Data-driven prognostics pipeline for Remaining Useful Life (RUL) estimation on NASA C-MAPSS FD001, built around:

- sequence modeling with LSTM
- quantile regression for uncertainty intervals (low / median / high)
- confidence estimation from interval width
- SHAP-based interpretability for feature importance

## Live project

- **Dashboard (live):** [https://rul-dashboard-app.vercel.app/](https://rul-dashboard-app.vercel.app/)

![Live dashboard preview](./assets/live-dashboard-preview.png)

## Project objective

Industrial assets degrade under noisy and non-stationary operating conditions. A single RUL number is often not enough for safe maintenance decisions.

This project estimates:

- a **central RUL prediction** (median)
- a **prediction interval** (lower and upper quantiles)
- a **confidence signal** derived from interval sharpness

The goal is to support risk-aware maintenance planning, not only point forecasting.

## Dataset

- **Benchmark:** NASA C-MAPSS FD001
- **Input:** multivariate engine sensor time series
- **Task:** estimate remaining cycles to failure

Core files are expected under `data/CMAPSSData/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

## Method summary

1. **Preprocessing**
  - remove low-information columns
  - create capped RUL target
  - smooth sensor streams (Kalman filtering)
2. **Feature engineering**
  - trend (`tr_`*), rate-of-change (`rc_*`), rolling statistics (`rs_*`)
3. **Model**
  - stacked LSTM over sliding windows (`SEQ_LEN = 30`)
  - three quantile heads (0.1, 0.5, 0.9)
4. **Training**
  - quantile (pinball) loss
  - validation-based training controls
5. **Evaluation**
  - point metrics + asymmetric risk metric + interval quality
6. **Interpretability**
  - global SHAP on median head

For narrative details, see:

- `artifacts/docs/problem.md`
- `artifacts/docs/method.md`
- `artifacts/docs/results.md`
- `artifacts/docs/literature.md`

## Current FD001 results (precomputed)

From `artifacts/data/metrics.json` (scope: `last_window_per_test_engine`, `n_engines = 100`):

- **RMSE:** `12.510066`
- **NASA_score:** `274.41338`
- **R2:** `0.902544`
- **Coverage:** `0.83`
- **Mean interval width:** `32.859956`
- **Within 10%:** `68.0`
- **Within 20%:** `88.0`

## Key artifacts

- `artifacts/data/metrics.json` - aggregate performance and interval quality
- `artifacts/data/predictions.json` - per-engine low/mid/high, width, confidence, interval hit
- `artifacts/data/shap_global.json` - global SHAP ranking
- `artifacts/models/fd001/preprocess_config.json` - preprocessing metadata
- `notebooks/model/fd001/config.json` - model/training configuration
- `notebooks/model/fd001/weights.weights.h5` - trained model weights
- `dashboard/public/data/` - dashboard-ready static data payloads

## Repository structure

```text
.
|- artifacts/
|  |- data/
|  |- docs/
|  `- models/
|- dashboard/
|  `- public/data/
|- data/CMAPSSData/
|- notebooks/
|  |- fd001.ipynb
|  `- model/fd001/
|- pyproject.toml
`- README.md
```

## Environment setup

This project uses Poetry (`pyproject.toml`).

```bash
poetry install
poetry shell
```

Python requirement: `>=3.12`

## Reproduce pipeline (notebook-driven)

Primary workflow is in:

- `notebooks/fd001.ipynb`

Typical execution:

1. open the notebook
2. run all cells in order
3. export artifacts to `artifacts/data/` and model outputs to `notebooks/model/fd001/`

## Dashboard usage

- Use the live app for full presentation and interaction:
  - [https://rul-dashboard-app.vercel.app/](https://rul-dashboard-app.vercel.app/)
- Dashboard payloads are precomputed and stored in:
  - `dashboard/public/data/engine_series/`
  - `dashboard/public/data/shap_local/`

## Why intervals + SHAP

- **Intervals** make uncertainty explicit for maintenance decisions.
- **NASA score** emphasizes operationally costly late errors.
- **SHAP** validates that predictions rely on degradation dynamics (trend/change features), improving trust and explainability.

## License

MIT (see `LICENSE`).

## Author

Sarthak Chandervanshi