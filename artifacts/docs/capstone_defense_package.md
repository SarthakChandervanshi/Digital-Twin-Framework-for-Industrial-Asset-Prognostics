# Capstone Defense Documentation Package
**Project:** Digital Twin Framework for Industrial Asset Prognostics (NASA C-MAPSS FD001)  
**Role framing:** Lead Research Assistant + Senior ML Engineer + Technical Architect + Defense Co-Pilot

---

## 1) Executive Project Overview

### What is a digital twin, how it works, and what it means in this project

**Definition (defense-ready):** A **digital twin** is a **virtual representation** of a physical asset (or fleet) that is **fed by real or simulated data** so operators and systems can **monitor condition**, **forecast behavior**, and **support decisions** as if they had a live software mirror of the asset—not only a spreadsheet of raw measurements. Industry definitions vary: some emphasize **3D models + IoT**; in **prognostics** the core is often a **behavioral twin**: data in, health or life predictions out.

**How a digital twin “works” in general:**  
1. **Sense / collect:** sensors and logs describe how the asset evolves over time.  
2. **Model:** rules, physics, or ML map that history to quantities of interest (here: **remaining useful life** and **uncertainty**).  
3. **Act / visualize:** dashboards, alerts, or maintenance workflows consume outputs; mature deployments **refresh** the twin as new data arrive (closed loop optional).

**How this project implements that idea:**  
- **Twin object:** Each **engine** is represented by **multivariate time-series windows** (sensor-derived features over consecutive cycles)—a **prognostic twin**, not a CAD geometry twin.  
- **Twin “brain”:** A **quantile LSTM** maps recent history to **RUL low / mid / high** plus derived **confidence** from interval width.  
- **Transparency:** **SHAP** ties predictions back to **which dynamics features** drove them (trust and debugging).  
- **Operator-facing layer:** **Exported JSON/CSV** and dashboard-oriented payloads show how a **fleet** view of risk and uncertainty would be consumed—this is the **decision-facing** half of the twin story in this repo.

**Scope to state honestly:** This capstone delivers the **ML pipeline, trained artifacts, explainability, and static dashboard data contracts**. A **full production** digital twin would typically add **live data ingestion**, **online inference** (e.g. API + ONNX), and **monitoring**—noted elsewhere as future work; here the twin is **demonstrated end-to-end as an artifact-driven prototype**.

### What this project is
A full ML pipeline that predicts **Remaining Useful Life (RUL)** for aircraft engines on NASA C-MAPSS FD001, with:
- **point prediction** (`rul_mid`)
- **uncertainty interval** (`rul_low`, `rul_high`)
- **confidence proxy** from interval width
- **interpretability** via SHAP (global + local)
- **deployed digital twin dashboard** for operational user interaction

### Main goal
Move from “single-number RUL prediction” to **risk-aware prognostics**: estimate not only *how much life is left* but *how uncertain that estimate is*.

### Real-world problem solved
Maintenance teams must decide when to inspect/replace assets under uncertainty. Overestimating RUL risks failure; underestimating causes unnecessary maintenance cost. This project gives **decision-grade outputs** (range + confidence + feature drivers).

### Why important
- Safety-critical systems (aviation, energy, heavy manufacturing)
- Asymmetric consequences of error
- Need for explainable models in high-stakes settings

### Industries and target users
- Aviation MRO / fleet reliability teams
- Industrial condition monitoring teams
- Predictive maintenance analysts
- PHM researchers and data scientists

### What is innovative here
- Quantile LSTM for **interval output** instead of only point regression
- A deployed dashboard experience that shows how users would consume this in production
- SHAP over sequence model outputs (median head) to validate model behavior

### Technical uniqueness
- End-to-end notebook pipeline -> trained model artifacts -> explainability -> deployed digital twin interface
- Explicit uncertainty mechanics: width, coverage, and confidence calculation
- Engine-wise evaluation protocol (`last_window_per_test_engine`) aligned with prognostics reporting

### Practical usefulness
Outputs map directly to triage:
- low `rul_mid` + wide interval = high-priority inspection
- narrow interval = higher trust in prediction

### Scalability
Current repo is notebook-centric; scalable path exists by modularizing:
- preprocessing service
- model inference service (ONNX runtime)
- periodic export jobs
- dashboard API layer

### Research contributions (within scope of this repo)
- Demonstrates quantile-based sequence prognostics on FD001
- Adds explainability and uncertainty-aware reporting
- Provides reproducible artifact-driven communication workflow

---

## 2) Background & Motivation

### Why selected
RUL is a classic PHM challenge with strong practical and research relevance. FD001 is a well-known benchmark enabling reproducible comparison.

### Industry pain points addressed
- Sensor noise and non-observable degradation states
- Decision-making under uncertainty
- Late-vs-early maintenance tradeoff

### Research gap this project tackles
Many models report only point metrics. This project adds:
- interval predictions
- confidence signal
- SHAP-based explainability

### Why AI/ML and this design
- Time-series degradation patterns are nonlinear -> sequence models (LSTM) are suitable
- Quantile heads naturally produce decision-facing uncertainty bands
- SHAP improves trust and interpretation

### Existing approach limitations (general)
- Point-only models hide uncertainty
- Hard to convert raw metrics into operational decisions
- Limited explainability in many deep models

### How this project improves
- Uncertainty-aware outputs
- Export package for dashboard deployment
- Structured artifacts (`metrics.json`, `predictions.json`, `shap_global.json`, local SHAP payloads)

### Related work framing (defense-ready)
Use this positioning:
- “Point forecasting literature is strong; my contribution is integrating **accuracy + uncertainty + interpretability + deployment-ready outputs** in one reproducible pipeline.”

---

## 3) Complete End-to-End Workflow

### Stage A — Data Input
**Input:**  
`data/CMAPSSData/train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`  
**Purpose:** Build supervised sequence regression problem.

### Stage B — Preprocessing
- Remove low-information/constant features
- Build RUL targets (with cap)
- Apply engine-wise smoothing (Kalman)
- Preserve consistent feature contracts in `preprocess_config.json`

**Output artifacts:**  
`artifacts/models/fd001/preprocess_config.json`, `scaler.joblib`

### Stage C — Feature Engineering
Creates dynamics-focused features:
- `tr_*` (trend/slope)
- `rc_*` (rate-of-change)
- `rs_*` (rolling stats)
- `si_*` (sensor interactions)

**Reason:** degradation is encoded in temporal change, not only absolute levels.

### Stage D — Windowing & Scaling
- StandardScaler fit on training data only
- Sliding windows (`seq_len = 30`)
- Shape metadata saved (`sequence_shapes.json`)

### Stage E — Modeling
**Model:** stacked LSTM with 3-output quantile head  
- q0.1 -> `rul_low`
- q0.5 -> `rul_mid`
- q0.9 -> `rul_high`

**Loss:** quantile (pinball) loss  
**Optimization:** Adam + schedule + callbacks (notebook controlled)

### Stage F — Evaluation
Primary scope: **last window per test engine**  
Metrics saved to `artifacts/data/metrics.json`:
- RMSE, R2, NASA_score
- coverage, mean_interval_width
- within_10_pct, within_20_pct
- weighted_MAE

### Stage G — Confidence & Calibration-facing outputs
From `predictions.json`:
- `width = rul_high - rul_low`
- `confidence = exp(-width / k)` where `k` is median width

Stored in:
- `artifacts/data/predictions.json`
- `artifacts/data/confidence_meta.json`

### Stage H — Explainability
- Global SHAP: `artifacts/data/shap_global.json`
- Local SHAP payloads per engine in `dashboard/public/data/shap_local/engine_*.json`

### Stage I — Export for dashboard consumption
Script:  
`notebooks/model/fd001/export_dashboard_pack.py`

Exports:
- `model_metrics.json`
- `fleet_predictions.csv`
- `shap_global.csv`
- `engine_timeseries.csv`
- `shap_local.csv`
- `raw_engine_timeseries.csv`
- `meta.json`

---

## 4) File-by-File Analysis (Important Files)

### Root-level
- `README.md`: project narrative, metrics, architecture diagram, artifact references, live link
- `pyproject.toml`: dependency and environment contract (TensorFlow, SHAP, Optuna, ONNX stack)
- `.gitignore`: general ignore rules

### Data
- `data/CMAPSSData/*`: raw benchmark files (train/test/RUL for FD001–FD004)

### Notebook & model files
- `notebooks/fd001.ipynb`: full research pipeline (EDA -> preprocessing -> training -> eval -> SHAP -> exports)
- `notebooks/model/fd001/config.json`: final run hyperparameters
- `notebooks/model/fd001/model_architecture.json`: serialized architecture
- `notebooks/model/fd001/weights.weights.h5`: trained model weights
- `notebooks/model/fd001/model.onnx`: portable inference format
- `notebooks/model/fd001/export_dashboard_pack.py`: post-processing and export orchestration

### Artifacts (evaluation and model contracts)
- `artifacts/models/fd001/preprocess_config.json`: feature list + constants (including RUL cap)
- `artifacts/models/fd001/sequence_shapes.json`: data shape metadata
- `artifacts/models/fd001/scaler.joblib`: preprocessing scaler
- `artifacts/data/metrics.json`: aggregate metrics
- `artifacts/data/predictions.json`: engine-level prediction outputs
- `artifacts/data/confidence_meta.json`: confidence formula metadata
- `artifacts/data/shap_global.json`: ranked global feature importance

### Docs
- `artifacts/docs/problem.md`: motivation/problem framing
- `artifacts/docs/method.md`: method and protocol details
- `artifacts/docs/results.md`: interpretation and caveats
- `artifacts/docs/literature.md`: comparison framing guidance

### Dashboard payload folders
- `dashboard/public/data/engine_series/engine_*.json` (100 files): per-engine cycle-level payload
- `dashboard/public/data/shap_local/engine_*.json` (100 files): per-engine local SHAP payload

### Export pack
- `artifacts/data/dashboard_export/*`: consolidated flat files for downstream dashboard/reporting

### Crucial defense point from file audit
This repo contains **pipeline + artifacts + static dashboard data**, but **not** a full frontend app codebase or backend API service implementation.

---

## 5) Technical Architecture Analysis

### Actual architecture present in this repository
1. **Notebook-centric ML pipeline**
2. **Artifact persistence layer** (JSON/CSV/model files)
3. **Static data packaging layer** (`export_dashboard_pack.py`)
4. **Dashboard payload files** (static JSON/CSV)

### What is *not* in this repo
- No backend API server implementation
- No frontend source app (React/Next/Vite/etc.) in this snapshot

### Why chosen (inferred from implementation)
- Fast research iteration in notebook
- Clear artifact contracts for reproducibility and communication
- Decoupled data export layer supports dashboard in another project/repo

### Modularity
Moderate:
- Preprocessing/model/eval mostly in notebook (high flexibility, lower production modularity)
- Export script is modular and reusable

### Scalability status
- Good for capstone/prototype
- Production would require:
  - moving notebook logic into versioned Python modules
  - serving model via API
  - scheduled batch inference and monitoring

---

## 6) Machine Learning / AI Explanation

### Model family used
Quantile regression with LSTM sequence encoder.

### Intuitive explanation
The LSTM reads recent engine behavior (30-cycle window) and outputs a conservative estimate (`low`), central estimate (`mid`), and optimistic bound (`high`).

### Mathematical core

#### Quantile loss (pinball)
For target $y$, prediction $\hat{y}_q$, quantile $q \in (0,1)$:

$$
L_q(y,\hat{y}_q)=\max\left(q(y-\hat{y}_q),(1-q)(\hat{y}_q-y)\right)
$$
$$
L_q(y,\hat{y}_q)=
\begin{cases}
q(y-\hat{y}_q), & y \ge \hat{y}_q \\
(1-q)(\hat{y}_q-y), & y < \hat{y}_q
\end{cases}
$$
```text
L_q(y, y_hat_q) = max( q * (y - y_hat_q), (1 - q) * (y_hat_q - y) )
```
- If underpredicting at high quantile, penalty differs from overpredicting.
- Enables direct learning of quantile bounds.

#### Interval width
$$
\text{width} = \text{rul\_high} - \text{rul\_low}
$$
```text
width = rul_high - rul_low
```

#### Confidence proxy (from repository)
$$
\text{confidence}=\exp\left(-\frac{\text{width}}{k}\right)
$$
```text
confidence = exp( - width / k )
```
where `k` is median interval width on evaluation scope.

#### Cosine learning-rate scheduler
The training schedule used in later iterations applies cosine annealing from an initial learning rate to a minimum learning rate:
$$
\eta_t=\eta_{\min}+\frac{1}{2}\left(\eta_{\max}-\eta_{\min}\right)\left(1+\cos\left(\pi\frac{t}{T}\right)\right)
$$
```text
lr(t) = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t / T))
```
where `t` is the current epoch/step and `T` is the total schedule length.

### Metrics in use (actual)
- RMSE:

$$
\sqrt{\frac{1}{n}\sum_i(\hat{y}_i-y_i)^2}
$$
```text
RMSE = sqrt( (1/n) * sum_i (y_hat_i - y_i)^2 )
```

- $R^2$: explained variance ratio
- NASA score: asymmetric error penalty (late predictions penalized differently)

$$
\mathrm{NASA\ score}=\sum_{i=1}^{n}
\begin{cases}
\exp\!\left(-\frac{\hat{y}_i-y_i}{13}\right)-1, & \hat{y}_i < y_i \\
\exp\!\left(\frac{\hat{y}_i-y_i}{10}\right)-1, & \hat{y}_i \ge y_i
\end{cases}
$$
```text
d = y_hat - y_true
if d < 0: exp(-d/13) - 1
else:     exp(d/10) - 1
NASA_score = sum(over all samples)
```
- Weighted MAE (critical-phase emphasis):

$$
\frac{\sum_i w_i\lvert \hat{y}_i-y_i \rvert}{\sum_i w_i},
\quad
w_i=
\begin{cases}
3, & y_i \le 20 \\
1, & y_i > 20
\end{cases}
$$
```text
weights = 3.0 if y_true <= 20 else 1.0
weighted_MAE = sum(weights * abs(y_hat - y_true)) / sum(weights)
```
- Coverage:

$$
\frac{1}{n}\sum_i \mathbf{1}(y_i \in [\text{low}_i,\text{high}_i])
$$
```text
coverage = (1/n) * sum_i I( y_i in [low_i, high_i] )
```

- Mean interval width:

$$
\frac{1}{n}\sum_i(\text{high}_i-\text{low}_i)
$$
```text
mean_interval_width = (1/n) * sum_i (high_i - low_i)
```

### Where uncertainty comes from, how it is turned into numbers, and simple examples (defense script)

This subsection ties together **sources of uncertainty**, the **quantile / interval outputs**, the **NASA** point-metric, and the **dashboard confidence** proxy so you can explain them in one coherent story (with or without a whiteboard).

#### 1) “Low / mid / high” in *this* project = three RUL numbers, not three “uncertainty levels”

A single RUL (e.g. “30 cycles”) does not say how **tight** or **loose** the model is about that number. The notebook therefore uses **one model with three outputs** on each time window:

| Output in code | Quantile | Plain-language role |
|----------------|----------|----------------------|
| `rul_low`  | 0.1 (10th percentile) | **Pessimistic** RUL: “if things are on the bad end of what this history suggests” |
| `rul_mid`  | 0.5 (median) | **Central** RUL: the usual “best single number” for planning |
| `rul_high` | 0.9 (90th percentile) | **Optimistic** RUL: “if things are on the good end of what this history suggests” |

**Uncertainty** is not a separate label; it shows up in **how far apart** the pessimistic and optimistic RUL are. **Narrow** gap (e.g. `rul_low` ≈ 25, `rul_high` ≈ 32) → the model is painting a **tight** range (you can act on the median with more comfort). **Wide** gap (e.g. 10 to 50) → the same history is consistent with **many** remaining-life outcomes → **wider** uncertainty.

#### 2) As implemented in `notebooks/fd001.ipynb` (one forward pass + one loss)

**Model (`build_lstm`).** For each window, input shape is `(seq_len, n_features)` (e.g. 30 time steps × 66 features). The network is:

- `Input` → `LSTM(32, return_sequences=True)` → `LSTM(16)` → `Dense(16, relu)` → `Dropout(0.2)` → **`Dense(3)`** (linear, no activation).

So **one** pass through the LSTM stack produces **three** numbers in a row: index `0` = low quantile, `1` = mid, `2` = high (this ordering matches `QUANTILES = (0.1, 0.5, 0.9)` and code like `rul_low, rul_mid, rul_high = raw[:, 0], raw[:, 1], raw[:, 2]`).

**Loss (`make_quantile_loss(QUANTILES)`).** The true RUL for that window is a **single** scalar `y`. The model predicts a vector of length 3, `y_pred`. The code does:

- `e = y - y_pred` (broadcast so the **same** true RUL is compared to each of the three outputs),
- **Pinball (quantile) loss** per head: for quantile `q`, contribution is `max(q * e, (q - 1) * e)` (element-wise for each of the three `e` values),
- The training step **averages** that loss over the three heads and over the batch.

Over many windows and engines, that forces: roughly **10%** of true RULs to fall *below* `rul_low`, **50%** below `rul_mid`, **90%** below `rul_high` (see **coverage** in `metrics.json` for how well the test set matches that story).

**After training — concrete toy example (one window).** Imagine the **true** RUL for that row is **30** cycles, and the model outputs **`[20, 30, 45]`** for `[rul_low, rul_mid, rul_high]`.

- You still report **30** as the “main” number via `rul_mid` (if the model is well trained, median tracks the label on average; this line is illustration).
- The **band** is **[20, 45]**: “remaining life is plausibly between 20 and 45 cycles, with 30 in the middle.”
- **Width** = `45 - 20 = 25` cycles. Compare to another engine on the same day that outputs `[28, 30, 32]` → width **4** → **much** tighter implied uncertainty, even if both have median 30.

**Dashboard confidence (post-hoc, same notebook).** `width = rul_high - rul_low`, then `k` = **median** of `width` over the last window per test engine, and `confidence = exp(-width / k)` so engines with **narrower** bands get **higher** numbers (ranking only; not a formal “95% probability”). Values are written to `predictions.json` / `confidence_meta.json`.

#### 3) NASA score: what it measures and a tiny hand calculation

**NASA score** uses the **median prediction** `rul_mid` only (not the interval). It is a **sum over test engines** of **asymmetric** penalties so that “wrong in the riskier direction” costs more. Define **signed error** as `d = ŷ - y` (predicted minus true RUL).

$$
\text{per-engine term} =
\begin{cases}
\exp\!\left(-\dfrac{d}{13}\right) - 1, & d < 0 \quad (\text{conservative: predicted RUL below true RUL})\\[6pt]
\exp\!\left(\dfrac{d}{10}\right) - 1, & d \ge 0 \quad (\text{optimistic: predicted RUL at or above true RUL})
\end{cases}
$$

**Intuition (one sentence):** If you **overestimate** RUL (think the engine is healthier than it is, `d > 0`), the score uses the `/10` branch and grows quickly—aligned with “late maintenance is dangerous.” If you **underestimate** RUL (`d < 0`, conservative early warning), the `/13` branch applies a different (often relatively smaller) cost. **Lower total NASA score is better.**

**Mini example (two engines only):** True RUL `y = 50` for both.

- **Engine 1 — conservative:** `ŷ = 40` → `d = -10` →  
  `exp(-(-10)/13) - 1 = exp(10/13) - 1 ≈ 2.16 - 1 = 1.16`
- **Engine 2 — optimistic:** `ŷ = 60` → `d = +10` →  
  `exp(10/10) - 1 = e - 1 ≈ 1.72`

So the **same absolute error (10 cycles)** in the “optimistic” direction gets a **larger** penalty here than the conservative one—on purpose. The **headline** `NASA_score` in `metrics.json` sums similar terms over all evaluation engines (same protocol as in your notebook / export).

#### 4) One-sentence links for the panel

- **Uncertainty in outputs:** from partial observability + data noise + what the quantile LSTM can learn, summarized as **interval width** between 10th and 90th predictions.
- **How calculated:** `width = rul_high - rul_low`; `confidence = exp(-width/k)` with `k` = median test width; **quality checks:** coverage + mean width in `metrics.json`.
- **NASA score:** asymmetric sum of `exp` penalties on **median** error only—**lower is better**; it complements RMSE, which is symmetric.

### Explainability
SHAP values computed for median output head:
- global: mean absolute SHAP ranking
- local: per-engine feature contributions

### Why this model choice is defendable
- Sequence-aware (LSTM) suitable for temporal degradation
- Quantile outputs directly align with maintenance decision uncertainty
- SHAP provides stakeholder-readable evidence of feature reliance

### Alternatives to discuss in Q&A
- Point-only regressors (Ridge, RF, GBR)
- 1D CNN (present in notebook experiments)
- Attention/Transformer variants (future scope)
- Conformal intervals (future calibration improvement)

---

## 7) 40-Minute Presentation Defense Plan

### Suggested time allocation
1. Problem + motivation: 5 min  
2. Data + EDA: 6 min  
3. Method + architecture: 10 min  
4. Results + interpretability: 10 min  
5. Dashboard/deployment + limitations/future: 6 min  
6. Buffer transitions: 3 min

### Slide-by-slide speaking flow (example 14 slides)

1. **Title + objective**  
   “I focus on risk-aware RUL, not just point prediction.”

2. **Why this problem matters**  
   “Late/early maintenance tradeoff in safety-critical operations.”

3. **Dataset overview**  
   FD001 structure, engines, sensors, sequence nature.

4. **EDA key findings**  
   Why temporal dynamics and engineered features matter.

5. **Pipeline architecture**  
   Input -> preprocessing -> features -> LSTM quantiles -> SHAP/export.

6. **Model details**  
   Quantile heads and pinball loss.

7. **Evaluation protocol**  
   Last-window-per-engine rationale.

8. **Primary metrics**  
   RMSE/R2/NASA + interval metrics.

9. **Uncertainty interpretation**  
   Coverage, width, confidence formula.

10. **SHAP global**  
    Trend/rate-of-change dominance.

11. **Engine-level outputs**  
    Example how low+wide drives priority.

12. **Export package + dashboard integration**  
    What files power what visuals.

13. **Limitations**  
    FD001 scope, calibration caveats, no production API in repo.

14. **Conclusion + next steps**  
    Impact and roadmap.

---

## 8) Panel Question Preparation

### A) Hard technical questions + answer patterns

**Q1: Why LSTM instead of Transformers?**  
- **Short:** Dataset size/structure + sequence length made LSTM an effective baseline with lower complexity.  
- **Deep:** LSTM captures temporal dependencies with fewer parameters; easier training stability in benchmark-sized setting. Transformer upgrade is future work.

**Q2: You don’t have time series in your model—so how are you using LSTM?**  
- **Short:** We *do* feed a time series: each training sample is a **sequence** of **consecutive operating cycles** per engine, not a single flat row.  
- **Deep:** After windowing in `fd001.ipynb`, the tensor going into the LSTM has shape **`(batch, seq_len, n_features)`**—for example **`seq_len = 30`** timesteps × **`n_features`** sensors/engineered columns at each cycle. The LSTM steps along **time within that window** (cycle \(t-29\) … \(t\)) for one engine. The **label** is the RUL at the **end** of that window. What you may see as “tabular” in a CSV is the **underlying** log; the model **never** sees one random row in isolation without the sliding window that **reconstructs** the temporal ordering. Baselines (Ridge, RF) flatten the same window to a vector; the LSTM keeps the **ordered** structure so it can use **temporal dynamics** across the lookback.

**Q3: How do you justify uncertainty quality?**  
- **Short:** Coverage and interval width together.  
- **Deep:** Coverage ~0.83 indicates many truths fall in interval; width controls sharpness; confidence derived from width operationalizes uncertainty ranking.

**Q4: Mid-cycle, or when data are noisy or a sensor fails—how does your uncertainty band react?**  
- **Short:** The band is **data-driven** from the quantile LSTM: it tends to **widen** when the pattern in the last window is **hard to place** (noise, rare regime, conflicting dynamics), and can look **tighter** when the history is smooth and “on distribution”—but the pipeline does **not** run a separate **sensor-fault detector**; a stuck sensor is **not** explicitly labeled.  
- **Deep:** (1) **Any cycle:** You can score **every** sliding window (not only at failure), so “mid-life” and late-life both get a `[rul_low, rul_high]`. (2) **Noise / quality:** If spikes distort the 30-step trajectory, the three heads often **spread**—larger `width`—because the model is **less** confident about a single RUL. That is the intended *warning-like* behavior, but it is **not** guaranteed: pathological patterns could still give **wrong** intervals; this is why we report **coverage** on hold-out data, not a proof under arbitrary faults. (3) **Sensor failure (missing/stuck):** C-MAPSS in this project **does not** ship true failure-injection labels for training; a missing sensor in deployment would need **governance** (anomaly rules, imputation, or masking) *before* the model, or a dedicated diagnostic model—stated as **limitation + future work**. (4) **What to say operationally:** Treat **wide** band + **odd** SHAP or residuals as a **triage** signal, not a substitute for **hardware** health checks.

**Q5: Is confidence a probability?**  
- **Short:** No, it is a width-derived proxy.  
- **Deep:** It is monotonic in width via `exp(-width/k)`, useful for ranking trust, not calibrated Bayesian probability.

**Q6: Why only the last window per engine when reporting metrics?**  
- **Short:** One **decision-time** estimate per test engine, aligned with C-MAPSS test labels and standard PHM reporting—**not** pooling every cycle, which would **reuse** the same engine many times.  
- **Deep:** (1) **Label match:** `RUL_FD001.txt` gives a single true RUL per test engine, defined at the **last** time step of that engine’s test run, so the prediction that is directly comparable is the one from the **last** `seq_len`-length window. (2) **Operational match:** A fleet dashboard ranks **current** risk; the last available window is the “**now**” view for that engine. (3) **Statistical match:** If you include **all** sliding windows, scores mix **dozens of almost-redundant** rows per engine and **overcount** long trajectories, which **inflates** sample size and can **misstate** how well you predict at a single read-out per asset. (4) You *can* analyze errors along the full test trajectory for diagnostics, but the **headline** RMSE / NASA / coverage in this project follow **one point per test engine** for a fair, comparable, decision-focused summary.

**Q7: What evidence that model learned meaningful behavior?**  
- SHAP top features are trend/rate-of-change (degradation dynamics), not random noise-only signals.

**Q8: Where is frontend/backend code?**  
- This repo contains model pipeline + static payloads. UI app lives separately; this repo provides data contracts/exports.

**Q9: Biggest deployment risk?**  
- Distribution shift beyond FD001; requires recalibration and monitoring.

**Q10: Pre-model sensitivity (which features to remove) vs post-model SHAP—are they the same?**  
- **Short:** It is **not** a contradiction—they answer **different** questions, and the names should **not** be interchanged. **Sensitivity analysis (pre-model):** use **data** (and domain knowledge) in EDA/preprocessing to decide what **enters** the model—e.g. remove constant / uninformative sensors, engineer dynamics features, RUL cap, **train-fitted** scaling. **SHAP (post-model):** **only exists once** a trained model exists; it **explains** what that **particular** LSTM uses—it is **not** a substitute for the earlier data-based screening step. In defense language: *sensitivity* ≈ *what the **dataset** justifies **keeping or dropping***; *SHAP* ≈ *what the **learned** mapping relies on* **given** those choices.  
- **Deep — terminology the panel may expect (align with your professor):**  
  - **Pre-model sensitivity / feature decisions:** **Data-centric**; goals include *remove noise, reduce dimensionality, respect physics* before any deep net. **Tools:** variance, missingness, trends, correlation with **train** RUL, plots by cycle/engine. **Advantages:** model-agnostic, fast, no architecture commitment. **Limits:** may miss value that only appears in **sequential, nonlinear** form—then the *model* still has a job, but the **input contract** is already cleaner.  
  - **Post-model SHAP:** **Model-centric**; explains **attributions** on a **fixed** feature set and **trained** weights. **Advantages:** faithfulness to this LSTM, interpretability for **stakeholders** and “did it learn something sensible?” **Limits:** does **not** by itself define which columns should have been **dropped in EDA**; using SHAP on the **test** set to *iteratively delete* features is bad practice (overfits the benchmark).  
- **This project in one line:** **EDA/preprocessing = sensitivity / selection on data**; **SHAP = explanation after training**; both are used, in that **order**—SHAP is **not** the same as pre-model sensitivity analysis, even if both feel like “importance.”

**Q11: Why does the model tend not to perform as well on “healthier” engines (high remaining useful life)?**  
- **Short:** Far from failure, **degradation is weak in the sensors**, so early-life RUL is **harder** to pin down than near-failure RUL; the dataset also uses a **capped** RUL target, which **compresses** many early segments into similar labels. This is a **known** PHM pattern, not a bug unique to your notebook.  
- **Deep:** (1) **Signal:** Informative damage progression often **emerges late**; at **high RUL**, readings sit near **nominal** operation and **noise + operating variability** dominate—small sensor differences must support **large** differences in true remaining life, which is **ill-posed**. (2) **Labels:** With an **RUL cap** (e.g. 125 cycles in setup), many windows early in a run share **plateaued** targets, so the model gets **less sharp** supervision for distinguishing “how healthy” vs “how very healthy.” (3) **Metrics:** Large absolute errors on **high-RUL** predictions can **inflate** RMSE even when **low-RUL** (critical) errors are small—your **weighted MAE** up-weights **low RUL** partly for this reason. (4) **Defense framing:** Say you **prioritize** end-of-life accuracy **operationally**; acknowledge **stratified** analysis (by RUL bin) as optional reporting for transparency.

**Q12: What is a “digital twin” here, and how does it work in your project?**  
- **Short:** A **virtual, data-driven mirror** of each engine’s health trajectory: time-series in → RUL and uncertainty out → **exports/dashboard payloads** for fleet-level decisions, plus **SHAP** for explainability. It is a **prognostic** twin, not a 3D CAD twin, and the repo is a **prototype** (batch artifacts + static dashboard data), not a full live control loop.  
- **Deep:** In general, a digital twin links **sensing**, **modeling**, and **presentation**. This work covers **(1) sensing surrogate:** C-MAPSS **windowed features**; **(2) modeling:** quantile **LSTM**; **(3) presentation:** metrics, per-engine tables, and dashboard export packages. A **production** twin would add **streaming** data, **online** API inference, and **continuous** model governance—acknowledged in **Section 1** and **Section 5** of this document.

### B) Skeptical researcher questions
- “How comparable is your literature table?”  
  -> Acknowledge protocol mismatch risk; compare only with matched assumptions.
- “Why not conformal prediction?”  
  -> Good future step; current method uses quantile regression for direct interval learning.
- “Did you evaluate low/high head SHAP separately?”  
  -> Current SHAP target is median head; tail-head explainability is identified future extension.

### C) Non-technical evaluator answers
- “What does this project give a company?”  
  -> A ranked list of engines by urgency and uncertainty, improving maintenance prioritization.

---

## 9) Implementation Challenges & Resolutions

- **Noisy signals** -> smoothing + engineered dynamics features
- **Point prediction limitations** -> quantile heads for intervals
- **Communicating uncertainty** -> width + confidence + coverage
- **Explainability need** -> SHAP global/local outputs
- **Dashboard data integration** -> export script with stable schemas
- **Parquet dependency friction** -> switched export to CSV for portability

Tradeoff examples:
- Better uncertainty can increase width (safety vs sharpness balance)
- Notebook agility vs production modularity

---

## 10) Validation & Testing

### ML validation
- Engine-level split strategy (per docs/method)
- Last-window-per-test-engine evaluation for headline metrics
- Residual/error analysis through exported fields (`error_signed`, `error_abs`)

### Artifact validation
- Schema-consistent files generated in `dashboard_export`
- Metadata provenance in `meta.json`
- Confidence formula tracked in `confidence_meta.json`

### System-level verification in this repo
- Re-runnable export script (`export_dashboard_pack.py`)
- Consistency between metrics/predictions/shap outputs and dashboard-ready files

### What proves correctness
- Stable metric artifact values
- Coherent interval behavior (coverage vs width)
- SHAP feature rankings matching engineered degradation intuition

---

## 11) Metrics & Results Analysis

### Actual final metrics in repository
From `artifacts/data/metrics.json`:
- RMSE: **12.510066**
- R2: **0.902544**
- NASA_score: **274.41338**
- weighted_MAE: **7.217562**
- coverage: **0.83**
- mean_interval_width: **32.859956**
- within_10_pct: **68.0**
- within_20_pct: **88.0**

### Why these metrics are appropriate
- RMSE/R2: overall regression fidelity
- NASA score: asymmetric penalty relevance to maintenance risk
- weighted MAE: emphasizes low-RUL (critical-phase) errors via higher weights
- Coverage + width: uncertainty calibration/sharpness proxy
- within_10/20: intuitive tolerance reporting for stakeholders

### Requested-but-not-used metrics (important to state clearly)
Metrics like MCC, ROC-AUC, PR-AUC, classification recall/precision/F1 are **not primary** here because this is a **regression/interval prognostics** problem, not a binary classifier in current implementation.

---

## 12) Limitations & Future Scope

### Current limitations
- Dataset scope mostly FD001-focused in this repo workflow
- No in-repo production backend/frontend service implementation
- Confidence is width-based proxy, not formal calibrated probability
- SHAP primarily for median head; tail-head explanations not fully explored
- **High-RUL (healthier) regimes** are intrinsically harder: weak degradation signal before failure is visible, RUL **capping** flattens early labels—errors can be **larger** there while **low-RUL** performance matters more for maintenance (see **Q11** in Section 8)

### Future work
- Extend benchmarking to FD002/FD003/FD004 with consistent protocol
- Conformal or post-hoc interval calibration
- Multi-head explainability (low/high quantile SHAP)
- MLOps architecture:
  - API-serving layer
  - scheduled inference and retraining
  - drift and calibration monitoring
- Enterprise scaling:
  - role-based dashboard
  - alerts and maintenance ticket integration

---

## 13) Conclusion

This project successfully delivers an uncertainty-aware RUL pipeline that is technically sound, artifact-driven, and operationally meaningful. It goes beyond point accuracy by packaging uncertainty and interpretability for decision support.

### Research-oriented close
“On FD001, the quantile LSTM demonstrates that temporal deep models can be made decision-facing through interval prediction and SHAP, with reproducible artifact outputs.”

### Industry-oriented close
“This system helps maintenance teams prioritize engines not only by predicted life left, but by how certain that estimate is.”

### Presentation ending script (30 sec)
“My final contribution is not just a model score. It is a complete decision pipeline: prediction, uncertainty, explanation, and deployment-ready exports. That combination is what makes this work practical and defensible.”

---

## 14) Defense Cheat Sheet (Rapid Revision)

### Core numbers (memorize)
- RMSE 12.51
- NASA 274.41
- R2 0.903
- Coverage 0.83
- Mean width 32.86

### Core formulas
- width = high - low
- confidence = exp(-width / k) (k = median test width; see `confidence_meta.json`)
- pinball loss (quantile)
- NASA: per engine, `d = rul_mid - y_true`; if `d < 0` use `exp(-d/13)-1`, else `exp(d/10)-1`; **sum** over engines, **lower is better**
- Uncertainty story: three quantile heads → width summarizes spread; NASA uses **median only**

### Architecture one-liner
“Engine time-series -> preprocessing + engineered dynamics -> LSTM quantiles -> interval + confidence + SHAP -> dashboard exports.”

### Digital twin one-liner
“**Behavioral** twin: sensor history → LSTM RUL/uncertainty → SHAP + dashboard-style exports; **prototype** decision layer, not a live IoT/CAD digital twin.”

### Critical implementation points
- `seq_len = 30`, `n_features = 66`, quantiles `[0.1,0.5,0.9]`
- Evaluation scope: last window per test engine
- Export script outputs all dashboard files

### Panel traps
- “No time series, so why LSTM?” -> Input is `(batch, seq_len, n_features)` sliding windows; 30 consecutive cycles per engine, not a single row
- “Noisy / failed sensor—what happens?” -> Band often widens when the trajectory is hard to read; not a dedicated fault model—pair with data QA / monitoring in production
- “Sensitivity vs SHAP?” -> **Sensitivity/EDA = before modeling**, data-driven which features to **remove**; **SHAP = after modeling**, model-based **explanation**—not the same method
- “Worse on healthy / high RUL?” -> **Weak** fault **signature** far from failure + **capped** RUL targets; critical-phase accuracy often **matters more** operationally
- “Confidence is probability?” -> No, ranking proxy
- “Where is app code?” -> Separate; this repo provides static payloads and ML pipeline
- “Can literature table be directly compared?” -> Only under matched protocol

### 15-second fallback answer
“This project predicts RUL with uncertainty, validates interval usefulness, explains feature influence with SHAP, and exports dashboard-ready data for operational prioritization.”

---

## 15) Important Execution Rules Compliance Notes

- This package is grounded in files and artifacts present in the repository.
- Missing components are explicitly identified (frontend/backend app source not in this repo snapshot).
- Claims are separated into “implemented now” vs “future/optional.”

---

## Appendix-Ready Add-ons

Potential next deliverables:
1. Slide deck script (exact 40-min speaking text)
2. Extended viva Q&A bank (100+ categorized questions with short/deep answers)
3. One-page examiner handout (architecture + metrics + contributions)
4. Mock defense rehearsal (interactive Q&A)