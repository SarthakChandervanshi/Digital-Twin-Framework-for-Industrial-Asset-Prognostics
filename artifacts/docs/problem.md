# Problem: remaining useful life (RUL) under uncertainty

## What RUL is

**Remaining useful life** is the number of operating cycles (or time) until an asset can no longer perform its function within specification—here, until engine failure in the NASA C-MAPSS turbofan benchmark. The model is trained on run-to-failure trajectories and asked to predict how many cycles remain at each point in life.

## Why it matters

RUL supports **condition-based maintenance**: scheduling inspections, parts, and downtime before failure while avoiding unnecessary stops. Errors are asymmetric in practice: predicting **too much** RUL (late) risks unexpected failure; predicting **too little** (early) wastes capacity. Prognostics therefore needs both **point estimates** and a sense of **how wrong** predictions might be.

## Why intervals and uncertainty

A single number (e.g. median RUL) hides spread: sensors are noisy, degradation paths differ across units, and the future is not uniquely determined by past windows. **Quantile-based prediction intervals** summarize plausible lower and upper bounds on RUL. Together with a clear **confidence** notion derived from interval width, decision-makers can see not only *when* the model thinks failure is near, but *how tight* that belief is. This project reports low / median / high quantiles, interval width, coverage (whether true RUL falls inside the band), and a confidence score—**numerical KPIs are stored in `artifacts/data/metrics.json`**, not duplicated here.
