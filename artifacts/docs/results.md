# Results (narrative)

This section is **interpretive only**. Quantitative scores are **not** duplicated here; they belong in **`artifacts/data/metrics.json`** and per-engine tables in **`artifacts/data/predictions.json`**.

## What we observe

The quantile LSTM is trained to minimize pinball loss across three heads simultaneously, so the **median** tracks the central tendency of RUL while the **low** and **high** heads adapt to dispersion. On the test fleet, point accuracy (e.g. RMSE on the median) and the **NASA** asymmetric score should be read together: the NASA metric penalizes late errors more than early ones, which often better matches operational risk than symmetric error alone.

**Coverage**—the fraction of engines whose true RUL lies between the predicted low and high quantiles—indicates whether the **stated intervals are calibrated** in a frequency sense. If coverage sits near the nominal level implied by the 10th and 90th percentiles, the bands are useful for decision-making; if not, the model may be over- or under-confident despite a good median.

**Confidence** derived from interval width is a **relative** readability aid on the dashboard: it ranks engines by how sharp the prediction band is, using a scale anchored at the typical band width (`k`). It is not a calibrated probability unless separately validated.

## Caveats

1. **Protocol sensitivity:** C-MAPSS scores depend on RUL capping, which engines are used for train/val/test, and whether metrics use **all windows** or **last window per engine**. Always compare numbers only when these choices match.

2. **Distribution shift:** The model is trained on historical run-to-failure data; performance on new fleets, sensors, or operating conditions may differ. Intervals narrow or widen with the data—**monitor coverage** on new deployments, not only RMSE.

3. **SHAP for the median head** explains contributions to the **central** prediction; it does not fully characterize uncertainty in the tails unless separate analyses are run for low/high heads.
