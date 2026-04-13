# Literature comparison (FD001)

C-MAPSS results in publications **are not directly comparable** unless authors use the **same RUL capping**, **train/test engine split**, and **evaluation rule** (e.g. last cycle only vs all cycles). Treat the table below as **illustrative**: fill the **This work** row from **`artifacts/data/metrics.json`** after each training run; update published rows when you cite specific papers with exact settings.

| Reference | Model / method | Reported RMSE | Reported NASA score | Dataset / notes |
|-----------|------------------|---------------|---------------------|-----------------|
| A. Saxena *et al.*, “Damage propagation modeling for aircraft engine run-to-failure simulation”, IEEE PHM 2008 | Benchmark description | — | — | C-MAPSS introduction; defines FD001 |
| Example: deep regression / LSTM papers on C-MAPSS FD001 | CNN, LSTM, GRU, etc. | *as in paper* | *as in paper* | FD001; protocol must match (RUL cap, split, last-cycle rule) |
| **This work (fill after each run)** | LSTM + quantile heads (median for point metrics) | *from `metrics.json`* | *from `metrics.json`* | FD001; scope field in `metrics.json` |

## How to use this table

1. Copy **RMSE** and **NASA_score** from `artifacts/data/metrics.json` into the last row.
2. When adding papers, record **their** capping (e.g. 125) and whether metrics are **last-cycle** or pooled.
3. Prefer comparing against **peer-reviewed** rows that explicitly use **FD001** and document preprocessing.
