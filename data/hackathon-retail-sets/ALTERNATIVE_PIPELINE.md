# Alternative direct-quantile pipeline

`alternative_direct_forecast.py` is a new, standalone preprocessing and
modelling route. It never reads the existing `processed/` files or changes the
notebooks/submission already in the project.

It builds clean weekly panels from the raw parquet files, makes direct forecasts
for all 13 horizons, and trains a conservative LightGBM quantile model. The
target and historical features are scaled by the expanding per-series MASE
denominator, so model fitting emphasises the same slow-moving articles as the
competition metric. A separately tuned guard then eliminates weak forecasts for
long-dormant or very intermittent articles. It also backtests several local
rolling-quantile forecasts and selects a winner on the older validation windows
instead of assuming the global learner wins.

The default validation is five sequential 13-week windows. It uses older
windows to choose the guard, with the newest two windows retained as a check.
Set A/C/D are loaded by default so their independent preprocessing is exercised,
but they only enter model training with `--with-aux`; compare the two variants
by their saved guarded backtests before trusting outside data.

```powershell
cd C:\Users\mehme\Desktop\Projects\District19\District19\data\hackathon-retail-sets
..\..\..\.venv\Scripts\python.exe alternative_direct_forecast.py --fast
..\..\..\.venv\Scripts\python.exe alternative_direct_forecast.py --run-final
..\..\..\.venv\Scripts\python.exe alternative_direct_forecast.py --with-aux --run-final
```

Each run writes only new results to `alternative_outputs/`:

- `backtest_raw.csv` and `backtest_guarded.csv`
- `guard_search.csv` (the full post-processing search)
- `run_report.json`
- `submission_alternative_<selected-model>.csv` when `--run-final` is used.
