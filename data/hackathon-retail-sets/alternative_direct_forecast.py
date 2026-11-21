"""Independent, leakage-safe direct forecasting pipeline for the LemeBase challenge.

This intentionally reads only the raw parquet files.  It does not import or
overwrite `processed/`, `preprocessing.ipynb`, `modelling.ipynb`, or an
existing submission.

The design is deliberately conservative:
  * train one direct model for each of the 13 horizons, represented by a
    horizon feature, rather than recursively feeding forecasts back in;
  * normalise each series by its expanding MASE denominator, which makes the
    L1/quantile loss line up with the competition's per-article MASE;
  * tune the zero/recency guard separately from the model, as spurious small
    positive forecasts are especially expensive on intermittent items;
  * use all of the test-shop's history for the final fit.  Auxiliary sources
    can be opted into, but the default stays test-only unless their rolling
    validation score wins.

Run from this folder (or any parent):

    ..\\..\\..\\.venv\\Scripts\\python.exe alternative_direct_forecast.py --run-final

For a quicker smoke test:

    ..\\..\\..\\.venv\\Scripts\\python.exe alternative_direct_forecast.py --fast
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd


HORIZON = 13
SEED = 19
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "alternative_outputs"


@dataclass
class Panel:
    """A dense weekly collection of independent demand series."""

    name: str
    values: np.ndarray  # series x weeks; NaN means unobserved/censored
    weeks: pd.DatetimeIndex
    on_hand: np.ndarray | None = None

    @property
    def n_series(self) -> int:
        return self.values.shape[0]


def monday_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="W-MON")


def _dense_weekly(
    frame: pd.DataFrame,
    keys: list[str],
    value: str = "units",
    fill_inside: bool = True,
    trim_before_first: bool = False,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Pivot an already-weekly table to dense rows.

    `fill_inside=True` is only used for sources where absence means no sale.
    Set C is handled separately because a missing daily row is not a zero.
    """
    frame = frame.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    grid = monday_grid(frame.week_start.min(), frame.week_start.max())
    series_key = frame[keys].astype(str).agg("|".join, axis=1)
    wide = frame.assign(_series=series_key).pivot(index="_series", columns="week_start", values=value)
    wide = wide.reindex(columns=grid)
    if fill_inside:
        if trim_before_first:
            listed = wide.notna().cumsum(axis=1).gt(0)
            wide = wide.where(~listed, wide.fillna(0.0))
        else:
            wide = wide.fillna(0.0)
    return wide.to_numpy(dtype=np.float64), grid


def load_test_shop() -> Panel:
    raw = pd.read_parquet(ROOT / "test-shop" / "history.parquet")
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    weeks = pd.DatetimeIndex(sorted(raw.week_start.unique()))
    units = raw.pivot(index="article_id", columns="week_start", values="units").reindex(columns=weeks)
    stock = raw.pivot(index="article_id", columns="week_start", values="on_hand").reindex(columns=weeks)
    assert units.notna().all().all(), "test shop should be a complete weekly grid"
    return Panel("test", units.to_numpy(float), weeks, stock.to_numpy(float))


def load_set_a() -> Panel:
    raw = pd.read_parquet(ROOT / "set-a" / "weekly.parquet")
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    # Sales from stock-out weeks are censored, not real zero demand.
    raw.loc[~raw["in_stock"].fillna(False), "units"] = np.nan
    values, weeks = _dense_weekly(raw, ["store_id", "article_id"], fill_inside=False)
    return Panel("a", values, weeks)


def load_set_c() -> Panel:
    raw = pd.read_parquet(ROOT / "set-c" / "daily.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["week_start"] = raw["date"] - pd.to_timedelta(raw.date.dt.day_of_week, unit="D")
    raw["units"] = pd.to_numeric(raw["units"], errors="coerce")
    # Keep only complete listed weeks.  Missing days in this source are not
    # observed zero-demand days, so treating a partial week as zero is unsafe.
    gcols = ["store_id", "article_id", "week_start"]
    weekly = raw.groupby(gcols, as_index=False).agg(units=("units", "sum"), n_days=("date", "nunique"))
    weekly.loc[weekly.n_days < 7, "units"] = np.nan
    values, weeks = _dense_weekly(weekly, ["store_id", "article_id"], fill_inside=False)
    return Panel("c", values, weeks)


def load_set_d() -> Panel:
    raw = pd.read_parquet(ROOT / "set-d" / "sales.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["week_start"] = raw["date"] - pd.to_timedelta(raw.date.dt.day_of_week, unit="D")
    raw["units"] = pd.to_numeric(raw["units"], errors="coerce")
    # The source is sales lines, so the calendar grid legitimately supplies
    # zero weeks.  Aggregate stores to article level, the closest match to the
    # held-out shop's article-level target.
    weekly = raw.groupby(["article_id", "week_start"], as_index=False)["units"].sum()
    values, weeks = _dense_weekly(weekly, ["article_id"], fill_inside=True, trim_before_first=True)
    return Panel("d", values, weeks)


def load_sources(names: Iterable[str]) -> list[Panel]:
    loaders = {"test": load_test_shop, "a": load_set_a, "c": load_set_c, "d": load_set_d}
    result = [loaders[name]() for name in names]
    for p in result:
        print(f"loaded {p.name}: {p.n_series:,} series x {p.values.shape[1]:,} weeks")
    return result


def _safe_stat(x: np.ndarray, fun, default: float = 0.0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            out = fun(x, axis=1)
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default)


def _last_sale_age(history: np.ndarray) -> np.ndarray:
    """Number of complete weeks since a positive sale; inf if it never sold."""
    sold = np.isfinite(history) & (history > 0)
    reverse = sold[:, ::-1]
    age = reverse.argmax(axis=1).astype(float)
    age[~reverse.any(axis=1)] = np.inf
    return age


def _feature_block(panel: Panel, origin: int, max_horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features and scaled direct targets at a single forecast origin.

    The final column that varies by horizon is added after base features are
    repeated.  Nothing after `origin` is consulted for a feature or scale.
    """
    x = panel.values
    hist = x[:, : origin + 1]
    n = len(x)
    observed = np.isfinite(hist)
    h_clean = np.where(observed, hist, np.nan)
    diffs = np.abs(np.diff(h_clean, axis=1))
    scale = _safe_stat(diffs, np.nanmean, default=1.0)
    scale = np.maximum(scale, 0.10)  # avoids exploding targets for near-constant items
    age = _last_sale_age(h_clean)

    def lag(k: int) -> np.ndarray:
        return h_clean[:, -(k)] if hist.shape[1] >= k else np.full(n, np.nan)

    def recent(k: int) -> np.ndarray:
        return h_clean[:, -min(k, hist.shape[1]) :]

    r4, r8, r13, r26, r52 = (recent(k) for k in (4, 8, 13, 26, 52))
    base = np.column_stack(
        [
            *[lag(k) / scale for k in (1, 2, 3, 4, 6, 8, 13, 26, 52)],
            _safe_stat(r4, np.nanmean) / scale,
            _safe_stat(r8, np.nanmean) / scale,
            _safe_stat(r13, np.nanmean) / scale,
            _safe_stat(r26, np.nanmean) / scale,
            _safe_stat(r52, np.nanmean) / scale,
            _safe_stat(r4, np.nanmedian) / scale,
            _safe_stat(r13, np.nanmedian) / scale,
            _safe_stat(r26, np.nanmedian) / scale,
            _safe_stat(r52, np.nanmedian) / scale,
            _safe_stat((r13 > 0) & np.isfinite(r13), np.mean),
            _safe_stat((r26 > 0) & np.isfinite(r26), np.mean),
            _safe_stat((r52 > 0) & np.isfinite(r52), np.mean),
            np.minimum(age, 53),
            np.log1p(np.maximum(scale, 0)),
            np.minimum(observed.sum(axis=1), 160),
            np.full(n, 1.0 if panel.name == "test" else 0.0),
            np.full(n, {"test": 0, "a": 1, "c": 2, "d": 3}[panel.name]),
        ]
    )
    # NaNs are retained: LightGBM treats them as a separate, informative state.
    rows = np.repeat(base, max_horizon, axis=0)
    horizons = np.tile(np.arange(1, max_horizon + 1), n)
    rows = np.column_stack([rows, horizons])

    seasonal = np.full((n, max_horizon), np.nan)
    for h in range(1, max_horizon + 1):
        idx = origin + h - 52
        if idx >= 0:
            seasonal[:, h - 1] = x[:, idx] / scale
    rows = np.column_stack([rows, seasonal.reshape(-1)])

    target = np.full((n, max_horizon), np.nan)
    available = min(max_horizon, x.shape[1] - origin - 1)
    if available:
        target[:, :available] = x[:, origin + 1 : origin + 1 + available] / scale[:, None]
    return rows.astype(np.float32), target.reshape(-1).astype(np.float32), scale


def training_rows(
    panels: Iterable[Panel],
    test_cutoff: int,
    include_auxiliary: bool,
    stride: int,
    min_history: int = 26,
) -> tuple[np.ndarray, np.ndarray]:
    """Build leakage-safe direct examples.

    Test-shop labels stop at `test_cutoff`.  Auxiliary chains are independent
    data, so their complete published histories can be used as external
    training examples.  Their stride is doubled to prevent them from swamping
    the actual target shop.
    """
    features, labels = [], []
    for panel in panels:
        if panel.name != "test" and not include_auxiliary:
            continue
        last_origin = (test_cutoff - HORIZON) if panel.name == "test" else (panel.values.shape[1] - HORIZON - 1)
        source_stride = stride if panel.name == "test" else stride * 2
        for origin in range(min_history - 1, last_origin + 1, source_stride):
            X, y, _ = _feature_block(panel, origin)
            ok = np.isfinite(y)
            if ok.any():
                features.append(X[ok])
                labels.append(y[ok])
    if not features:
        raise ValueError("No train rows: increase cutoff or reduce min_history")
    return np.concatenate(features), np.concatenate(labels)


def fit_model(X: np.ndarray, y: np.ndarray, alpha: float, fast: bool) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=140 if fast else 360,
        learning_rate=0.06 if fast else 0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=750,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def forecast_at(model: lgb.LGBMRegressor, test: Panel, cutoff: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    X, _, scale = _feature_block(test, cutoff)
    raw = model.predict(X).reshape(test.n_series, HORIZON) * scale[:, None]
    return np.clip(raw, 0, None), forecast_info(test, cutoff)


def forecast_info(test: Panel, cutoff: int) -> dict[str, np.ndarray]:
    hist = test.values[:, : cutoff + 1]
    return {
        "age": _last_sale_age(hist),
        "sale_rate_13": np.mean(hist[:, -13:] > 0, axis=1),
        "sale_rate_26": np.mean(hist[:, -26:] > 0, axis=1),
    }


def baseline_forecasts(test: Panel, cutoff: int) -> dict[str, np.ndarray]:
    """Robust competitors for the global learner, all based on past demand only."""
    hist = test.values[:, : cutoff + 1]

    def flat(q: float, window: int) -> np.ndarray:
        point = np.quantile(hist[:, -window:], q, axis=1)
        return np.repeat(point[:, None], HORIZON, axis=1)

    q40_20 = flat(0.40, 20)
    # Dates were shifted by whole weeks, but a small annual component can still
    # help repeatable patterns without copying last year's noise wholesale.
    seasonal = np.empty_like(q40_20)
    for h in range(1, HORIZON + 1):
        idx = cutoff + h - 52
        seasonal[:, h - 1] = hist[:, idx] if idx >= 0 else q40_20[:, h - 1]
    return {
        "q35_w13": flat(0.35, 13),
        "q40_w20": q40_20,
        "q40_w26": flat(0.40, 26),
        "q45_w20": flat(0.45, 20),
        "median_w20": flat(0.50, 20),
        "median_w26": flat(0.50, 26),
        "q40_w20_season25": 0.75 * q40_20 + 0.25 * seasonal,
    }


def apply_guard(raw: np.ndarray, info: dict[str, np.ndarray], dead_weeks: int, floor: float, min_rate: float) -> np.ndarray:
    """Turn weak intermittent forecasts into zero without touching strong items."""
    keep = (info["age"] < dead_weeks) & (info["sale_rate_26"] >= min_rate)
    out = np.where(keep[:, None], raw, 0.0)
    return np.where(out >= floor, out, 0.0)


def score(y: np.ndarray, cutoff: int, forecast: np.ndarray) -> dict[str, float]:
    hist = y[:, : cutoff + 1]
    actual = y[:, cutoff + 1 : cutoff + 1 + HORIZON]
    denom = np.abs(np.diff(hist, axis=1)).mean(axis=1)
    # Constant-zero histories have an undefined MASE denominator and are not
    # part of the competition mean.  Mask them before averaging (rather than
    # allowing a correct 0/0 to become NaN but an incorrect forecast to become
    # an infinite score).
    with np.errstate(divide="ignore", invalid="ignore"):
        per_week = np.where(denom[:, None] > 0, np.abs(actual - forecast) / denom[:, None], np.nan)
    return {
        "all": float(100 * (1 - np.nanmean(per_week))),
        "w1_w7": float(100 * (1 - np.nanmean(per_week[:, :7]))),
        "w8_w13": float(100 * (1 - np.nanmean(per_week[:, 7:]))),
    }


def backtest(
    panels: list[Panel],
    cutoffs: list[int],
    include_auxiliary: bool,
    alpha: float,
    stride: int,
    fast: bool,
) -> tuple[list[dict], list[tuple[np.ndarray, dict[str, np.ndarray], int]]]:
    test = next(p for p in panels if p.name == "test")
    rows, cached = [], []
    for cutoff in cutoffs:
        t0 = time.perf_counter()
        X, y = training_rows(panels, cutoff, include_auxiliary, stride)
        model = fit_model(X, y, alpha, fast)
        raw, info = forecast_at(model, test, cutoff)
        cached.append((raw, info, cutoff))
        rows.append(
            {
                "cutoff": str(test.weeks[cutoff].date()),
                "rows": len(y),
                "fit_seconds": round(time.perf_counter() - t0, 1),
                **score(test.values, cutoff, raw),
            }
        )
        print(f"cutoff {test.weeks[cutoff].date()} | {len(y):,} train rows | raw score {rows[-1]['all']:.2f}")
    return rows, cached


def choose_guard(
    cached: list[tuple[np.ndarray, dict[str, np.ndarray], int]], test: Panel, tune_count: int
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune only the older windows; reserve the newest window(s) as a check."""
    candidates = []
    for dead in (5, 7, 9, 12, 16, 24):
        for floor in (0.0, 0.20, 0.35, 0.50, 0.75, 1.0):
            for rate in (0.0, 0.04, 0.08, 0.12):
                vals = [score(test.values, c, apply_guard(raw, info, dead, floor, rate)) for raw, info, c in cached]
                candidates.append(
                    {
                        "dead_weeks": dead,
                        "floor": floor,
                        "min_rate": rate,
                        "tune_all": float(np.mean([v["all"] for v in vals[:tune_count]])),
                        "check_all": float(np.mean([v["all"] for v in vals[tune_count:]])),
                        "tune_w8_w13": float(np.mean([v["w8_w13"] for v in vals[:tune_count]])),
                        "check_w8_w13": float(np.mean([v["w8_w13"] for v in vals[tune_count:]])),
                    }
                )
    table = pd.DataFrame(candidates).sort_values(["tune_w8_w13", "tune_all"], ascending=False)
    best = table.iloc[0][["dead_weeks", "floor", "min_rate"]].to_dict()
    best["dead_weeks"] = int(best["dead_weeks"])
    return best, table


def select_candidate(
    candidates: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray], int]]], test: Panel, tune_count: int
) -> tuple[str, dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Choose model + guard from old windows, retaining recent ones as a check."""
    summaries, guard_tables = [], []
    selected_guards: dict[str, dict[str, float]] = {}
    for name, cached in candidates.items():
        guard, table = choose_guard(cached, test, tune_count)
        selected_guards[name] = guard
        guarded = [score(test.values, c, apply_guard(raw, info, **guard)) for raw, info, c in cached]
        summaries.append(
            {
                "candidate": name,
                **guard,
                "tune_all": float(np.mean([v["all"] for v in guarded[:tune_count]])),
                "check_all": float(np.mean([v["all"] for v in guarded[tune_count:]])),
                "tune_w8_w13": float(np.mean([v["w8_w13"] for v in guarded[:tune_count]])),
                "check_w8_w13": float(np.mean([v["w8_w13"] for v in guarded[tune_count:]])),
            }
        )
        guard_tables.append(table.assign(candidate=name))
    summary = pd.DataFrame(summaries).sort_values(["tune_w8_w13", "tune_all"], ascending=False)
    winner = summary.iloc[0]
    return str(winner["candidate"]), selected_guards[str(winner["candidate"])], summary, pd.concat(guard_tables, ignore_index=True)


def write_submission(test: Panel, forecast: np.ndarray, name: str) -> Path:
    ids = pd.read_parquet(ROOT / "test-shop" / "history.parquet")["article_id"].drop_duplicates().sort_values().to_numpy()
    out = pd.DataFrame(forecast, index=ids, columns=[f"w{i}" for i in range(1, HORIZON + 1)]).reset_index(names="article_id")
    assert out.shape == (2904, 14) and out.notna().all().all() and (out.iloc[:, 1:] >= 0).all().all()
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="test,a,c,d", help="Comma-separated: test,a,c,d")
    parser.add_argument("--with-aux", action="store_true", help="Allow A/C/D examples in the global model")
    parser.add_argument("--alpha", type=float, default=0.40, help="LightGBM quantile (lower is more conservative)")
    parser.add_argument("--windows", type=int, default=5, help="13-week rolling test-shop windows")
    parser.add_argument("--stride", type=int, default=2, help="Origin stride for the target shop")
    parser.add_argument("--fast", action="store_true", help="Use fewer trees and only three validation windows")
    parser.add_argument("--run-final", action="store_true", help="Fit full history and create a new submission CSV")
    args = parser.parse_args()

    names = [x.strip() for x in args.sources.split(",") if x.strip()]
    if "test" not in names:
        raise ValueError("--sources must include test")
    panels = load_sources(names)
    test = next(p for p in panels if p.name == "test")
    n_windows = 3 if args.fast else args.windows
    cutoffs = [test.values.shape[1] - HORIZON * (i + 1) - 1 for i in range(n_windows - 1, -1, -1)]
    rows, cached = backtest(panels, cutoffs, args.with_aux, args.alpha, args.stride, args.fast)
    # Compare learned direct forecasts with strong, deliberately simple local
    # forecasts.  This keeps an elaborate global model from being selected on
    # faith when a robust local quantile has lower MASE.
    candidates: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray], int]]] = {"direct_quantile": cached}
    for _, info, cutoff in cached:
        for name, forecast in baseline_forecasts(test, cutoff).items():
            candidates.setdefault(name, []).append((forecast, info, cutoff))

    # Oldest windows tune model/guard; newest windows are a less-biased check.
    tune_count = max(1, len(cached) - 2)
    winner, guard, selection, guard_table = select_candidate(candidates, test, tune_count)
    guarded = []
    for raw, info, cutoff in candidates[winner]:
        guarded.append({"cutoff": str(test.weeks[cutoff].date()), **score(test.values, cutoff, apply_guard(raw, info, **guard))})

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "backtest_raw.csv", index=False)
    pd.DataFrame(guarded).to_csv(OUT / "backtest_guarded.csv", index=False)
    selection.to_csv(OUT / "candidate_selection.csv", index=False)
    guard_table[guard_table["candidate"] == winner].head(30).to_csv(OUT / "guard_search.csv", index=False)
    report = {
        "sources_loaded": names,
        "auxiliary_examples_used": args.with_aux,
        "quantile_alpha": args.alpha,
        "selected_candidate": winner,
        "chosen_guard": guard,
        "raw_backtest": rows,
        "guarded_backtest": guarded,
        "candidate_selection": selection.to_dict(orient="records"),
    }
    (OUT / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("selected candidate:", winner, "| chosen guard:", guard)
    print("candidate selection:\n", selection.round(2).to_string(index=False))
    print("guarded backtest:\n", pd.DataFrame(guarded).round(2).to_string(index=False))

    if args.run_final:
        final_cutoff = test.values.shape[1] - 1
        if winner == "direct_quantile":
            X, y = training_rows(panels, final_cutoff, args.with_aux, args.stride)
            model = fit_model(X, y, args.alpha, args.fast)
            raw, info = forecast_at(model, test, final_cutoff)
        else:
            raw = baseline_forecasts(test, final_cutoff)[winner]
            info = forecast_info(test, final_cutoff)
        final = apply_guard(raw, info, **guard)
        path = write_submission(test, final, f"submission_alternative_{winner}.csv")
        print(f"saved {path}")
        print(f"all-zero forecasts: {(final.sum(axis=1) == 0).sum():,}/{test.n_series:,}; mean weekly forecast: {final.mean():.3f}")


if __name__ == "__main__":
    main()
