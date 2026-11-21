"""A second, independent forecasting attempt for the LemeBase retail challenge.

This file deliberately does not import `alternative_direct_forecast.py` or the
existing notebooks.  The first alternative selected a flat rolling quantile.
Its live week-by-week result showed that a flat level can miss short seasonal
events, so this script tests *shrunk* annual horizon effects instead of blindly
copying last year.

All candidates are local per-article forecasts and need only the raw test-shop
history.  The seasonal adjustment is always shrunk to a rolling quantile and
can be gated to regular sellers; this prevents a noisy one-off sale last year
from creating a non-zero forecast for an intermittent article.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


H = 13
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "seasonal_outputs"


@dataclass(frozen=True)
class Spec:
    base_q: float
    base_window: int
    kind: str = "flat"  # flat, add, level_blend, ratio_blend, global_ratio
    strength: float = 0.0
    min_rate: float = 0.0
    dormant_weeks: int = 5
    floor: float = 0.0

    def name(self) -> str:
        return (
            f"q{int(self.base_q * 100):02d}_w{self.base_window}_"
            f"{self.kind}_s{self.strength:.2f}_r{self.min_rate:.2f}_"
            f"dead{self.dormant_weeks}_floor{self.floor:.2f}"
        )


def load_history() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_parquet(ROOT / "test-shop" / "history.parquet")
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    weeks = pd.DatetimeIndex(sorted(frame.week_start.unique()))
    wide = frame.pivot(index="article_id", columns="week_start", values="units").reindex(columns=weeks)
    assert wide.shape == (2904, 143) and wide.notna().all().all()
    return wide.to_numpy(float), weeks, wide.index.to_numpy()


def recency_and_rate(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sold = hist > 0
    age = sold[:, ::-1].argmax(axis=1).astype(float)
    age[~sold.any(axis=1)] = np.inf
    return age, sold[:, -min(26, hist.shape[1]) :].mean(axis=1)


def annual_effects(x: np.ndarray, cutoff: int, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last year's horizon values, their prior local level, and a chain factor.

    For h=1, the annual observation is x[t-51], the week corresponding to the
    first forecast week one year ago.  The anchor is the 26 weeks before that
    prior forecast block, so an additive residual expresses a yearly *change*
    rather than another demand-level estimate.
    """
    n = x.shape[0]
    seasonal = np.empty((n, H), dtype=float)
    anchor = np.empty((n, H), dtype=float)
    chain_factor = np.ones(H, dtype=float)
    for h in range(1, H + 1):
        idx = cutoff + h - 52
        if idx < 0:
            seasonal[:, h - 1] = base[:, 0]
            anchor[:, h - 1] = base[:, 0]
            continue
        seasonal[:, h - 1] = x[:, idx]
        start = max(0, idx - 26)
        prior = x[:, start:idx]
        anchor[:, h - 1] = np.quantile(prior, 0.40, axis=1) if prior.shape[1] else base[:, 0]
        prior_total = np.median(x[:, start:idx].sum(axis=0)) if prior.shape[1] else 0.0
        chain_factor[h - 1] = x[:, idx].sum() / prior_total if prior_total > 0 else 1.0
    # Retail peaks are real but the global ratio should not explode because of
    # assortment turnover or a one-week chain-wide outlier.
    return seasonal, anchor, np.clip(chain_factor, 0.55, 1.80)


def forecast(x: np.ndarray, cutoff: int, spec: Spec) -> np.ndarray:
    hist = x[:, : cutoff + 1]
    point = np.quantile(hist[:, -spec.base_window :], spec.base_q, axis=1)
    base = np.repeat(point[:, None], H, axis=1)
    seasonal, anchor, chain_factor = annual_effects(x, cutoff, base)
    if spec.kind == "flat":
        out = base
    elif spec.kind == "add":
        # A small annual residual works even where the level has changed.
        out = base + spec.strength * (seasonal - anchor)
    elif spec.kind == "level_blend":
        # Convex blend is useful for stable best sellers with repeated events.
        out = (1 - spec.strength) * base + spec.strength * seasonal
    elif spec.kind == "ratio_blend":
        ratio = (seasonal + 0.5) / (anchor + 0.5)
        out = base * np.power(np.clip(ratio, 0.25, 4.0), spec.strength)
    elif spec.kind == "global_ratio":
        out = base * (1 + spec.strength * (chain_factor[None, :] - 1))
    else:
        raise ValueError(spec.kind)

    age, rate = recency_and_rate(hist)
    keep = (age < spec.dormant_weeks) & (rate >= spec.min_rate)
    out = np.where(keep[:, None], out, 0.0)
    out = np.clip(out, 0, None)
    return np.where(out >= spec.floor, out, 0.0)


def score(x: np.ndarray, cutoff: int, fc: np.ndarray) -> dict[str, float]:
    hist = x[:, : cutoff + 1]
    actual = x[:, cutoff + 1 : cutoff + H + 1]
    denominator = np.abs(np.diff(hist, axis=1)).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(denominator[:, None] > 0, np.abs(actual - fc) / denominator[:, None], np.nan)
    out = {
        "all": float(100 * (1 - np.nanmean(err))),
        "w1_w7": float(100 * (1 - np.nanmean(err[:, :7]))),
        "w8_w13": float(100 * (1 - np.nanmean(err[:, 7:]))),
    }
    out.update({f"h{h + 1}": float(100 * (1 - np.nanmean(err[:, h]))) for h in range(H)})
    return out


def candidate_specs() -> list[Spec]:
    bases = [(0.35, 13), (0.40, 20), (0.40, 26), (0.45, 20), (0.50, 20), (0.50, 26)]
    specs: list[Spec] = []
    for q, window in bases:
        for dormant in (3, 5, 7, 9):
            for floor in (0.0, 0.5, 1.0):
                specs.append(Spec(q, window, dormant_weeks=dormant, floor=floor))

    # Let recurring annual patterns contribute only a fraction of the forecast.
    # Restrict local annual adjustments to a reasonably active article.
    for q, window in [(0.40, 26), (0.45, 20), (0.50, 26)]:
        for kind in ("add", "level_blend", "ratio_blend"):
            for strength in (0.10, 0.20, 0.35, 0.50):
                for min_rate in (0.0, 0.20, 0.40):
                    for dormant in (3, 5, 7):
                        specs.append(Spec(q, window, kind, strength, min_rate, dormant, 0.0))
        for strength in (0.20, 0.40, 0.60, 0.80):
            for dormant in (3, 5, 7):
                specs.append(Spec(q, window, "global_ratio", strength, 0.0, dormant, 0.0))
    return specs


def evaluate(x: np.ndarray, weeks: pd.DatetimeIndex, specs: list[Spec], windows: int) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    # Oldest -> newest permits a final two-window check that was not used to
    # choose the candidate.  The newest validation origin is 13 weeks before
    # the last observed target, exactly mirroring the competition horizon.
    cutoffs = [x.shape[1] - H * (i + 1) - 1 for i in range(windows - 1, -1, -1)]
    rows, forecasts = [], {}
    for spec in specs:
        key = spec.name()
        forecasts[key] = []
        for cutoff in cutoffs:
            fc = forecast(x, cutoff, spec)
            forecasts[key].append(fc)
            rows.append({"candidate": key, "cutoff": str(weeks[cutoff].date()), **asdict(spec), **score(x, cutoff, fc)})
    frame = pd.DataFrame(rows)
    return frame, forecasts


def choose(results: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Rank by old validation windows and report the newest two as a check."""
    all_cutoffs = sorted(results.cutoff.unique())
    tune = all_cutoffs[:-2] if len(all_cutoffs) >= 4 else all_cutoffs[:-1]
    check = all_cutoffs[-2:] if len(all_cutoffs) >= 4 else all_cutoffs[-1:]
    summary = results.groupby("candidate", as_index=False).agg(
        mean_all=("all", "mean"),
        median_all=("all", "median"),
        mean_w1_w7=("w1_w7", "mean"),
        mean_w8_w13=("w8_w13", "mean"),
    )
    tune_summary = results[results.cutoff.isin(tune)].groupby("candidate", as_index=False).agg(
        tune_all=("all", "mean"), tune_w1_w7=("w1_w7", "mean"), tune_w8_w13=("w8_w13", "mean")
    )
    check_summary = results[results.cutoff.isin(check)].groupby("candidate", as_index=False).agg(
        check_all=("all", "mean"), check_w1_w7=("w1_w7", "mean"), check_w8_w13=("w8_w13", "mean")
    )
    summary = summary.merge(tune_summary, on="candidate").merge(check_summary, on="candidate")
    # Live leaderboard observes w1–w7.  Keep later weeks in the score so a
    # candidate cannot win merely by moving all errors into the final horizon.
    summary["selection_score"] = 0.70 * summary.tune_w1_w7 + 0.30 * summary.tune_w8_w13
    summary = summary.sort_values(["selection_score", "tune_all"], ascending=False).reset_index(drop=True)
    return summary.iloc[0], summary


def horizon_plan(results: pd.DataFrame, train_cutoffs: list[str]) -> list[str]:
    """Pick one candidate per horizon using only the supplied prior cutoffs."""
    mean_scores = results[results.cutoff.isin(train_cutoffs)].groupby("candidate")[[f"h{i}" for i in range(1, H + 1)]].mean()
    return [str(mean_scores[f"h{i}"].idxmax()) for i in range(1, H + 1)]


def horizon_hybrid(forecasts: dict[str, list[np.ndarray]], forecast_index: int, plan: list[str]) -> np.ndarray:
    return np.column_stack([forecasts[candidate][forecast_index][:, h] for h, candidate in enumerate(plan)])


def validate_horizon_hybrid(
    x: np.ndarray, weeks: pd.DatetimeIndex, results: pd.DataFrame, forecasts: dict[str, list[np.ndarray]]
) -> tuple[pd.DataFrame, list[str]]:
    """Walk-forward validation of horizon-specific selection.

    Each validation origin uses only earlier origins to decide which candidate
    supplies its h1, h2, ..., h13 values.  It is deliberately more stringent
    than fitting a horizon mixture on all backtests then quoting its in-sample
    score.
    """
    cutoffs = sorted(results.cutoff.unique())
    cutoff_index = {str(week.date()): i for i, week in enumerate(weeks)}
    rows = []
    for pos in range(3, len(cutoffs)):
        plan = horizon_plan(results, cutoffs[:pos])
        cutoff = cutoff_index[cutoffs[pos]]
        fc = horizon_hybrid(forecasts, pos, plan)
        rows.append({"cutoff": cutoffs[pos], **score(x, cutoff, fc), **{f"model_h{i + 1}": name for i, name in enumerate(plan)}})
    return pd.DataFrame(rows), horizon_plan(results, cutoffs)


def write_submission(ids: np.ndarray, fc: np.ndarray, filename: str) -> Path:
    out = pd.DataFrame(fc, index=ids, columns=[f"w{i}" for i in range(1, H + 1)]).reset_index(names="article_id")
    assert out.shape == (2904, 14) and out.article_id.is_unique
    assert np.isfinite(out.iloc[:, 1:].to_numpy()).all() and (out.iloc[:, 1:] >= 0).all().all()
    OUT.mkdir(exist_ok=True)
    path = OUT / filename
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--run-final", action="store_true")
    args = parser.parse_args()

    x, weeks, ids = load_history()
    specs = candidate_specs()
    results, forecasts = evaluate(x, weeks, specs, args.windows)
    winner, summary = choose(results)
    hybrid_validation, final_horizon_plan = validate_horizon_hybrid(x, weeks, results, forecasts)
    OUT.mkdir(exist_ok=True)
    results.to_csv(OUT / "all_window_scores.csv", index=False)
    summary.to_csv(OUT / "candidate_summary.csv", index=False)
    hybrid_validation.to_csv(OUT / "horizon_hybrid_validation.csv", index=False)
    pd.DataFrame({"horizon": [f"h{i}" for i in range(1, H + 1)], "candidate": final_horizon_plan}).to_csv(
        OUT / "horizon_hybrid_plan.csv", index=False
    )
    (OUT / "run_report.json").write_text(
        json.dumps(
            {
                "selected_candidate": winner.to_dict(),
                "horizon_hybrid_plan": final_horizon_plan,
                "horizon_hybrid_validation": hybrid_validation.to_dict(orient="records"),
                "windows": args.windows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("selected candidate:\n", winner.to_string())
    print("\ntop candidates:\n", summary.head(20).round(3).to_string(index=False))
    print("\nhorizon hybrid validation:\n", hybrid_validation[["cutoff", "all", "w1_w7", "w8_w13"]].round(3).to_string(index=False))
    print("\nfinal horizon plan:", final_horizon_plan)

    if args.run_final:
        spec = {candidate.name(): candidate for candidate in specs}[str(winner["candidate"])]
        fc = forecast(x, x.shape[1] - 1, spec)
        path = write_submission(ids, fc, "submission_seasonal_horizon_selected.csv")
        final_forecasts = {candidate.name(): forecast(x, x.shape[1] - 1, candidate) for candidate in specs}
        hybrid_fc = np.column_stack([final_forecasts[candidate][:, h] for h, candidate in enumerate(final_horizon_plan)])
        hybrid_path = write_submission(ids, hybrid_fc, "submission_seasonal_horizon_hybrid.csv")
        print(f"\nsaved {path}")
        print(f"saved {hybrid_path}")
        print(f"all-zero articles: {(fc.sum(axis=1) == 0).sum():,}/{len(ids):,}; mean forecast: {fc.mean():.3f}")
        print(f"hybrid all-zero articles: {(hybrid_fc.sum(axis=1) == 0).sum():,}/{len(ids):,}; mean forecast: {hybrid_fc.mean():.3f}")


if __name__ == "__main__":
    main()
