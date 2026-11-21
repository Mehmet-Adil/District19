"""Sliding-window, multi-step forecast for the test shop, with several regression algorithms.

The idea
--------
For every step k = 1..13 we train a model M_k: "given the last L weeks (default 20), predict the
week k steps after the last one".

The 13 forecast weeks are filled in order h = 1..13. For week h, every step k <= h gives a forecast:
the window of M_k ends k weeks before h. If that end is inside the history, the window is pure history
(a direct forecast). Otherwise its last weeks are our own forecasts for the weeks already filled
(recursive). So k = 1 is the classic "predict the 21st, slide one week, repeat", and k = h is a
purely direct forecast.

The forecasts for week h are averaged with weights that grow with k (bigger step = fewer of our own
forecasts in the window = less error piling up). Finally the 13 weeks are smoothed slightly.

Weight schemes (--schemes):
  recursive  only k = 1 (pure sliding window, for reference)
  direct     only k = h (pure direct, for reference)
  equal      k = 1..h, same weight
  linear     weight k
  square     weight k^2

Like the metric, inputs and targets are divided by the series' typical week-to-week change (the MASE
denominator up to the window end).

Usage (from this folder):
  ..\\..\\..\\.venv\\Scripts\\python.exe recursive_step_forecast.py --fast          # 2 windows, quick check
  ..\\..\\..\\.venv\\Scripts\\python.exe recursive_step_forecast.py                 # 8-window backtest
  ..\\..\\..\\.venv\\Scripts\\python.exe recursive_step_forecast.py --with-a 2      # + Set A, weight 2
  ..\\..\\..\\.venv\\Scripts\\python.exe recursive_step_forecast.py --final --scheme linear --smooth 0.5

Writes only to recursive_outputs/.
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

# CPU cores this script may use (set RSF_THREADS to change). Must be set before numpy / sklearn load.
THREADS = int(os.environ.get('RSF_THREADS', 4))
os.environ.setdefault('OMP_NUM_THREADS', str(THREADS))

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
OUT = BASE / 'recursive_outputs'
H = 13
SEED = 0


# ----------------------------------------------------------------------------- data
def load_test_shop():
    df = pd.read_parquet(BASE / 'test-shop' / 'history.parquet')
    df['week_start'] = pd.to_datetime(df['week_start'])
    Y = df.pivot(index='article_id', columns='week_start', values='units').sort_index(axis=1)
    return Y.astype(float)


def load_set_a():
    """Store x article weeks; weeks with in_stock = False are unknown (NaN), not zero."""
    df = pd.read_parquet(BASE / 'set-a' / 'weekly.parquet')
    df['units'] = df['units'].astype(float).where(df['in_stock'].astype(bool))
    A = df.pivot(index=['store_id', 'article_id'], columns='week_start', values='units').sort_index(axis=1)
    return A.astype(float)


def series_stats(M):
    """Per series and week t: scale (mean |week-to-week change| over weeks 0..t), age (weeks since the
    first sale, -1 before it) and weeks since the last sale."""
    n, T = M.shape
    d = np.abs(np.diff(M, axis=1))
    d = np.concatenate([np.full((n, 1), np.nan), d], axis=1)
    cnt = np.cumsum(np.isfinite(d), axis=1)
    cs = np.nancumsum(d, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = cs / cnt
        pos = np.where(M > 0, M, np.nan)
        nz_mean = np.nancumsum(np.nan_to_num(pos), axis=1) / np.cumsum(np.isfinite(pos), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, nz_mean)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    sold = np.nan_to_num(M) > 0
    started = np.maximum.accumulate(sold, axis=1)
    age = np.cumsum(started, axis=1) - 1
    age = np.where(started, age, -1)
    idx = np.where(sold, np.arange(T)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    wss = np.where(last >= 0, np.arange(T)[None, :] - last, 999)
    return scale, age, wss


def make_features(win, scale, age):
    """win: (n, L) units, scale: (n,), age: (n,) -> (n, L + 2) float32."""
    return np.column_stack([win / scale[:, None], np.log(scale), np.minimum(age, 200)]).astype(np.float32)


def origin_rows(M, scale, age, L, last_origin, last_target, weight, rng, zero_keep=0.1):
    """All origins t (window M[:, t-L+1 .. t]) with t <= last_origin. Targets for k = 1..H, NaN past
    last_target. Windows of only zeros are thinned to `zero_keep` and up-weighted to compensate."""
    Xs, Ys, Ws = [], [], []
    for t in range(L - 1, last_origin + 1):
        win = M[:, t - L + 1:t + 1]
        ok = np.isfinite(win).all(axis=1)
        allzero = ok & (np.nan_to_num(win).sum(axis=1) == 0)
        keep = ok & (~allzero | (rng.random(len(M)) < zero_keep))
        if not keep.any():
            continue
        sc = scale[keep, t]
        tgt = np.full((keep.sum(), H), np.nan, dtype=np.float32)
        for k in range(1, H + 1):
            if t + k <= last_target:
                tgt[:, k - 1] = M[keep, t + k] / sc
        w = np.where(allzero[keep], weight / zero_keep, weight)
        Xs.append(make_features(win[keep], sc, age[keep, t]))
        Ys.append(tgt)
        Ws.append(w.astype(np.float32))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)


# ----------------------------------------------------------------------------- models
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class LGB:
    def __init__(self, **kw):
        self.params = dict(objective='l1', learning_rate=0.1, num_leaves=31, min_data_in_leaf=200,
                           feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                           verbose=-1, seed=SEED, num_threads=THREADS)
        self.params.update(kw)
        self.rounds = self.params.pop('rounds', 150)

    def fit(self, X, y, w):
        self.m = lgb.train(self.params, lgb.Dataset(X, y, weight=w), num_boost_round=self.rounds)
        return self

    def predict(self, X):
        return self.m.predict(X)


class XGB:
    """XGBoost on the NVIDIA GPU (device='cuda'). `clip` caps the scaled target for squared loss."""
    def __init__(self, rounds=150, clip=None, **kw):
        self.params = dict(device='cuda', tree_method='hist', max_bin=128, seed=SEED, nthread=THREADS,
                           verbosity=0)
        self.params.update(kw)
        self.rounds, self.clip = rounds, clip

    def fit(self, X, y, w):
        y = np.minimum(y, self.clip) if self.clip is not None else y
        self.m = xgb.train(self.params, xgb.DMatrix(X, y, weight=w), num_boost_round=self.rounds)
        return self

    def predict(self, X):
        return self.m.inplace_predict(X)


class SK:
    """sklearn regressor; `clip` caps the scaled target for squared-loss models (means are pulled
    hard by the huge scaled values of new articles)."""
    def __init__(self, make, clip=None, max_rows=None):
        self.make, self.clip, self.max_rows = make, clip, max_rows

    def fit(self, X, y, w):
        if self.max_rows and len(X) > self.max_rows:
            idx = np.random.default_rng(SEED).choice(len(X), self.max_rows, replace=False)
            X, y, w = X[idx], y[idx], w[idx]
        if self.clip is not None:
            y = np.minimum(y, self.clip)
        self.m = self.make()
        final = self.m.steps[-1][1] if hasattr(self.m, 'steps') else self.m
        if isinstance(final, KNeighborsRegressor):          # no sample weights
            self.m.fit(X, y)
        elif hasattr(self.m, 'steps'):                       # pipelines need the step name
            self.m.fit(X, y, **{f'{self.m.steps[-1][0]}__sample_weight': w})
        else:
            self.m.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(X)


def _linear(est):
    return lambda: make_pipeline(FunctionTransformer(signed_log), StandardScaler(), est)


ALGOS = {
    # GPU
    'xgb_l1_gpu': lambda: XGB(objective='reg:absoluteerror', learning_rate=0.1, max_depth=8, min_child_weight=200,
                              subsample=0.7, colsample_bynode=0.8, reg_lambda=1.0),
    'xgb_rf_gpu': lambda: XGB(rounds=1, clip=50, objective='reg:squarederror', learning_rate=1.0,
                              num_parallel_tree=100, max_depth=12, min_child_weight=30, subsample=0.3,
                              colsample_bynode=0.5),
    # CPU
    'lightgbm_l1': lambda: LGB(),
    'lightgbm_q45': lambda: LGB(objective='quantile', alpha=0.45),
    'hist_gb_l1': lambda: SK(lambda: HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=200, learning_rate=0.05, min_samples_leaf=200, random_state=SEED)),
    'random_forest': lambda: SK(lambda: RandomForestRegressor(
        n_estimators=60, min_samples_leaf=30, max_features=0.5, max_samples=0.3, n_jobs=THREADS,
        random_state=SEED), clip=50),
    'extra_trees': lambda: SK(lambda: ExtraTreesRegressor(
        n_estimators=60, min_samples_leaf=30, max_features=0.7, bootstrap=True, max_samples=0.3, n_jobs=THREADS,
        random_state=SEED), clip=50),
    'ridge': lambda: SK(_linear(Ridge(alpha=1.0)), clip=50),
    'linear_l1': lambda: SK(_linear(SGDRegressor(loss='epsilon_insensitive', epsilon=0.0, alpha=1e-5,
                                                 average=True, max_iter=10, tol=None, random_state=SEED))),
    'knn': lambda: SK(lambda: make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                                            KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=THREADS)),
                      clip=50, max_rows=150_000),
}


# Default run: the GPU models plus the cheap CPU ones (lightgbm_*, random_forest and knn are slow on a
# busy CPU; pass them with --algos to include them).
DEFAULT_ALGOS = ['xgb_l1_gpu', 'xgb_rf_gpu', 'hist_gb_l1', 'extra_trees', 'ridge', 'linear_l1']


# ----------------------------------------------------------------------------- forecasting
def step_weights(scheme, h):
    k = np.arange(1, h + 1, dtype=float)
    w = {'recursive': (k == 1).astype(float), 'direct': (k == h).astype(float), 'equal': np.ones(h),
         'linear': k, 'square': k ** 2}[scheme]
    return w / w.sum()


def smooth(F, a):
    """Blend each week with the mean of itself and its neighbours (edges: the neighbour that exists)."""
    if a <= 0:
        return F
    P = np.pad(F, ((0, 0), (1, 1)), mode='edge')
    ma = (P[:, :-2] + P[:, 1:-1] + P[:, 2:]) / 3
    ma[:, 0], ma[:, -1] = (F[:, 0] + F[:, 1]) / 2, (F[:, -2] + F[:, -1]) / 2
    return (1 - a) * F + a * ma


def forecast(models, M, scale, age, c, L, scheme, cache):
    """Fill weeks c+1..c+H for every series of M. `cache` holds the direct (history-only) predictions,
    which are the same for every scheme."""
    n = len(M)
    ext = np.zeros((n, c + 1 + H))
    ext[:, :c + 1] = M[:, :c + 1]
    sc, ag = scale[:, c], age[:, c]
    for h in range(1, H + 1):
        w = step_weights(scheme, h)
        acc = np.zeros(n)
        for k in range(1, h + 1):
            if w[k - 1] == 0:
                continue
            off = h - k                                   # how many of our own forecasts the window holds
            if off == 0 and k in cache:
                p = cache[k]
            else:
                end = c + off
                win = ext[:, end - L + 1:end + 1]
                p = np.clip(models[k].predict(make_features(win, sc, np.where(ag >= 0, ag + off, -1))), 0, None) * sc
                if off == 0:
                    cache[k] = p
            acc += w[k - 1] * p
        ext[:, c + h] = acc
    return ext[:, c + 1:]


def postprocess(F, wss_c, age_c, dead_weeks=8, young_zero=True):
    F = np.clip(F, 0, None)
    F = np.where((wss_c >= dead_weeks)[:, None], 0, F)
    if young_zero:
        F = np.where(((age_c >= 0) & (age_c < 13))[:, None], 0, F)
    return np.round(F)


def score(Y, c, F):
    """(1 - MASE) x 100 for weeks 1-7, 8-13 and all 13, as in DATASETS.md."""
    hist, fut = Y[:, :c + 1], Y[:, c + 1:c + 1 + H]
    den = np.abs(np.diff(hist, axis=1)).mean(axis=1)
    ok = den > 0
    e = np.abs(fut - F) / np.where(ok, den, 1)[:, None]
    return {'w1-7': (1 - e[ok, :7].mean(1).mean()) * 100, 'w8-13': (1 - e[ok, 7:].mean(1).mean()) * 100,
            'all': (1 - e[ok].mean(1).mean()) * 100}


# ----------------------------------------------------------------------------- runs
def fit_models(algo, X, Yt, W):
    models = {}
    for k in range(1, H + 1):
        ok = np.isfinite(Yt[:, k - 1])
        models[k] = ALGOS[algo]().fit(X[ok], Yt[ok, k - 1], W[ok])
    return models


def training_set(Ym, ts_stats, c, L, set_a, rng):
    scale, age, _ = ts_stats
    X, Yt, W = origin_rows(Ym, scale, age, L, last_origin=c - 1, last_target=c, weight=1.0, rng=rng)
    if set_a is not None:
        A, a_stats, a_weight = set_a
        Xa, Ya, Wa = origin_rows(A, a_stats[0], a_stats[1], L, last_origin=A.shape[1] - 2,
                                 last_target=A.shape[1] - 1, weight=a_weight, rng=rng)
        X, Yt, W = np.concatenate([X, Xa]), np.concatenate([Yt, Ya]), np.concatenate([W, Wa])
    return X, Yt, W


def backtest(args, Ym, ts_stats, set_a):
    T = Ym.shape[1]
    cutoffs = [T - 1 - H - args.step * i for i in range(args.windows)]
    scale, age, wss = ts_stats
    rows, all_F = [], {}
    for c in cutoffs:
        t0 = time.perf_counter()
        X, Yt, W = training_set(Ym, ts_stats, c, args.window, set_a, np.random.default_rng(SEED))
        print(f'\n=== cutoff index {c} | {X.shape[0]:,} training rows ({time.perf_counter() - t0:.0f}s)', flush=True)
        for name, F in reference_forecasts(Ym, c).items():
            for yz in (False, True):
                G = postprocess(F, wss[:, c], age[:, c], young_zero=yz)
                rows.append({'cutoff': c, 'algo': name, 'scheme': '-', 'smooth': 0, 'young_zero': yz, **score(Ym, c, G)})
        for algo in args.algos:
            t0 = time.perf_counter()
            models = fit_models(algo, X, Yt, W)
            fit_s = time.perf_counter() - t0
            cache = {}
            for scheme in args.schemes:
                F_raw = forecast(models, Ym, scale, age, c, args.window, scheme, cache)
                for a in args.smooths:
                    F = smooth(F_raw, a)
                    all_F[(c, algo, scheme, a)] = F
                    for yz in (False, True):
                        G = postprocess(F, wss[:, c], age[:, c], young_zero=yz)
                        rows.append({'cutoff': c, 'algo': algo, 'scheme': scheme, 'smooth': a, 'young_zero': yz,
                                     **score(Ym, c, G)})
            best = max((r for r in rows if r['cutoff'] == c and r['algo'] == algo), key=lambda r: r['all'])
            print(f'  {algo:<14s} fit {fit_s:5.0f}s | total {time.perf_counter() - t0:5.0f}s | best here: '
                  f'{best["scheme"]}/{best["smooth"]}/young0={best["young_zero"]} -> {best["all"]:.2f}', flush=True)
        # median of all algorithms (same scheme / smoothing)
        for scheme in args.schemes:
            for a in args.smooths:
                F = np.median(np.stack([all_F[(c, al, scheme, a)] for al in args.algos]), axis=0)
                for yz in (False, True):
                    G = postprocess(F, wss[:, c], age[:, c], young_zero=yz)
                    rows.append({'cutoff': c, 'algo': 'ENS_median_all', 'scheme': scheme, 'smooth': a,
                                 'young_zero': yz, **score(Ym, c, G)})
    return pd.DataFrame(rows)


def reference_forecasts(Ym, c):
    hist = Ym[:, :c + 1]
    rep = lambda v: np.repeat(v[:, None], H, axis=1)
    return {'REF_median_26w': rep(np.median(hist[:, -26:], axis=1)),
            'REF_median_13w': rep(np.median(hist[:, -13:], axis=1))}


def final(args, Y, Ym, ts_stats, set_a):
    c = Ym.shape[1] - 1
    scale, age, wss = ts_stats
    X, Yt, W = training_set(Ym, ts_stats, c, args.window, set_a, np.random.default_rng(SEED))
    print(f'final fit: origin index {c} ({Y.columns[-1].date()}), {X.shape[0]:,} training rows')
    OUT.mkdir(exist_ok=True)
    Fs = {}
    for algo in args.algos:
        t0 = time.perf_counter()
        models = fit_models(algo, X, Yt, W)
        Fs[algo] = smooth(forecast(models, Ym, scale, age, c, args.window, args.scheme, {}), args.smooth)
        print(f'  {algo} ({time.perf_counter() - t0:.0f}s)')
    Fs['ENS_median_all'] = np.median(np.stack(list(Fs.values())), axis=0)
    for name, F in Fs.items():
        G = postprocess(F, wss[:, c], age[:, c], young_zero=not args.no_young_zero)
        sub = pd.DataFrame(G, index=Y.index, columns=[f'w{i}' for i in range(1, H + 1)]).reset_index()
        assert len(sub) == 2904 and not sub.isna().any().any()
        path = OUT / f'submission_recursive_{name}_{args.scheme}_s{args.smooth}.csv'
        sub.to_csv(path, index=False)
        print(f'  wrote {path.name}: mean {G.mean():.2f}, articles all-zero {(G.sum(1) == 0).sum()}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--window', type=int, default=20, help='weeks given to the model (L)')
    p.add_argument('--windows', type=int, default=6, help='backtest windows')
    p.add_argument('--step', type=int, default=4, help='weeks between backtest cutoffs')
    p.add_argument('--algos', nargs='+', default=DEFAULT_ALGOS, help=f'any of {list(ALGOS)}')
    p.add_argument('--schemes', nargs='+', default=['recursive', 'direct', 'equal', 'linear', 'square'])
    p.add_argument('--smooths', nargs='+', type=float, default=[0.0, 0.3, 0.6])
    p.add_argument('--with-a', type=float, default=0.0, help='also train on Set A with this sample weight (0 = off)')
    p.add_argument('--fast', action='store_true', help='2 windows, 3 algorithms')
    p.add_argument('--final', action='store_true', help='fit on all history and write submissions')
    p.add_argument('--scheme', default='linear', help='--final: weight scheme')
    p.add_argument('--smooth', type=float, default=0.3, help='--final: smoothing strength')
    p.add_argument('--no-young-zero', action='store_true', help='--final: keep forecasts for articles < 13 weeks old')
    p.add_argument('--tag', default='')
    args = p.parse_args()
    if args.fast:
        args.windows, args.algos = 2, ['xgb_l1_gpu', 'xgb_rf_gpu', 'ridge']

    Y = load_test_shop()
    Ym = Y.to_numpy(float)
    ts_stats = series_stats(Ym)
    set_a = None
    if args.with_a > 0:
        A = load_set_a().to_numpy(float)
        set_a = (A, series_stats(A), args.with_a)

    if args.final:
        final(args, Y, Ym, ts_stats, set_a)
        return

    t0 = time.perf_counter()
    res = backtest(args, Ym, ts_stats, set_a)
    OUT.mkdir(exist_ok=True)
    tag = args.tag or ('fast' if args.fast else f'L{args.window}' + (f'_A{args.with_a:g}' if set_a else ''))
    res.to_csv(OUT / f'backtest_{tag}.csv', index=False)
    summ = (res.groupby(['algo', 'scheme', 'smooth', 'young_zero'])[['w1-7', 'w8-13', 'all']].mean()
            .sort_values('w8-13', ascending=False))
    summ.round(2).to_csv(OUT / f'summary_{tag}.csv')
    pd.set_option('display.width', 200)
    print(f'\nMean over {args.windows} windows ({time.perf_counter() - t0:.0f}s). Top 25 by w8-13:')
    print(summ.head(25).round(2).to_string())
    print('\nBest config per algorithm:')
    print(summ.reset_index().groupby('algo').head(1).set_index('algo').round(2).to_string())
    (OUT / f'run_{tag}.json').write_text(json.dumps(vars(args), indent=2))


if __name__ == '__main__':
    main()
