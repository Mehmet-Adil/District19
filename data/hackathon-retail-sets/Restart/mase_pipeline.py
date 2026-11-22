"""MASE-aligned forecasting pipeline for the test shop.

The competition score is (1 - MASE) * 100, where MASE is computed per article
(13-week MAE / the article's mean absolute week-to-week change in its history)
and then averaged UNWEIGHTED over articles. Everything here is aligned to that:

* the global model's target and lag features are divided by each article's
  MASE denominator, and every row gets the same weight, so the L1 / quantile
  loss the model minimises is exactly the per-article MASE;
* validation uses several rolling 13-week windows and reports w1-w7 and
  w8-w13 separately (the final ranking uses w8-w13);
* a post-processing guard zeros out dead / very slow articles, where
  over-forecasting is the biggest risk.
"""
import numpy as np
import pandas as pd

H = 13
EPS = 1e-3          # floor for the MASE denominator (articles that never moved)


# --------------------------------------------------------------------------- data
def load_wide(path='../test-shop/history.parquet'):
    """units and on_hand as (articles x weeks) arrays."""
    d = pd.read_parquet(path)
    d['week_start'] = pd.to_datetime(d['week_start'])
    units = d.pivot(index='article_id', columns='week_start', values='units').sort_index()
    on_hand = d.pivot(index='article_id', columns='week_start', values='on_hand').reindex_like(units)
    return units, on_hand


def load_set_a_wide(path='../set-a/weekly.parquet'):
    """Set A as (store x article) series. Out-of-stock weeks become NaN (masked)."""
    a = pd.read_parquet(path)
    a['week_start'] = pd.to_datetime(a['week_start'])
    a['sid'] = 'A_' + a['store_id'].astype(str) + '_' + a['article_id'].astype(str)
    u = a.pivot(index='sid', columns='week_start', values='units').sort_index()
    s = a.pivot(index='sid', columns='week_start', values='in_stock').reindex_like(u)
    return u, s.astype(bool)


def mase_denom(Y, floor=EPS):
    """Mean absolute week-to-week change per row (NaN-aware), floored."""
    d = np.nanmean(np.abs(np.diff(Y, axis=1)), axis=1)
    return np.maximum(np.nan_to_num(d, nan=0.0), floor)


# ------------------------------------------------------------------------ scoring
def score(Y_true, F, denom):
    """Competition score: mean over articles of (1 - MAE/denom) * 100.
    Articles with denom == 0 (never moved in their history) are skipped: in the
    real task every article has a non-zero denominator, so they only appear in
    backtests (articles not launched yet at the cutoff)."""
    ok = denom > 0
    err = np.abs(Y_true - F)[ok]
    denom = denom[ok]
    m = lambda e: 100 * (1 - (e.mean(axis=1) / denom).mean())
    return {'all': m(err), 'w1_7': m(err[:, :7]), 'w8_13': m(err[:, 7:])}


def windows(n_weeks, n_windows=4):
    """Start index of each 13-week validation window, newest first."""
    return [n_weeks - H * (k + 1) for k in range(n_windows)]


def evaluate(Y, forecaster, n_windows=4, verbose=True, **kw):
    """Run `forecaster(Y_hist, **kw) -> (n, 13)` on each rolling window."""
    rows = []
    for c in windows(Y.shape[1], n_windows):
        hist, true = Y[:, :c], Y[:, c:c + H]
        F = np.clip(forecaster(hist, **kw), 0, None)
        s = score(true, F, mase_denom(hist, floor=0.0))
        s['cutoff_idx'] = c
        rows.append(s)
    out = pd.DataFrame(rows).set_index('cutoff_idx')
    out.loc['mean'] = out.mean()
    if verbose:
        print(out.round(2).to_string())
    return out


# ---------------------------------------------------------------------- baselines
def weeks_since_sale(hist):
    """Weeks since last non-zero week (len(hist) if never sold)."""
    sold = hist > 0
    last = np.where(sold.any(axis=1), hist.shape[1] - 1 - np.argmax(sold[:, ::-1], axis=1), -1)
    return hist.shape[1] - 1 - last


def median_baseline(hist, k=26, dead_after=None):
    f = np.median(hist[:, -k:], axis=1)
    if dead_after is not None:
        f = np.where(weeks_since_sale(hist) >= dead_after, 0.0, f)
    return np.repeat(f[:, None], H, axis=1)


def quantile_baseline(hist, k=26, q=0.5, dead_after=None):
    f = np.quantile(hist[:, -k:], q, axis=1)
    if dead_after is not None:
        f = np.where(weeks_since_sale(hist) >= dead_after, 0.0, f)
    return np.repeat(f[:, None], H, axis=1)


# ------------------------------------------------------------------ global model
def _roll(Z, o, k, fn):
    lo = max(0, o - k + 1)
    return fn(Z[:, lo:o + 1], axis=1)


def origin_block(Y, OH, o):
    """Features of every series as of week index o (data up to and including o).
    Returns (features DataFrame, scale s_o, sold_before mask)."""
    n = Y.shape[0]
    hist = Y[:, :o + 1]
    # expanding MASE denominator: exactly what the scorer will use at the final origin
    s = mase_denom(hist, floor=0.0) if o >= 1 else np.zeros(n)
    sold = (hist > 0).any(axis=1)
    s_safe = np.where(s > 0, s, 1.0)
    Z = hist / s_safe[:, None]
    f = {}
    for k in range(13):
        f[f'lag{k}'] = Z[:, o - k] if o - k >= 0 else np.full(n, np.nan)
    for k in (4, 8, 13, 26, 52):
        f[f'mean{k}'] = _roll(Z, o, k, np.mean)
    for k in (8, 13, 26, 52):
        f[f'med{k}'] = _roll(Z, o, k, np.median)
    w26 = Z[:, max(0, o - 25):o + 1]
    f['q25_26'] = np.quantile(w26, 0.25, axis=1)
    f['q75_26'] = np.quantile(w26, 0.75, axis=1)
    f['q90_52'] = _roll(Z, o, 52, lambda a, axis: np.quantile(a, 0.9, axis=axis))
    f['std13'] = _roll(Z, o, 13, np.std)
    f['max13'] = _roll(Z, o, 13, np.max)
    for k in (4, 13, 26, 52):
        f[f'zero{k}'] = _roll(hist == 0, o, k, np.mean)
    wss = weeks_since_sale(hist).astype(float)
    f['weeks_since_sale'] = np.minimum(wss, 156)
    first = np.where(sold, np.argmax(hist > 0, axis=1), o + 1)
    f['age'] = (o - first).astype(float)
    nz = Z[:, max(0, o - 51):o + 1]
    cnt = (nz > 0).sum(axis=1)
    f['nz_size52'] = np.where(cnt > 0, nz.sum(axis=1) / np.maximum(cnt, 1), 0.0)
    f['adi52'] = np.where(cnt > 0, nz.shape[1] / np.maximum(cnt, 1), 99.0)
    f['log_s'] = np.log(np.where(s > 0, s, 1e-4))
    f['log_mean52_raw'] = np.log1p(_roll(hist, o, 52, np.mean))
    oh = OH[:, o]
    f['oh_nan'] = np.isnan(oh).astype(float)
    f['oh_pos'] = (np.nan_to_num(oh) > 0).astype(float)
    f['oh_cover'] = np.clip(np.nan_to_num(oh) / (_roll(hist, o, 13, np.mean) + 0.1), -5, 100)
    f['oh_scaled'] = np.clip(np.nan_to_num(oh) / s_safe, -50, 500)
    return pd.DataFrame(f), s, sold


def target_feats(Y, s_safe, o, h):
    """Known-in-advance features for target week o + h: last year's value and the
    last-year ratio of (weeks around the target) to (13 weeks before the origin)."""
    n = Y.shape[0]
    t = o + h
    f = {'h': np.full(n, h, dtype=float)}
    if t - 52 >= 0:
        f['lag52_t'] = Y[:, t - 52] / s_safe
        lo, hi = max(0, t - 53), min(Y.shape[1], t - 50)
        f['lag52_t3'] = Y[:, lo:hi].mean(axis=1) / s_safe
    else:
        f['lag52_t'] = f['lag52_t3'] = np.full(n, np.nan)
    if o - 52 - 12 >= 0:
        base = Y[:, o - 64:o - 51].mean(axis=1)
        around = Y[:, max(0, t - 53):t - 50].mean(axis=1)
        f['ly_ratio'] = np.clip((around + 0.1) / (base + 0.1), 0, 10)
    else:
        f['ly_ratio'] = np.full(n, np.nan)
    return pd.DataFrame(f)


def build_rows(Y, OH, origins, horizons=range(1, H + 1), max_target=None, with_target=True):
    """Stack (series, origin, horizon) rows. Target = Y[o+h] / s_o."""
    parts = []
    for o in origins:
        F, s, sold = origin_block(Y, OH, o)
        keep = sold & (s > 0)
        s_safe = np.where(s > 0, s, 1.0)
        for h in horizons:
            t = o + h
            if with_target and (t >= Y.shape[1] or (max_target is not None and t >= max_target)):
                continue
            part = pd.concat([F, target_feats(Y, s_safe, o, h)], axis=1)
            part['series'] = np.arange(Y.shape[0])
            part['origin'] = o
            part['scale'] = s
            if with_target:
                part['y'] = Y[:, t] / s_safe
                part['y_raw'] = Y[:, t]
            parts.append(part[keep])
    return pd.concat(parts, ignore_index=True)


META = {'series', 'origin', 'scale', 'y', 'y_raw', 'src'}


def feat_cols(df):
    return [c for c in df.columns if c not in META]


def make_xgb(n_estimators=4000, quantile=None, early=True, device='cuda', **over):
    import xgboost as xgb
    obj = (dict(objective='reg:absoluteerror') if quantile is None
           else dict(objective='reg:quantileerror', quantile_alpha=quantile))
    params = dict(device=device, tree_method='hist', n_estimators=n_estimators,
                  learning_rate=0.03, max_depth=7, min_child_weight=100,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, max_bin=256,
                  eval_metric='mae', early_stopping_rounds=200 if early else None)
    params.update(obj)
    params.update(over)
    return xgb.XGBRegressor(**params)


def fit_predict(Y, OH, c, stride=2, quantile=None, verbose=False, extra=None, n_trees=None, **xgb_over):
    """Train on targets before week c, forecast the 13 weeks from origin c-1.
    Tree count comes from an inner window [c-13, c) unless n_trees is given.
    extra: optional DataFrame of extra training rows (e.g. Set A) with the same columns.
    Returns raw-unit forecast (n, 13) and the fitted model."""
    first = 12
    tr = build_rows(Y, OH, range(first, c - 1, stride), max_target=c)
    if extra is not None:
        tr = pd.concat([tr, extra[extra['origin'] + extra['h'] < c]], ignore_index=True)
    X = feat_cols(tr)
    w = tr['w'].values if 'w' in tr else None
    if n_trees is None:
        inner = c - H
        m_in = (tr['origin'] + tr['h']) < inner
        va = build_rows(Y, OH, [inner - 1], max_target=c)
        va = va[va['origin'] + va['h'] < c]
        m = make_xgb(quantile=quantile, **xgb_over)
        m.fit(tr.loc[m_in, X], tr.loc[m_in, 'y'],
              sample_weight=None if w is None else w[m_in.values],
              eval_set=[(va[X], va['y'])], verbose=False)
        n_trees = int((m.best_iteration + 1) * 1.1)
        if verbose:
            print(f'  cutoff {c}: inner best trees {m.best_iteration + 1}')
    model = make_xgb(n_estimators=max(n_trees, 50), quantile=quantile, early=False, **xgb_over)
    model.fit(tr[X], tr['y'], sample_weight=w, verbose=False)
    model.set_params(device='cpu')
    te = build_rows(Y[:, :c], OH[:, :c], [c - 1], with_target=False)
    pred = np.clip(model.predict(te[X]), 0, None) * te['scale'].values
    F = np.zeros((Y.shape[0], H))
    F[te['series'].values, te['h'].values.astype(int) - 1] = pred
    return F, model, n_trees


def backtest_model(Y, OH, n_windows=4, **kw):
    """Model forecasts for each rolling window: {cutoff: F}."""
    out = {}
    for c in windows(Y.shape[1], n_windows):
        F, _, nt = fit_predict(Y, OH, c, **kw)
        out[c] = F
        print(f'  cutoff {c}: trees {nt}')
    return out


def score_windows(Y, preds, post=None, verbose=True):
    rows = []
    for c, F in preds.items():
        hist = Y[:, :c]
        G = post(F, hist) if post else F
        s = score(Y[:, c:c + H], np.clip(G, 0, None), mase_denom(hist, floor=0.0))
        s['cutoff_idx'] = c
        rows.append(s)
    out = pd.DataFrame(rows).set_index('cutoff_idx')
    out.loc['mean'] = out.mean()
    if verbose:
        print(out.round(2).to_string())
    return out


# ------------------------------------------------------------- final forecaster
def quantile_ensemble(hist, ks=(13, 26), q=0.45, dead_after=6):
    """Flat forecast = average of the q-quantile over the last k weeks (k in ks).
    q slightly below the median keeps forecasts conservative on slow sellers
    (over-forecasting tiny-denominator articles is what MASE punishes most).
    Articles with no sale in the last `dead_after` weeks get 0."""
    f = np.mean([np.quantile(hist[:, -k:], q, axis=1) for k in ks], axis=0)
    if dead_after is not None:
        f = np.where(weeks_since_sale(hist) >= dead_after, 0.0, f)
    return np.repeat(f[:, None], H, axis=1)


def make_submission(article_ids, F, path):
    F = np.clip(np.nan_to_num(F), 0, None)
    sub = pd.DataFrame(F, columns=[f'w{i}' for i in range(1, H + 1)])
    sub.insert(0, 'article_id', list(article_ids))
    assert len(sub) == 2904 and sub['article_id'].is_unique
    sub.to_csv(path, index=False)
    return sub
