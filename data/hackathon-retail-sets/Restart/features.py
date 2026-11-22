import pandas as pd
import numpy as np

WEEK = pd.Timedelta(days=7)
HORIZON = 13


# ---------------------------------------------------------------- 1. panel prep
def fill_gaps(combined):
    """One row per series per week between its first and last week.
    Missing weeks get units = 0 and was_missing = 1."""
    keep = ['source', 'store_id', 'article_id', 'scale', 'is_test', 'weight']
    meta = combined.groupby('series_id')[keep].first()
    span = combined.groupby('series_id')['week_start'].agg(['min', 'max'])
    grid = pd.DataFrame({
        'series_id': span.index,
        'week_start': [pd.date_range(a, b, freq='W-MON') for a, b in zip(span['min'], span['max'])],
    }).explode('week_start')
    grid['week_start'] = pd.to_datetime(grid['week_start'])

    p = grid.merge(combined[['series_id', 'week_start', 'units', 'on_hand', 'in_stock']],
                   on=['series_id', 'week_start'], how='left')
    p['was_missing'] = p['units'].isna().astype(np.float32)
    p['units'] = p['units'].fillna(0.0)
    p = p.join(meta, on='series_id')
    p['units_scaled'] = p['units'] / p['scale']
    p['on_hand_scaled'] = p['on_hand'] / p['scale']
    return p.sort_values(['series_id', 'week_start']).reset_index(drop=True)


def add_future_rows(p, n=HORIZON):
    """Append n empty weeks after the last week of every TEST series (the weeks to predict)."""
    last = p[p.is_test == 1].groupby('series_id').tail(1)
    fut = []
    for k in range(1, n + 1):
        f = last.copy()
        f['week_start'] = f['week_start'] + k * WEEK
        f[['units', 'units_scaled', 'on_hand', 'on_hand_scaled', 'in_stock']] = np.nan
        f['was_missing'] = 0.0
        fut.append(f)
    out = pd.concat([p] + fut, ignore_index=True)
    out['is_future'] = out['units'].isna().astype(int)
    return out.sort_values(['series_id', 'week_start']).reset_index(drop=True)


# ------------------------------------------------- 2. features at the forecast start
def origin_features(p):
    """Features describing a series AS OF week o (using only data up to and including o)."""
    g = p.groupby('series_id', sort=False)
    y = g['units_scaled']
    f = pd.DataFrame(index=p.index)

    for k in range(4):
        f[f'lag{k}'] = y.shift(k)
    for w in (4, 8, 13, 26, 52):
        f[f'mean{w}'] = y.transform(lambda s: s.rolling(w, min_periods=1).mean())
    f['std13'] = y.transform(lambda s: s.rolling(13, min_periods=2).std())
    f['max13'] = y.transform(lambda s: s.rolling(13, min_periods=1).max())
    f['ewm4'] = y.transform(lambda s: s.ewm(span=4).mean())
    f['zero_share13'] = g['units'].transform(lambda s: (s == 0).rolling(13, min_periods=1).mean())
    f['trend'] = f['mean4'] / (f['mean26'] + 1e-3)

    f['in_stock0'] = p['in_stock']
    f['in_stock_mean4'] = g['in_stock'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    f['stockout_share13'] = g['in_stock'].transform(lambda s: (s < 1).rolling(13, min_periods=1).mean())
    f['on_hand0'] = p['on_hand_scaled']
    f['missing_share13'] = g['was_missing'].transform(lambda s: s.rolling(13, min_periods=1).mean())
    f['age'] = g.cumcount()
    return f


# ------------------------------------------- 3. stack one copy per horizon h = 1..13
def build_dataset(p, horizons=range(1, HORIZON + 1), a_frac=1.0, seed=0):
    """Each output row = (series, target week t, horizon h); features come from origin t - h."""
    if a_frac < 1.0:
        a_ids = p.loc[p.is_test == 0, 'series_id'].unique()
        keep = np.random.default_rng(seed).choice(a_ids, int(len(a_ids) * a_frac), replace=False)
        p = p[(p.is_test == 1) | p.series_id.isin(keep)].reset_index(drop=True)

    F = origin_features(p).astype(np.float32)
    gs = p['series_id']
    base = p[['series_id', 'week_start', 'units', 'units_scaled', 'scale', 'is_test', 'weight']].copy()
    if 'is_future' in p:
        base['is_future'] = p['is_future']
    # target-week features (known in advance)
    base['woy'] = p['week_start'].dt.isocalendar().week.astype(np.int16)
    base['month'] = p['week_start'].dt.month.astype(np.int8)
    base['lag52_target'] = p.groupby('series_id')['units_scaled'].shift(52).astype(np.float32)
    base['log_scale'] = np.log1p(p['scale']).astype(np.float32)

    parts = []
    for h in horizons:
        part = pd.concat([base, F.groupby(gs).shift(h)], axis=1)
        part['h'] = np.int8(h)
        part['origin'] = part['week_start'] - h * WEEK
        parts.append(part[part['age'].notna()])          # need an origin inside the series
    return pd.concat(parts, ignore_index=True)


def feature_cols(ds):
    drop = {'series_id', 'week_start', 'origin', 'units', 'units_scaled', 'scale',
            'weight', 'is_future'}
    return [c for c in ds.columns if c not in drop]


# ------------------------------------------------------------- 4. backtest split
def backtest_split(ds, cutoff, n=HORIZON):
    """Train: every row whose target week is before the cutoff.
    Valid: test series, forecast made at the last week before the cutoff, next n weeks."""
    cutoff = pd.Timestamp(cutoff)
    train = ds[ds.week_start < cutoff]
    valid = ds[(ds.is_test == 1) & (ds.origin == cutoff - WEEK) &
               (ds.week_start >= cutoff) & (ds.week_start < cutoff + n * WEEK)]
    return train, valid


def wape(actual, pred):
    return np.abs(actual - pred).sum() / np.abs(actual).sum()