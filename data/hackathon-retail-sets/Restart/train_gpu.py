import numpy as np
import xgboost as xgb
from features import backtest_split, wape

OBJECTIVES = {
    'l1':      dict(objective='reg:absoluteerror'),
    'tweedie': dict(objective='reg:tweedie', tweedie_variance_power=1.3),
}


def make_model(objective='l1', n_estimators=3000, early_stopping=True, device='cuda'):
    return xgb.XGBRegressor(
        device=device,              # RTX 3060; use 'cpu' to compare
        tree_method='hist',
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        max_bin=256,
        eval_metric='mae',
        early_stopping_rounds=100 if early_stopping else None,
        **OBJECTIVES[objective],
    )


def fit_with_inner_stop(train, X, inner_cutoff, objective='l1', device='cuda'):
    """1) find the number of trees on an inner window inside `train`,
       2) refit on all of `train` with ~10% more trees."""
    tr_in, va_in = backtest_split(train, inner_cutoff)
    m = make_model(objective, device=device)
    m.fit(tr_in[X], tr_in.units_scaled,
          sample_weight=tr_in.weight * tr_in.scale,
          eval_set=[(va_in[X], va_in.units_scaled)],
          sample_weight_eval_set=[va_in.scale],
          verbose=200)
    best = m.best_iteration + 1
    print(f"best trees on inner window: {best}")

    final = make_model(objective, n_estimators=int(best * 1.1), early_stopping=False, device=device)
    final.fit(train[X], train.units_scaled, sample_weight=train.weight * train.scale, verbose=False)
    return final


def run_backtest(ds, X, cutoff, inner_cutoff, objective='l1', device='cuda'):
    train, valid = backtest_split(ds, cutoff)
    model = fit_with_inner_stop(train, X, inner_cutoff, objective, device)
    pred = np.clip(model.predict(valid[X]), 0, None) * valid.scale
    score = wape(valid.units, pred)
    naive = wape(valid.units, valid.mean13 * valid.scale)
    print(f"[{objective}] cutoff {cutoff}: model WAPE {score:.4f} | naive {naive:.4f}")
    return model, score