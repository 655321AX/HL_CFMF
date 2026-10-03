# qm9lib/models.py — fitting and evaluation helpers.
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import SEED


def fit_rf(X_tr, y_tr, params, n_estimators=500):
    return RandomForestRegressor(
        n_estimators=n_estimators, random_state=SEED, n_jobs=-1, **params
    ).fit(X_tr, y_tr)


def fit_ridge(X_tr, y_tr, alpha):
    return Ridge(alpha=alpha).fit(X_tr, y_tr)


def fit_lgbm(X_tr, y_tr, X_va, y_va, params, learning_rate=0.05,
             max_rounds=15000, patience=200, log_period=1000):
    fixed = {
        "objective": "regression", "metric": "l1",
        "learning_rate": learning_rate, "bagging_freq": 1,
        "seed": SEED, "verbosity": -1,
    }
    dtrain = lgb.Dataset(X_tr, y_tr, params={"feature_pre_filter": False})
    dval = lgb.Dataset(X_va, y_va, reference=dtrain,
                       params={"feature_pre_filter": False})
    return lgb.train(
        {**fixed, **params}, dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(patience, verbose=False),
                   lgb.log_evaluation(period=log_period)],
    )


def evaluate(y_true, y_pred, train_mean):
    """MAE, RMSE, R2, plus MAE relative to a mean-predictor baseline."""
    mae = mean_absolute_error(y_true, y_pred)
    baseline = float(np.abs(y_true - train_mean).mean())
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "baseline_mae": baseline,
        "frac_of_baseline": float(mae / baseline),
    }