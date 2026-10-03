# 05c_tune_cfmf.py — random search for LightGBM on CF-MF, per target,
# early stopping on random-split val.
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler

from config import DATA_DIR, SEED, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import save_params

FEATURES = {"min_count": 10, "max_radius": 2, "binary": False}
N_ITER = 20
LEARNING_RATE = 0.1
MAX_ROUNDS = 5000
PATIENCE = 100

SPACE = {
    "num_leaves":        [31, 63, 127, 255],
    "min_child_samples": [5, 10, 20, 50],
    "colsample_bytree":  [0.3, 0.5, 0.7, 1.0],
    "subsample":         [0.7, 0.85, 1.0],
    "reg_lambda":        [0, 1, 5, 10],
}
FIXED = {
    "objective": "regression",   # L2 loss for training
    "metric": "l1",              # MAE for early stopping
    "learning_rate": LEARNING_RATE,
    "bagging_freq": 1,           # required for subsample to take effect
    "seed": SEED,
    "verbosity": -1,
}

if __name__ == "__main__":
    (X_tr, X_va, _), y, _ = load_cfmf("random", **FEATURES)
    print(f"train {X_tr.shape} | val {X_va.shape}\n")

    candidates = list(ParameterSampler(SPACE, n_iter=N_ITER, random_state=SEED))

    rows = []
    for t in TARGETS:
        y_tr, y_va, _ = y[t]

        t0 = time.time()
        dtrain = lgb.Dataset(X_tr, y_tr, params={"feature_pre_filter": False},
                             free_raw_data=False)
        dval = lgb.Dataset(X_va, y_va, reference=dtrain)
        dtrain.construct()
        print(f"{t}: dataset built in {time.time() - t0:.0f}s")

        for i, p in enumerate(candidates):
            t0 = time.time()
            booster = lgb.train(
                {**FIXED, **p}, dtrain,
                num_boost_round=MAX_ROUNDS,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(PATIENCE, verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )
            n_trees = booster.best_iteration
            mae = mean_absolute_error(y_va, booster.predict(X_va, num_iteration=n_trees))
            rows.append({"target": t, "cand": i, **p, "n_trees": n_trees,
                         "val_mae": mae, "sec": time.time() - t0})
            cap = "  <-- hit MAX_ROUNDS" if n_trees >= MAX_ROUNDS - PATIENCE else ""
            print(f"  [{i+1:>2}/{N_ITER}] {p}  trees={n_trees:<4} "
                  f"val MAE={mae:.4f}  ({time.time() - t0:.0f}s){cap}")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "tune_cfmf_lgbm.csv", index=False)

    best = {"features": FEATURES, "search_learning_rate": LEARNING_RATE}
    for t in TARGETS:
        r = res[res["target"] == t].sort_values("val_mae")
        print(f"\n=== {t}: top 5 ===")
        print(r.drop(columns=["target", "cand"]).head(5).to_string(index=False))
        best[t] = candidates[int(r.iloc[0]["cand"])]

    save_params("cfmf_lgbm", best)
    print(f"\nsaved -> best_params.json['cfmf_lgbm']")
    print("reference val MAE (eV): Ridge CF-MF 0.120 / 0.144 | RF descriptors 0.121 / 0.128")