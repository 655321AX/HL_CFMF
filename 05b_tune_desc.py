# 05b_tune_desc.py — random search for descriptor RF, per target, on random-split val.
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler

from config import DATA_DIR, SEED, TARGETS
from qm9lib.data import save_params
from qm9lib.descriptors import load_descriptors

N_ITER = 20
N_TREES_SEARCH = 100
SPACE = {
    "max_features": [0.2, 0.33, 0.5, 0.7, 1.0],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_depth": [None, 15, 25],
    "max_samples": [None, 0.5],
}

if __name__ == "__main__":
    (X_tr, X_va, X_te), y, cols = load_descriptors("random")
    print(f"features: {len(cols)} | train {len(X_tr)} | val {len(X_va)}")
    print(f"NaN cells: train={X_tr.isna().sum().sum()} "
          f"val={X_va.isna().sum().sum()} test={X_te.isna().sum().sum()}\n")

    candidates = list(ParameterSampler(SPACE, n_iter=N_ITER, random_state=SEED))

    rows = []
    for t in TARGETS:
        y_tr, y_va, _ = y[t]
        for i, p in enumerate(candidates):
            t0 = time.time()
            model = RandomForestRegressor(
                n_estimators=N_TREES_SEARCH, random_state=SEED, n_jobs=-1, **p
            ).fit(X_tr, y_tr)
            mae = mean_absolute_error(y_va, model.predict(X_va))
            rows.append({"target": t, "cand": i, **p, "val_mae": mae,
                         "sec": time.time() - t0})
            print(f"{t} [{i+1:>2}/{N_ITER}] {p}  val MAE={mae:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "tune_desc_rf.csv", index=False)

    best = {}
    for t in TARGETS:
        r = res[res["target"] == t].sort_values("val_mae")
        print(f"\n=== {t}: top 5 ===")
        print(r.drop(columns=["target", "cand"]).head(5).to_string(index=False))
        best[t] = candidates[int(r.iloc[0]["cand"])]

    save_params("desc_rf", best)
    print(f"\nsaved best params -> best_params.json: {best}")