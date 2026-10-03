# 06a_train.py — final models on both splits, evaluated once on test.
import time

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_params
from qm9lib.descriptors import load_descriptors
from qm9lib.models import evaluate, fit_lgbm, fit_ridge, fit_rf

RF_TREES = 500
LGBM_LR = 0.05
LGBM_FEATURES = {"min_count": 10, "max_radius": 2, "binary": False}

if __name__ == "__main__":
    p_rf = load_params("desc_rf")
    p_ridge = load_params("cfmf_ridge")
    p_lgbm = load_params("cfmf_lgbm")
    rows = []

    for kind in ("random", "scaffold"):
        print(f"\n{'=' * 60}\n{kind} split\n{'=' * 60}")

        desc = load_descriptors(kind)
        ridge_feat = load_cfmf(kind, **p_ridge["features"])
        lgbm_feat = load_cfmf(kind, **LGBM_FEATURES)

        jobs = [
            ("desc_rf", desc,
             lambda Xs, y_tr, y_va, t: fit_rf(Xs[0], y_tr, p_rf[t], RF_TREES)),
            ("cfmf_ridge", ridge_feat,
             lambda Xs, y_tr, y_va, t: fit_ridge(Xs[0], y_tr, p_ridge[t]["alpha"])),
            ("cfmf_lgbm", lgbm_feat,
             lambda Xs, y_tr, y_va, t: fit_lgbm(Xs[0], y_tr, Xs[1], y_va,
                                                p_lgbm[t], LGBM_LR)),
        ]

        for name, (Xs, y, _), fit in jobs:
            for t in TARGETS:
                y_tr, y_va, y_te = y[t]
                t0 = time.time()
                model = fit(Xs, y_tr, y_va, t)
                sec = time.time() - t0

                n_trees = getattr(model, "best_iteration", None)
                metrics = evaluate(y_te, model.predict(Xs[2]), y_tr.mean())
                rows.append({"split": kind, "model": name, "target": t,
                             "n_trees": n_trees, "sec": sec, **metrics})

                joblib.dump(model, MODEL_DIR / f"{name}_{kind}_{t}.joblib")
                print(f"{name:>11} {t}  test MAE={metrics['mae']:.4f}  "
                      f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
                      f"{'' if n_trees is None else f'  trees={n_trees}'}"
                      f"  ({sec:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "results_test.csv", index=False)

    print(f"\n{'=' * 60}\nTEST SET RESULTS\n{'=' * 60}")
    print(res.pivot_table(index=["model", "target"], columns="split",
                          values=["mae", "rmse", "r2"]).round(4).to_string())
    print(f"\nsaved -> {DATA_DIR / 'results_test.csv'} and models in {MODEL_DIR}")