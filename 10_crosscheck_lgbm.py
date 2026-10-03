# 10_crosscheck_lgbm.py — does a nonlinear model recover the carbonyl HOMO effect?
import joblib
import pandas as pd

from config import DATA_DIR, MODEL_DIR, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_splits
from qm9lib.explain import CARBONYL_MERGE, lgbm_group_shap, merge_groups, summarize

FEATURES = {"min_count": 10, "max_radius": 2, "binary": False}
N_EVAL = 3000
DROP = {"carbon skeleton", "unassigned", "baseline"}

if __name__ == "__main__":
    (_, _, X_te), _, col_meta = load_cfmf("random", **FEATURES)
    _, _, test_ids = load_splits("random")
    test_ids = test_ids[:N_EVAL]

    for t in TARGETS:
        booster = joblib.load(MODEL_DIR / f"cfmf_lgbm_small_random_{t}.joblib")
        print(f"\n{t}: TreeSHAP on {N_EVAL} molecules...")
        totals, present = lgbm_group_shap(booster, X_te[:N_EVAL], test_ids, col_meta)
        totals.to_parquet(DATA_DIR / f"group_shap_lgbm_{t}.parquet")

        tm, pm = merge_groups(totals, present, CARBONYL_MERGE)
        s = summarize(tm, pm, 50)
        print(f"\n=== {t}: LightGBM group contribution (eV) ===")
        print(s[~s["group"].isin(DROP)].to_string(index=False, float_format="%.3f"))