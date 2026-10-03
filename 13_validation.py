# 13_validation.py — stability, removal and sufficiency tests for group attributions.
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from config import DATA_DIR, MODEL_DIR, SEED, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_params, load_splits
from qm9lib.explain import (CARBONYL_MERGE, merge_groups,
                            per_molecule_group_shap, summarize)
from qm9lib.groups import assign_columns
from qm9lib.fragments import describe_columns

N_BOOTSTRAP = 10
N_EVAL = 3000
DROP = {"carbon skeleton", "unassigned", "baseline"}
MIN_MOLECULES = 50


def column_groups(totals_cols, assign, n_cols):
    """Map each fingerprint column to its merged group name."""
    g = assign.set_index("col")["group"].map(lambda x: CARBONYL_MERGE.get(x, x))
    out = pd.Series("unassigned", index=np.arange(n_cols))
    out.loc[g.index] = g.to_numpy()
    return out


if __name__ == "__main__":
    feat = load_params("cfmf_ridge")["features"]
    alphas = {t: load_params("cfmf_ridge")[t]["alpha"] for t in TARGETS}
    (X_tr, _, X_te), y, col_meta = load_cfmf("random", **feat)
    train_ids, _, test_ids = load_splits("random")
    assign = pd.read_parquet(DATA_DIR / "cfmf_groups.parquet")
    col_group = column_groups(None, assign, X_te.shape[1])

    # ---------- 1. stability across bootstrap refits ----------
    print("=== stability: group medians across bootstrap refits ===")
    rng = np.random.default_rng(SEED)
    eval_ids = test_ids[:N_EVAL]
    X_ev = X_te[:N_EVAL]

    for t in TARGETS:
        y_tr = y[t][0]
        medians = []
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, X_tr.shape[0], X_tr.shape[0])
            model = Ridge(alpha=alphas[t]).fit(X_tr[idx], y_tr[idx])
            totals, present = per_molecule_group_shap(
                model, X_tr[idx], X_ev, eval_ids, col_meta
            )
            tm, pm = merge_groups(totals, present, CARBONYL_MERGE)
            s = summarize(tm, pm, MIN_MOLECULES).set_index("group")["median"]
            medians.append(s)
            print(f"  {t} bootstrap {b+1}/{N_BOOTSTRAP}")

        M = pd.DataFrame(medians)
        stab = pd.DataFrame({"mean": M.mean(), "sd": M.std(),
                             "cv": (M.std() / M.mean().abs()).replace(np.inf, np.nan)})
        stab = stab.drop(index=[g for g in DROP if g in stab.index])
        print(f"\n{t}: median contribution across {N_BOOTSTRAP} refits (eV)")
        print(stab.sort_values("mean").to_string(float_format="%.3f"))

    # ---------- 2 & 3. removal and sufficiency on LightGBM ----------
    lg_feat = {"min_count": 10, "max_radius": 2, "binary": False}
    (_, _, Xl_te), yl, lg_meta = load_cfmf("random", **lg_feat)
    lg_frags = describe_columns(lg_meta)
    lg_assign = assign_columns(lg_frags)
    lg_group = column_groups(None, lg_assign, Xl_te.shape[1])
    Xl = Xl_te[:N_EVAL].tocsc()

    for t in TARGETS:
        booster = joblib.load(MODEL_DIR / f"cfmf_lgbm_small_random_{t}.joblib")
        base_pred = booster.predict(Xl)
        shap_tot = pd.read_parquet(DATA_DIR / f"group_shap_lgbm_{t}.parquet")
        shap_tot, _ = merge_groups(shap_tot, shap_tot != 0, CARBONYL_MERGE)

        rows = []
        for g in shap_tot.columns:
            if g in DROP:
                continue
            cols = np.flatnonzero((lg_group == g).to_numpy())
            if len(cols) == 0:
                continue
            # presence and SHAP median from the SAME per-molecule assignment
            g_shap = shap_tot[g].to_numpy()[:len(base_pred)]
            has = g_shap != 0
            if has.sum() < MIN_MOLECULES:
                continue

            # removal: zero this group's columns
            Xr = Xl.copy()
            for c in cols:
                Xr.data[Xr.indptr[c]:Xr.indptr[c + 1]] = 0
            Xr.eliminate_zeros()
            delta_removal = (booster.predict(Xr) - base_pred)[has]

            # sufficiency: keep ONLY this group's columns
            Xs = sparse.csc_matrix(Xl.shape, dtype=Xl.dtype)
            Xs = Xs.tolil()
            Xs[:, cols] = Xl[:, cols]
            dev_full = (base_pred - base_pred.mean())[has]
            dev_only = (booster.predict(Xs.tocsr()) - base_pred.mean())[has]
            suff = np.corrcoef(dev_full, dev_only)[0, 1]

            rows.append({
                "group": g,
                "n": int(has.sum()),
                "shap_median": float(np.median(g_shap[has])),
                "removal_median": float(np.median(-delta_removal)),
                "sufficiency_r": float(suff),
            })

        res = pd.DataFrame(rows).sort_values("shap_median")
        res["agreement"] = res["removal_median"] / res["shap_median"]
        print(f"\n=== {t}: removal and sufficiency (LightGBM) ===")
        print(res.to_string(index=False, float_format="%.3f"))
        res.to_csv(DATA_DIR / f"validation_{t}.csv", index=False)