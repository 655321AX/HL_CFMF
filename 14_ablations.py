# 14_ablations.py — what actually improved the model?
import time

import pandas as pd
from sklearn.metrics import mean_absolute_error

from config import DATA_DIR, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_params
from qm9lib.descriptors import load_descriptors
from qm9lib.models import fit_lgbm, fit_rf

RF_TREES = 300
LGBM_SMALL = {"num_leaves": 63, "min_child_samples": 20,
              "colsample_bytree": 0.7, "subsample": 0.85, "reg_lambda": 1}

OLD20 = [
    "ExactMolWt", "MolLogP", "TPSA", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "NumAliphaticRings", "NumHAcceptors", "NumHDonors",
    "FractionCSP3", "BertzCT", "LabuteASA", "Chi0n", "Chi1n", "Kappa1",
    "Kappa2", "NumHeteroatoms", "NumSaturatedRings", "MolMR", "HeavyAtomCount",
]
ELECTRONIC = ("PEOE_VSA", "EState_VSA", "VSA_EState", "PartialCharge",
              "BCUT2D_CHG", "EStateIndex")

if __name__ == "__main__":
    rows = []

    # ---------- A: descriptor set ----------
    (X_tr, _, X_te), y, cols = load_descriptors("random")
    p_rf = load_params("desc_rf")

    sets = {
        "old20":         [c for c in OLD20 if c in cols],
        "no_electronic": [c for c in cols if not any(k in c for k in ELECTRONIC)],
        "no_fr":         [c for c in cols if not c.startswith("fr_")],
        "full162":       list(cols),
    }
    missing = [c for c in OLD20 if c not in cols]
    if missing:
        print(f"note: {len(missing)} of the old 20 were removed by the filter: {missing}")

    for name, use in sets.items():
        for t in TARGETS:
            y_tr, _, y_te = y[t]
            t0 = time.time()
            rf = fit_rf(X_tr[use], y_tr, p_rf[t], RF_TREES)
            mae = mean_absolute_error(y_te, rf.predict(X_te[use]))
            rows.append({"ablation": "descriptors", "variant": name,
                         "n_features": len(use), "target": t,
                         "test_mae": mae, "sec": time.time() - t0})
            print(f"desc {name:>14} {t}  n={len(use):>3}  "
                  f"MAE={mae:.4f}  ({time.time()-t0:.0f}s)")

    # ---------- B: fingerprint vocabulary ----------
    for min_count in (10, 1):
        (Xf_tr, Xf_va, Xf_te), yf, _ = load_cfmf(
            "random", min_count=min_count, max_radius=2, binary=False
        )
        for t in TARGETS:
            y_tr, y_va, y_te = yf[t]
            t0 = time.time()
            b = fit_lgbm(Xf_tr, y_tr, Xf_va, y_va, LGBM_SMALL,
                         learning_rate=0.1, max_rounds=6000,
                         patience=100, log_period=3000)
            mae = mean_absolute_error(y_te, b.predict(Xf_te))
            rows.append({"ablation": "fingerprint", "variant": f"min{min_count}",
                         "n_features": Xf_tr.shape[1], "target": t,
                         "test_mae": mae, "n_trees": b.num_trees(),
                         "sec": time.time() - t0})
            print(f"cfmf min{min_count:<3} {t}  n={Xf_tr.shape[1]:>6}  "
                  f"MAE={mae:.4f}  trees={b.num_trees()}  ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "ablations.csv", index=False)

    print("\n=== test MAE (eV) ===")
    print(res.pivot_table(index=["ablation", "variant", "n_features"],
                          columns="target", values="test_mae")
             .round(4).to_string())