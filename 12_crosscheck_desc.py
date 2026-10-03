# 12_crosscheck_desc.py — independent group check using RDKit fr_* group counts.
import time

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error

from config import DATA_DIR, MODEL_DIR, SEED, TARGETS
from qm9lib.descriptors import load_descriptors
from qm9lib.models import fit_rf

# small enough for TreeSHAP; accuracy reported so the trade is visible
SMALL_RF = {"max_features": 0.5, "min_samples_leaf": 20,
            "max_depth": None, "max_samples": None}
N_TREES = 300
N_EVAL = 3000

# fr_* features grouped into the same categories as the fingerprint analysis
FR_GROUPS = {
    "carbonyl C=O":    ["fr_C_O", "fr_aldehyde", "fr_ketone", "fr_ester",
                        "fr_amide", "fr_priamide", "fr_lactone", "fr_lactam",
                        "fr_Al_COO", "fr_Ar_COO", "fr_imide", "fr_urea",
                        "fr_alkyl_carbamate"],
    "nitrile":         ["fr_nitrile"],
    "imine":           ["fr_Imine", "fr_amidine", "fr_guanido", "fr_oxime"],
    "alcohol OH":      ["fr_Al_OH", "fr_Al_OH_noTert", "fr_HOCCN"],
    "phenol OH":       ["fr_Ar_OH", "fr_phenol"],
    "ether O":         ["fr_ether", "fr_methoxy", "fr_morpholine"],
    "epoxide":         ["fr_epoxide"],
    "primary amine":   ["fr_NH2"],
    "secondary amine": ["fr_NH1"],
    "tertiary amine":  ["fr_NH0", "fr_quatN"],
    "aromatic N":      ["fr_ArN", "fr_Ar_N", "fr_Ar_NH", "fr_aniline",
                        "fr_pyridine", "fr_imidazole", "fr_oxazole",
                        "fr_tetrazole", "fr_piperdine", "fr_piperzine"],
    "furan-type O":    ["fr_furan"],
    "alkyne":          ["fr_term_acetylene"],
    "halide":          ["fr_halogen", "fr_alkyl_halide"],
}

if __name__ == "__main__":
    (X_tr, _, X_te), y, cols = load_descriptors("random")
    X_ev = X_te.iloc[:N_EVAL]

    present = {g: [c for c in feats if c in cols] for g, feats in FR_GROUPS.items()}
    missing = {g: set(f) - set(cols) for g, f in FR_GROUPS.items()}
    print("fr_* features dropped by the descriptor filter:")
    for g, m in missing.items():
        if m:
            print(f"  {g}: {sorted(m)}")

    fr_cols = sorted({c for feats in present.values() for c in feats})
    print(f"\ntraining on {len(fr_cols)} fr_* group counts only")

    rows = []
    for t in TARGETS:
        y_tr, _, y_te = y[t]

        rf = fit_rf(X_tr[fr_cols], y_tr, SMALL_RF, N_TREES)
        mae = mean_absolute_error(y_te, rf.predict(X_te[fr_cols]))
        joblib.dump(rf, MODEL_DIR / f"desc_fr_rf_random_{t}.joblib")
        print(f"{t}: fr_*-only RF test MAE={mae:.4f} eV "
              f"(interpretability probe, not a competitive model)")

        sv = shap.TreeExplainer(rf).shap_values(X_ev[fr_cols])
        shap_df = pd.DataFrame(sv, columns=fr_cols, index=X_ev.index)

        for g, feats in present.items():
            if not feats:
                continue
            total = shap_df[feats].sum(axis=1)
            has = (X_ev[feats].sum(axis=1) > 0).to_numpy()
            v = total[has].to_numpy()
            if len(v) < 50:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rows.append({"target": t, "group": g, "n_molecules": len(v),
                         "median": med, "q1": q1, "q3": q3, "iqr": q3 - q1,
                         "frac_negative": float((v < 0).mean())})

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "crosscheck_desc.csv", index=False)

    for t in TARGETS:
        print(f"\n=== {t}: fr_* descriptor group contribution (eV) ===")
        print(res[res.target == t].drop(columns="target")
                 .sort_values("median")
                 .to_string(index=False, float_format="%.3f"))