# 15_shap_desc.py — SHAP on the descriptor RF: which descriptors drive HOMO/LUMO?
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error

from config import DATA_DIR, FIG_DIR, MODEL_DIR, TARGETS
from qm9lib.descriptors import load_descriptors
from qm9lib.models import fit_rf

INTERP_RF = {"max_features": 0.5, "min_samples_leaf": 10,
             "max_depth": None, "max_samples": None}
N_TREES = 300
N_EVAL = 2000
TOP_N = 20

FAMILIES = {
    "PEOE_VSA": "charge-weighted surface (PEOE)",
    "EState_VSA": "electrotopological surface",
    "VSA_EState": "electrotopological surface",
    "SMR_VSA": "refractivity-weighted surface",
    "SlogP_VSA": "logP-weighted surface",
    "BCUT2D": "BCUT eigenvalue",
    "fr_": "functional group count",
    "Chi": "connectivity index",
    "Kappa": "shape index",
}


def family_of(name):
    for key, label in FAMILIES.items():
        if name.startswith(key) or key in name:
            return label
    return "other"


if __name__ == "__main__":
    (X_tr, _, X_te), y, cols = load_descriptors("random")
    X_ev = X_te.iloc[:N_EVAL]
    results = {}

    for t in TARGETS:
        y_tr, _, y_te = y[t]

        t0 = time.time()
        rf = fit_rf(X_tr, y_tr, INTERP_RF, N_TREES)
        mae = mean_absolute_error(y_te, rf.predict(X_te))
        full = joblib.load(MODEL_DIR / f"desc_rf_random_{t}.joblib")
        mae_full = mean_absolute_error(y_te, full.predict(X_te))
        leaves = int(np.mean([e.get_n_leaves() for e in rf.estimators_]))
        joblib.dump(rf, MODEL_DIR / f"desc_rf_interp_random_{t}.joblib")
        print(f"\n{t}: interp RF MAE={mae:.4f} vs full {mae_full:.4f} "
              f"({mae/mae_full - 1:+.1%}), {leaves} leaves/tree, {time.time()-t0:.0f}s")

        t0 = time.time()
        sv = shap.TreeExplainer(rf).shap_values(X_ev)
        print(f"  TreeSHAP on {N_EVAL} molecules: {time.time()-t0:.0f}s")

        shap_df = pd.DataFrame(sv, columns=cols, index=X_ev.index)
        shap_df.to_parquet(DATA_DIR / f"shap_desc_{t}.parquet")

        summary = pd.DataFrame({
            "mean_abs_shap": shap_df.abs().mean(),
            "mean_shap": shap_df.mean(),
            "family": [family_of(c) for c in cols],
            # sign of the feature-value / SHAP relationship
            "direction": [np.corrcoef(X_ev[c], shap_df[c])[0, 1]
                          if X_ev[c].std() > 0 and shap_df[c].std() > 0 else np.nan
                          for c in cols],
        }).sort_values("mean_abs_shap", ascending=False)
        summary.to_csv(DATA_DIR / f"shap_desc_summary_{t}.csv")
        results[t] = summary

        print(f"\n=== {t}: top {TOP_N} descriptors ===")
        print(summary.head(TOP_N).to_string(float_format="%.4f"))

        print(f"\n=== {t}: importance by descriptor family ===")
        fam = (summary.groupby("family")["mean_abs_shap"]
                      .agg(["sum", "count"]).sort_values("sum", ascending=False))
        fam["share"] = fam["sum"] / fam["sum"].sum()
        print(fam.to_string(float_format="%.4f"))

    # beeswarm: distribution and direction per descriptor
    for t in TARGETS:
        sv = pd.read_parquet(DATA_DIR / f"shap_desc_{t}.parquet")
        plt.figure(figsize=(9, 8))
        shap.summary_plot(sv.to_numpy(), X_ev, feature_names=list(cols),
                          max_display=TOP_N, show=False, plot_size=None)
        plt.title(f"{t.upper()}: descriptor SHAP (RF, {N_EVAL} test molecules)",
                  fontsize=12, pad=12)
        plt.xlabel(f"SHAP value — contribution to {t.upper()} (eV)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_desc_{t}.png", dpi=250, bbox_inches="tight")
        plt.close()
        print(f"saved -> {FIG_DIR / f'shap_desc_{t}.png'}")