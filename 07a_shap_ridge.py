# 07a_shap_ridge.py — exact Ridge SHAP on CF-MF, ranked substructure table.
import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_params
from qm9lib.fragments import describe_columns

MIN_SUPPORT = 30


def ridge_shap_summary(model, X_bg, X_eval):
    """Exact Ridge SHAP summaries without densifying anything.

    phi_j = w_j * (x_j - mean_j), so mean|phi_j| = |w_j| * mean|x_j - mean_j|.
    """
    w = model.coef_
    xbar = np.asarray(X_bg.mean(axis=0)).ravel()
    n = X_eval.shape[0]

    coo = X_eval.tocoo()
    dev_nonzero = np.bincount(
        coo.col, weights=np.abs(coo.data - xbar[coo.col]), minlength=len(w)
    )
    nnz = np.diff(X_eval.tocsc().indptr)          # molecules containing each column
    mean_abs_dev = (dev_nonzero + (n - nnz) * xbar) / n

    return pd.DataFrame({
        "weight": w,                               # eV per occurrence
        "mean_abs_shap": np.abs(w) * mean_abs_dev, # dataset importance
        "n_molecules": nnz,                        # support in the eval set
    })


if __name__ == "__main__":
    feat = load_params("cfmf_ridge")["features"]
    (X_tr, _, X_te), _, col_meta = load_cfmf("random", **feat)
    print(f"columns: {X_te.shape[1]} | eval molecules: {X_te.shape[0]}")

    print("extracting fragments...")
    frags = describe_columns(col_meta)
    frags.to_parquet(DATA_DIR / "cfmf_fragments.parquet", index=False)
    print(f"  {frags['fragment'].isna().sum()} columns without a fragment string\n")

    out = {}
    for t in TARGETS:
        model = joblib.load(MODEL_DIR / f"cfmf_ridge_random_{t}.joblib")
        summary = ridge_shap_summary(model, X_tr, X_te)
        tab = frags.join(summary)
        tab.to_parquet(DATA_DIR / f"shap_ridge_{t}.parquet", index=False)
        out[t] = tab

        keep = tab[tab["n_molecules"] >= MIN_SUPPORT]
        for r in sorted(keep["radius"].unique()):
            sub = keep[keep["radius"] == r].nlargest(15, "mean_abs_shap")
            print(f"=== {t}: radius {r}, top 15 (support >= {MIN_SUPPORT}) ===")
            print(sub[["fragment", "weight", "mean_abs_shap", "n_molecules"]]
                  .to_string(index=False, float_format="%.4f"))
            print()

    for t in TARGETS:
        s = out[t][out[t]["n_molecules"] >= MIN_SUPPORT]
        print(f"{t}: of {len(s)} columns with support >= {MIN_SUPPORT}, "
              f"{(s.weight > 0).sum()} raise, {(s.weight < 0).sum()} lower")