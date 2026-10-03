# 05a_screen.py — compare CF-MF configurations with Ridge (random split, val set).
import itertools
import time

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from config import DATA_DIR, TARGETS
from qm9lib.cfmf import load_cfmf

ALPHAS = [0.3, 3, 30]
CONFIGS = list(itertools.product((10, 1), (2, 1), (False, True)))

if __name__ == "__main__":
    rows = []
    for min_count, max_radius, binary in CONFIGS:
        (X_tr, X_va, _), y, _ = load_cfmf("random", min_count, max_radius, binary)
        tag = f"min{min_count} r<={max_radius} {'bin' if binary else 'cnt'}"
        print(f"\n{tag}: {X_tr.shape[1]} columns")

        for t in TARGETS:
            y_tr, y_va, _ = y[t]
            for a in ALPHAS:
                t0 = time.time()
                model = Ridge(alpha=a).fit(X_tr, y_tr)
                mae = mean_absolute_error(y_va, model.predict(X_va))
                rows.append({
                    "min_count": min_count, "max_radius": max_radius,
                    "binary": binary, "n_cols": X_tr.shape[1],
                    "target": t, "alpha": a, "val_mae": mae,
                    "sec": time.time() - t0,
                })
                print(f"  {t} alpha={a:<4} val MAE={mae:.4f} eV  ({time.time()-t0:.1f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "screen_cfmf.csv", index=False)

    cfg = ["min_count", "max_radius", "binary", "target"]
    best = res.loc[res.groupby(cfg)["val_mae"].idxmin()]
    edge = best["alpha"].isin([min(ALPHAS), max(ALPHAS)])

    print("\n=== best alpha per configuration ===")
    print(best.sort_values(["target", "val_mae"]).to_string(index=False))
    if edge.any():
        print(f"\nWARNING: {edge.sum()} configs have best alpha at grid edge "
              f"-> optimum may lie outside {ALPHAS}")