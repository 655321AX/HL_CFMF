# 05a_followup.py — extended alpha grid + min1 vs min10 on random and scaffold splits.
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from config import DATA_DIR, TARGETS
from qm9lib.cfmf import load_cfmf

ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3]

if __name__ == "__main__":
    rows = []
    for kind in ("random", "scaffold"):
        for min_count in (1, 10):
            (X_tr, X_va, _), y, _ = load_cfmf(kind, min_count, max_radius=2, binary=False)
            print(f"\n{kind} min{min_count}: {X_tr.shape[1]} columns")
            for t in TARGETS:
                y_tr, y_va, _ = y[t]
                for a in ALPHAS:
                    model = Ridge(alpha=a).fit(X_tr, y_tr)
                    mae = mean_absolute_error(y_va, model.predict(X_va))
                    rows.append({"split": kind, "min_count": min_count,
                                 "target": t, "alpha": a, "val_mae": mae})
                    print(f"  {t} alpha={a:<5} val MAE={mae:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "screen_cfmf_followup.csv", index=False)

    best = res.loc[res.groupby(["split", "min_count", "target"])["val_mae"].idxmin()]
    table = best.pivot_table(index=["split", "target"], columns="min_count",
                             values=["val_mae", "alpha"])
    print("\n=== best per split / target ===")
    print(table.round(4).to_string())