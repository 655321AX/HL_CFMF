# 08_groups.py — assign fingerprint columns to functional groups; report coverage.
import pandas as pd

from config import DATA_DIR, TARGETS
from qm9lib.groups import assign_columns

if __name__ == "__main__":
    frags = pd.read_parquet(DATA_DIR / "cfmf_fragments.parquet")
    print(f"assigning {len(frags)} columns...")

    assign = assign_columns(frags)
    assign.to_parquet(DATA_DIR / "cfmf_groups.parquet", index=False)

    print("\n=== columns per group ===")
    print(assign["group"].value_counts().to_string())
    print(f"\nmulti-group columns: {(assign['n_groups_matched'] > 1).sum()}")
    print(f"unassigned: {(assign['group'] == 'unassigned').sum()} "
          f"({(assign['group'] == 'unassigned').mean():.1%})")

    # how much model importance does each group carry?
    for t in TARGETS:
        shap = pd.read_parquet(DATA_DIR / f"shap_ridge_{t}.parquet")
        m = shap.join(assign.set_index("col")[["group"]], on="col")
        agg = (m.groupby("group")
                 .agg(total_importance=("mean_abs_shap", "sum"),
                      n_cols=("col", "count"))
                 .sort_values("total_importance", ascending=False))
        agg["share"] = agg["total_importance"] / agg["total_importance"].sum()
        print(f"\n=== {t}: share of total |SHAP| by group ===")
        print(agg.head(20).to_string(float_format="%.4f"))