# 09_group_effects.py — per-molecule group contributions + the final figure.
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, FIG_DIR, MODEL_DIR, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_params
from qm9lib.explain import ridge_group_shap, summarize
from qm9lib.data import load_params, load_splits
from qm9lib.explain import per_molecule_group_shap, summarize
from qm9lib.explain import (CARBONYL_MERGE, merge_groups,
                            per_molecule_group_shap, summarize)

DROP = {"carbon skeleton", "unassigned", "baseline"}
MIN_MOLECULES = 50

if __name__ == "__main__":
    feat = load_params("cfmf_ridge")["features"]
    (X_tr, _, X_te), _, col_meta = load_cfmf("random", **feat)
    _, _, test_ids = load_splits("random")

    summaries, split_summaries = {}, {}
    for t in TARGETS:
        model = joblib.load(MODEL_DIR / f"cfmf_ridge_random_{t}.joblib")
        totals, present = per_molecule_group_shap(
            model, X_tr, X_te, test_ids, col_meta
        )
        totals.to_parquet(DATA_DIR / f"group_shap_{t}.parquet")

        # subtype-resolved (reliable for LUMO only)
        s_split = summarize(totals, present, MIN_MOLECULES)
        split_summaries[t] = s_split[~s_split["group"].isin(DROP)]

        # carbonyls merged — the trustworthy view for both targets
        tm, pm = merge_groups(totals, present, CARBONYL_MERGE)
        s = summarize(tm, pm, MIN_MOLECULES)
        summaries[t] = s[~s["group"].isin(DROP)]

        print(f"\n=== {t}: group contribution, carbonyls merged (eV) ===")
        print(summaries[t].to_string(index=False, float_format="%.3f"))

    print("\n=== LUMO: carbonyl subtypes (stable; reported separately) ===")
    sub = split_summaries["lumo"]
    print(sub[sub["group"].isin(CARBONYL_MERGE)]
          .to_string(index=False, float_format="%.3f"))

    # figure: one panel per target, groups ordered by HOMO median
    order = summaries["homo"]["group"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

    for ax, t, colour in zip(axes, TARGETS, ("#1f77b4", "#d62728")):
        s = summaries[t].set_index("group").loc[
            [g for g in order if g in summaries[t]["group"].values]
        ]
        ypos = np.arange(len(s))
        ax.barh(ypos, s["median"],
                xerr=[s["median"] - s["q1"], s["q3"] - s["median"]],
                color=colour, alpha=0.8, error_kw={"lw": 1, "capsize": 3})
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(s.index)
        ax.set_xlabel(f"{t.upper()} shift (eV)")
        ax.set_title(f"{t.upper()}: median contribution per molecule")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Functional group contributions to frontier orbital energies\n"
                 "CF-MF Ridge, QM9 test set; bars = median, whiskers = IQR",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "group_effects.png", dpi=200, bbox_inches="tight")
    print(f"\nsaved -> {FIG_DIR / 'group_effects.png'}")