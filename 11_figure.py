# 11_figure.py — final group-effect figure, Ridge vs LightGBM.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, FIG_DIR, TARGETS
from qm9lib.explain import CARBONYL_MERGE, merge_groups, summarize

DROP = {"carbon skeleton", "unassigned", "baseline"}
MIN_MOLECULES = 50
CONSISTENT = 0.95          # |frac_negative - 0.5| threshold for a filled marker


def load_summary(model_tag, target):
    path = DATA_DIR / (f"group_shap_{target}.parquet" if model_tag == "ridge"
                       else f"group_shap_lgbm_{target}.parquet")
    totals = pd.read_parquet(path)
    present = totals != 0
    tm, pm = merge_groups(totals, present, CARBONYL_MERGE)
    s = summarize(tm, pm, MIN_MOLECULES)
    return s[~s["group"].isin(DROP)].set_index("group")


if __name__ == "__main__":
    data = {(m, t): load_summary(m, t) for m in ("ridge", "lgbm") for t in TARGETS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    colours = {"ridge": "#3b6ea5", "lgbm": "#c8553d"}

    for ax, t in zip(axes, TARGETS):
        r, g = data[("ridge", t)], data[("lgbm", t)]
        groups = [x for x in r.sort_values("median").index if x in g.index]
        y = np.arange(len(groups))

        for off, tag, src in ((-0.2, "ridge", r), (0.2, "lgbm", g)):
            s = src.loc[groups]
            ax.barh(y + off, s["median"], height=0.38,
                    xerr=[s["median"] - s["q1"], s["q3"] - s["median"]],
                    color=colours[tag], alpha=0.85, lw=0,
                    error_kw={"lw": 0.9, "capsize": 2, "ecolor": "#555555"},
                    label={"ridge": "Ridge (min1, n=13075)",
                           "lgbm": "LightGBM (min10, n=3000)"}[tag])

            consistent = (s["frac_negative"] >= CONSISTENT) | \
                         (s["frac_negative"] <= 1 - CONSISTENT)
            ax.scatter(s["median"], y + off, s=14, zorder=3,
                       facecolors=np.where(consistent, "white", "none"),
                       edgecolors="#222222", linewidths=0.8)

        ax.axvline(0, color="k", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(groups, fontsize=9)
        ax.set_xlabel(f"Contribution to {t.upper()} (eV)")
        ax.set_title(f"{t.upper()}", fontsize=12, weight="bold")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.invert_yaxis()

    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Functional group contributions to frontier orbital energies\n"
        "Collision-free Morgan fingerprints (r=2), QM9 random-split test set",
        fontsize=12,
    )
    fig.text(0.5, 0.01,
             "Bars: median per-molecule SHAP contribution over molecules containing the group. "
             "Whiskers: interquartile range (spread = context dependence, not uncertainty).\n"
             "Filled marker: effect has the same sign in \u226595% of molecules. "
             "Carbonyl subtypes merged; see Table X for the LUMO subtype breakdown.",
             ha="center", fontsize=7.5, color="#444444")

    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(FIG_DIR / "group_effects.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "group_effects.pdf", bbox_inches="tight")
    print(f"saved -> {FIG_DIR / 'group_effects.png'} (+ .pdf)")

    for t in TARGETS:
        r, g = data[("ridge", t)], data[("lgbm", t)]
        common = r.index.intersection(g.index)
        d = (r.loc[common, "median"] - g.loc[common, "median"]).abs().sort_values()
        print(f"\n{t}: largest Ridge-LightGBM disagreements (eV)")
        print(d.tail(4).to_string(float_format="%.3f"))