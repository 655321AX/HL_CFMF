# 16_outputs.py — all parity figures and consolidated result tables.
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_DIR, FIG_DIR, MODEL_DIR, TARGETS
from qm9lib.cfmf import load_cfmf
from qm9lib.data import load_params
from qm9lib.descriptors import load_descriptors

MODELS = [
    ("desc_rf", "RF — descriptors"),
    ("cfmf_ridge", "Ridge — CF-MF"),
    ("cfmf_lgbm", "LightGBM — CF-MF"),
]
SPLITS = ["random", "scaffold"]
KDE_SAMPLE = 4000          # KDE is O(n^2); subsample for speed


def collect_predictions():
    """Predicted vs true on the test set for every model, split and target."""
    out = {}
    ridge_feat = load_params("cfmf_ridge")["features"]
    lgbm_feat = {"min_count": 10, "max_radius": 2, "binary": False}

    for split in SPLITS:
        feats = {
            "desc_rf": load_descriptors(split),
            "cfmf_ridge": load_cfmf(split, **ridge_feat),
            "cfmf_lgbm": load_cfmf(split, **lgbm_feat),
        }
        for name, _ in MODELS:
            (_, _, X_te), y, _ = feats[name]
            for t in TARGETS:
                model = joblib.load(MODEL_DIR / f"{name}_{split}_{t}.joblib")
                out[(name, split, t)] = (y[t][2], model.predict(X_te))
        print(f"  predictions done: {split}")
    return out


def metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": r2_score(true, pred),
    }


def parity_figure(preds, split, style):
    """One figure: rows = targets, columns = models."""
    fig, axes = plt.subplots(len(TARGETS), len(MODELS),
                             figsize=(4.3 * len(MODELS), 4.1 * len(TARGETS)))

    for i, t in enumerate(TARGETS):
        lo = min(preds[(n, split, t)][0].min() for n, _ in MODELS)
        hi = max(preds[(n, split, t)][0].max() for n, _ in MODELS)
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        for j, (name, label) in enumerate(MODELS):
            ax = axes[i, j]
            true, pred = preds[(name, split, t)]
            m = metrics(true, pred)

            if style == "scatter":
                ax.scatter(true, pred, s=3, alpha=0.08, lw=0, color="#2b6cb0",
                           rasterized=True)
            elif style == "hexbin":
                hb = ax.hexbin(true, pred, gridsize=70, extent=(lo, hi, lo, hi),
                               cmap="viridis", norm=LogNorm(), mincnt=1,
                               linewidths=0)
                if j == len(MODELS) - 1:
                    fig.colorbar(hb, ax=ax, label="molecules", pad=0.02)
            elif style == "density":
                k = min(KDE_SAMPLE, len(true))
                sel = np.random.default_rng(0).choice(len(true), k, replace=False)
                xy = np.vstack([true[sel], pred[sel]])
                dens = gaussian_kde(xy)
                gx, gy = np.mgrid[lo:hi:120j, lo:hi:120j]
                z = dens(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                cs = ax.contourf(gx, gy, z, levels=14, cmap="magma")
                ax.contour(gx, gy, z, levels=6, colors="white",
                           linewidths=0.4, alpha=0.5)
                if j == len(MODELS) - 1:
                    fig.colorbar(cs, ax=ax, label="density", pad=0.02)

            ax.plot([lo, hi], [lo, hi], "--", lw=1.1, color="#c0392b", zorder=4)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.text(0.04, 0.96,
                    f"MAE {m['MAE']:.3f}\nRMSE {m['RMSE']:.3f}\nR² {m['R2']:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(fc="white", ec="#bbbbbb", alpha=0.85, pad=3))
            if i == 0:
                ax.set_title(label, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"Predicted {t.upper()} (eV)")
            ax.set_xlabel(f"DFT {t.upper()} (eV)")
            ax.grid(alpha=0.2, lw=0.4)
            ax.set_axisbelow(True)

    fig.suptitle(f"Parity plots — QM9 test set, {split} split", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = FIG_DIR / f"parity_{style}_{split}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def error_figure(preds, split):
    """Signed error distributions, all models overlaid per target."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(11, 4.2))
    for ax, t in zip(axes, TARGETS):
        for (name, label), c in zip(MODELS, ("#3b6ea5", "#c8553d", "#4f8a53")):
            true, pred = preds[(name, split, t)]
            ax.hist(pred - true, bins=90, histtype="step", lw=1.4,
                    color=c, label=label, density=True)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Predicted − DFT {t.upper()} (eV)")
        ax.set_ylabel("density")
        ax.set_title(t.upper())
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Prediction error distributions — {split} split", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / f"errors_{split}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved errors_{split}.png")


def write(df, name, caption, float_fmt="%.3f"):
    df.to_csv(DATA_DIR / f"table_{name}.csv")
    with open(DATA_DIR / f"table_{name}.tex", "w") as f:
        f.write(df.to_latex(float_format=float_fmt, caption=caption,
                            label=f"tab:{name}", escape=True))
    print(f"  table_{name}.csv / .tex")


if __name__ == "__main__":
    print("collecting predictions...")
    preds = collect_predictions()

    print("\nfigures...")
    for split in SPLITS:
        for style in ("scatter", "hexbin", "density"):
            parity_figure(preds, split, style)
        error_figure(preds, split)

    print("\ntables...")

    # --- Table 1: model comparison ---
    rows = []
    for (name, label) in MODELS:
        for split in SPLITS:
            for t in TARGETS:
                true, pred = preds[(name, split, t)]
                rows.append({"Model": label, "Split": split, "Target": t.upper(),
                             **metrics(true, pred)})
    t1 = (pd.DataFrame(rows)
            .pivot_table(index=["Model", "Target"], columns="Split",
                         values=["MAE", "RMSE", "R2"]))
    write(t1, "models", "Test-set performance on random and scaffold splits.", "%.4f")
    print(t1.round(4).to_string())

    # --- Table 2: group effects, three methods ---
    try:
        parts = []
        for tag, path in [("Ridge CF-MF", "group_shap_{t}.parquet"),
                          ("LightGBM CF-MF", "group_shap_lgbm_{t}.parquet")]:
            from qm9lib.explain import CARBONYL_MERGE, merge_groups, summarize
            for t in TARGETS:
                tot = pd.read_parquet(DATA_DIR / path.format(t=t))
                tm, pm = merge_groups(tot, tot != 0, CARBONYL_MERGE)
                s = summarize(tm, pm, 50)[["group", "median"]]
                s["method"], s["target"] = tag, t.upper()
                parts.append(s)
        fr = pd.read_csv(DATA_DIR / "crosscheck_desc.csv")
        fr = fr[["group", "median", "target"]].assign(method="RF fr_* counts")
        fr["target"] = fr["target"].str.upper()
        parts.append(fr)

        t2 = (pd.concat(parts)
                .pivot_table(index="group", columns=["target", "method"],
                             values="median"))
        drop = ["carbon skeleton", "unassigned", "baseline"]
        t2 = t2.drop(index=[g for g in drop if g in t2.index])
        write(t2, "groups", "Median per-molecule group contribution (eV), three methods.")
        print("\n" + t2.round(3).to_string())
    except FileNotFoundError as e:
        print(f"  skipped group table: {e}")

    # --- Tables 3 and 4: validation and ablations ---
    try:
        v = pd.concat([pd.read_csv(DATA_DIR / f"validation_{t}.csv").assign(target=t.upper())
                       for t in TARGETS])
        t3 = v.set_index(["target", "group"])[
            ["n", "shap_median", "removal_median", "sufficiency_r", "agreement"]]
        write(t3, "validation", "Removal and sufficiency tests (LightGBM, CF-MF).")
        print("\n" + t3.round(3).to_string())
    except FileNotFoundError as e:
        print(f"  skipped validation table: {e}")

    try:
        a = pd.read_csv(DATA_DIR / "ablations.csv")
        t4 = a.pivot_table(index=["ablation", "variant", "n_features"],
                           columns="target", values="test_mae")
        write(t4, "ablations", "Ablation study: test MAE (eV).", "%.4f")
        print("\n" + t4.round(4).to_string())
    except FileNotFoundError as e:
        print(f"  skipped ablation table: {e}")