# qm9lib/explain.py — aggregate per-column SHAP into per-molecule group totals.
import numpy as np
import pandas as pd
from scipy import sparse


def group_indicator(assign, n_cols, groups=None):
    """Sparse (n_cols x n_groups) 0/1 matrix from column -> group assignment."""
    if groups is None:
        groups = sorted(assign["group"].unique())
    idx = {g: k for k, g in enumerate(groups)}
    col = assign["col"].to_numpy()
    grp = assign["group"].map(idx).to_numpy()
    M = sparse.csr_matrix(
        (np.ones(len(col)), (col, grp)), shape=(n_cols, len(groups))
    )
    return M, groups


def ridge_group_shap(model, X_bg, X_eval, assign, groups=None):
    """Per-molecule SHAP totals per group, without densifying anything.

    phi_j = w_j * (x_j - mean_j)  =>  group total = sum_j in G of phi_j
    """
    M, groups = group_indicator(assign, X_eval.shape[1], groups)
    w = model.coef_

    wx = X_eval.multiply(w)                      # sparse: w_j * x_ij
    totals = np.asarray((wx @ M).todense())      # (n_molecules, n_groups)
    xbar = np.asarray(X_bg.mean(axis=0)).ravel()
    totals -= (w * xbar) @ M                     # subtract the constant part

    present = np.asarray(((X_eval > 0).astype(np.int8) @ M).todense()) > 0
    return pd.DataFrame(totals, columns=groups), pd.DataFrame(present, columns=groups)


def summarize(totals, present, min_molecules=50):
    """Median, IQR and support for each group, over molecules containing it."""
    rows = []
    for g in totals.columns:
        v = totals[g].to_numpy()[present[g].to_numpy()]
        if len(v) < min_molecules:
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append({"group": g, "n_molecules": len(v), "median": med,
                     "q1": q1, "q3": q3, "iqr": q3 - q1,
                     "mean": v.mean(), "frac_negative": float((v < 0).mean())})
    return pd.DataFrame(rows).sort_values("median").reset_index(drop=True)

def per_molecule_group_shap(model, X_bg, X_eval, eval_ids, col_meta, radius=2):
    """Assign SHAP per molecule by each occurrence's centre atom in THAT molecule.

    A low-radius identifier (e.g. carbonyl O) is context-free and occurs in
    several group contexts, so its group cannot be fixed per column. Here each
    occurrence is attributed to the group of the atom it is centred on.

    Returns (totals, present): DataFrames indexed like eval_ids. The 'baseline'
    column holds the -w*mean terms of columns absent from the molecule, so
    totals.sum(axis=1) == prediction - base value.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from qm9lib.data import load_clean
    from qm9lib.groups import GROUP_SMARTS, groups_in

    w = model.coef_
    xbar = np.asarray(X_bg.mean(axis=0)).ravel()
    wxbar = w * xbar
    total_offset = wxbar.sum()

    col_of = {int(i): c for c, i in enumerate(col_meta["identifier"])}
    names = [g for g, _ in GROUP_SMARTS] + ["unassigned", "baseline"]
    order = {g: k for k, g in enumerate(names)}
    smiles_by_id = load_clean()["canonical_smiles"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    totals = np.zeros((len(eval_ids), len(names)))
    present = np.zeros((len(eval_ids), len(names)), dtype=bool)

    for i, mid in enumerate(eval_ids):
        mol = Chem.MolFromSmiles(smiles_by_id[mid])

        atom_groups = {}
        for name, occurrences in groups_in(mol).items():
            for atoms in occurrences:
                for a in atoms:
                    prev = atom_groups.get(a)
                    if prev is None or order[name] < order[prev]:
                        atom_groups[a] = name

        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetSparseCountFingerprint(mol, additionalOutput=ao)

        seen = 0.0
        for ident, occurrences in ao.GetBitInfoMap().items():
            col = col_of.get(int(ident))
            if col is None:                      # identifier not in training set
                continue
            phi = w[col] * len(occurrences) - wxbar[col]
            seen += wxbar[col]
            share = phi / len(occurrences)
            for centre, _ in occurrences:
                k = order[atom_groups.get(centre, "unassigned")]
                totals[i, k] += share
                present[i, k] = True

        totals[i, order["baseline"]] -= total_offset - seen

    import pandas as pd
    idx = pd.Index(eval_ids, name="mol_id")
    return (pd.DataFrame(totals, columns=names, index=idx),
            pd.DataFrame(present, columns=names, index=idx))

# Carbonyl subtypes share the radius-0 C=O oxygen identifier, so an additive
# model cannot separate them reliably. Merging them makes the total trustworthy
# at the cost of the subtype distinction.
CARBONYL_MERGE = {
    "aldehyde": "carbonyl C=O",
    "ketone": "carbonyl C=O",
    "ester": "carbonyl C=O",
    "amide": "carbonyl C=O",
    "carboxylic acid": "carbonyl C=O",
    "anhydride": "carbonyl C=O",
}


def merge_groups(totals, present, mapping):
    """Sum group columns that map to the same name; OR their presence flags."""
    new = [mapping.get(c, c) for c in totals.columns]
    t = totals.T.groupby(new, sort=False).sum().T
    p = present.T.groupby(new, sort=False).any().T
    return t, p

def lgbm_group_shap(booster, X_eval, eval_ids, col_meta, radius=2,
                    chunk=500, num_iteration=None):
    """Per-molecule group totals from TreeSHAP, assigned by centre atom.

    Same attribution rule as per_molecule_group_shap, but the per-column SHAP
    values come from TreeSHAP instead of the Ridge closed form. Processed in
    chunks so the dense SHAP block never exceeds a few hundred MB.
    """
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from qm9lib.data import load_clean
    from qm9lib.groups import GROUP_SMARTS, groups_in

    col_of = {int(i): c for c, i in enumerate(col_meta["identifier"])}
    names = [g for g, _ in GROUP_SMARTS] + ["unassigned", "baseline"]
    order = {g: k for k, g in enumerate(names)}
    smiles_by_id = load_clean()["canonical_smiles"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    totals = np.zeros((len(eval_ids), len(names)))
    present = np.zeros((len(eval_ids), len(names)), dtype=bool)

    for start in range(0, len(eval_ids), chunk):
        stop = min(start + chunk, len(eval_ids))
        contrib = booster.predict(X_eval[start:stop], pred_contrib=True,
                                  num_iteration=num_iteration)

        for local, i in enumerate(range(start, stop)):
            mol = Chem.MolFromSmiles(smiles_by_id[eval_ids[i]])

            atom_groups = {}
            for name, occurrences in groups_in(mol).items():
                for atoms in occurrences:
                    for a in atoms:
                        prev = atom_groups.get(a)
                        if prev is None or order[name] < order[prev]:
                            atom_groups[a] = name

            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            gen.GetSparseCountFingerprint(mol, additionalOutput=ao)

            assigned = set()
            for ident, occurrences in ao.GetBitInfoMap().items():
                col = col_of.get(int(ident))
                if col is None:
                    continue
                assigned.add(col)
                share = contrib[local, col] / len(occurrences)
                for centre, _ in occurrences:
                    k = order[atom_groups.get(centre, "unassigned")]
                    totals[i, k] += share
                    present[i, k] = True

            # SHAP of columns the molecule does not contain has no atom to own it
            rest = contrib[local, :-1].sum() - contrib[local, list(assigned)].sum()
            totals[i, order["baseline"]] += rest

        print(f"  {stop}/{len(eval_ids)}")

    idx = pd.Index(eval_ids, name="mol_id")
    return (pd.DataFrame(totals, columns=names, index=idx),
            pd.DataFrame(present, columns=names, index=idx))