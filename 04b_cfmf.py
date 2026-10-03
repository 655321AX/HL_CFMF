# 04b_cfmf.py — raw collision-free Morgan matrix + train-fitted column selections.
import json

import numpy as np
from scipy import sparse

from config import DATA_DIR
from qm9lib.cfmf import build_matrix, compute_morgan, select_bits
from qm9lib.data import load_clean, load_splits

RADIUS = 2
MIN_COUNTS = (1, 10)

if __name__ == "__main__":
    df = load_clean()

    print(f"computing unfolded Morgan r={RADIUS} for {len(df)} molecules...")
    results = compute_morgan(df["canonical_smiles"].tolist(), radius=RADIUS)
    X, meta, n_multi = build_matrix(results)

    meta["example_mol_id"] = df.index[meta["example_row"]].to_numpy()
    meta = meta.drop(columns="example_row")

    print(f"matrix: {X.shape}  nnz={X.nnz}  ({X.nnz / X.shape[0]:.1f} identifiers/molecule)")
    print(f"identifiers by radius: {meta['radius'].value_counts().sort_index().to_dict()}")
    print(f"identifiers seen at >1 radius (pre-hash collisions): {n_multi}\n")

    sparse.save_npz(DATA_DIR / "morgan_r2_counts.npz", X)
    meta.to_parquet(DATA_DIR / "morgan_r2_meta.parquet", index=False)

    selected = {}
    for kind in ("random", "scaffold"):
        train_ids, _, _ = load_splits(kind)
        pos = df.index.get_indexer(train_ids)
        X_train = X[pos]
        selected[kind] = {}
        for m in MIN_COUNTS:
            cols = select_bits(X_train, m)
            selected[kind][f"min{m}"] = cols.tolist()
            r1 = int((meta["radius"].to_numpy()[cols] <= 1).sum())
            print(f"{kind:>8} min{m:<3}: {len(cols):>6} columns  ({r1} with radius <= 1)")

    with open(DATA_DIR / "morgan_selected.json", "w") as f:
        json.dump(selected, f)
    print(f"\nsaved -> {DATA_DIR}")