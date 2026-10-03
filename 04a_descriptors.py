# 04_features.py — part A: descriptors.
import json

from config import DATA_DIR
from qm9lib.data import load_clean, load_splits
from qm9lib.descriptors import compute_descriptors, select_descriptors

if __name__ == "__main__":
    df = load_clean()

    print(f"computing descriptors for {len(df)} molecules...")
    X = compute_descriptors(df["canonical_smiles"].tolist())
    X.index = df.index
    X.to_parquet(DATA_DIR / "desc_raw.parquet")
    print(f"raw descriptor table: {X.shape}\n")

    selected = {}
    for kind in ("random", "scaffold"):
        train_ids, _, _ = load_splits(kind)
        cols = select_descriptors(X.loc[train_ids])
        selected[kind] = cols
        n_fr = sum(c.startswith("fr_") for c in cols)
        print(f"{kind}: {len(cols)} kept ({n_fr} fr_* group counts)")

    with open(DATA_DIR / "desc_selected.json", "w") as f:
        json.dump(selected, f, indent=1)

    print("\nkept (random split):")
    print(", ".join(selected["random"]))