# 03_split.py — create and save random + scaffold splits (by mol_id).
import json

import numpy as np
import pandas as pd

from config import DATA_DIR, SEED, TARGETS
from qm9lib.splits import random_split, scaffold_split

FRACS = (0.8, 0.1, 0.1)

df = pd.read_parquet(DATA_DIR / "qm9_clean.parquet")
assert df["mol_id"].is_unique
ids = df["mol_id"].tolist()
print(f"molecules: {len(ids)}\n")

r_train, r_val, r_test = random_split(ids, FRACS, SEED)
s_train, s_val, s_test, groups = scaffold_split(
    ids, df["canonical_smiles"].tolist(), FRACS, SEED
)

# --- scaffold statistics ---
sizes = sorted((len(g) for g in groups.values()), reverse=True)
print(f"unique scaffolds: {len(groups)}")
print(f"acyclic (empty scaffold): {len(groups.get('', []))}")
print(f"largest 5 groups: {sizes[:5]}")
print(f"singleton scaffolds: {sum(s == 1 for s in sizes)}\n")

# --- sanity checks + per-split label summary ---
lookup = df.set_index("mol_id")
splits = {
    "random":   {"train": r_train, "val": r_val, "test": r_test},
    "scaffold": {"train": s_train, "val": s_val, "test": s_test},
}

for name, sp in splits.items():
    tr, va, te = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (tr & va or tr & te or va & te), f"{name}: overlap"
    assert len(tr | va | te) == len(ids), f"{name}: missing molecules"

    print(f"--- {name} ---")
    for part, members in sp.items():
        sub = lookup.loc[members]
        stats = "  ".join(
            f"{t}: {sub[t + '_ev'].mean():+.3f}±{sub[t + '_ev'].std():.3f}"
            for t in TARGETS
        )
        print(f"{part:>5}: {len(members):>6} ({len(members)/len(ids):.1%})  {stats}")
    print()

with open(DATA_DIR / "splits.json", "w") as f:
    json.dump({"seed": SEED, "fracs": FRACS, **splits}, f)
print(f"saved -> {DATA_DIR / 'splits.json'}")