# 02_clean.py — apply cleaning decisions, write canonical dataset to disk.
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from config import DATA_RAW, DATA_DIR, HARTREE_TO_EV, TARGETS

RDLogger.DisableLog("rdApp.*")

UNCHARACTERIZED = DATA_DIR / "uncharacterized.txt"   # optional
MIN_HEAVY_ATOMS = 2

df = pd.read_csv(DATA_RAW)
print(f"start: {len(df)}")

# --- 1. sanity check: does gap == lumo - homo? ---
gap_err = np.abs(df["gap"] - (df["lumo"] - df["homo"])).max()
print(f"max |gap - (lumo - homo)|: {gap_err:.2e} Ha  (expect <=1e-4: values rounded to 4 dp)")

# --- 2. canonicalize ---
mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
df["canonical_smiles"] = [Chem.MolToSmiles(m) for m in mols]
df["n_heavy"] = [m.GetNumHeavyAtoms() for m in mols]

# --- 3. drop uncharacterized (if list available) ---
if UNCHARACTERIZED.exists():
    with open(UNCHARACTERIZED) as f:
        bad_ids = {
            int(line.split()[0])
            for line in f
            if line.split() and line.split()[0].isdigit()
        }
    gdb_idx = df["mol_id"].str.replace("gdb_", "", regex=False).astype(int)
    mask = ~gdb_idx.isin(bad_ids)
    print(f"uncharacterized removed: {(~mask).sum()}")
    df = df[mask]
else:
    print("WARNING: uncharacterized.txt not found — 3054 noisy labels retained")

# --- 4. drop tiny molecules ---
small = df["n_heavy"] < MIN_HEAVY_ATOMS
print(f"molecules with <{MIN_HEAVY_ATOMS} heavy atoms removed: {small.sum()}")
df = df[~small]

# --- 5. deduplicate on canonical structure ---
before = len(df)
df = (
    df.groupby("canonical_smiles", as_index=False)
      .agg({**{t: "mean" for t in TARGETS},
            "mol_id": "first",
            "n_heavy": "first"})
)
print(f"duplicates collapsed: {before - len(df)}")

# --- 6. convert to eV, finalize ---
for t in TARGETS:
    df[t + "_ev"] = df[t] * HARTREE_TO_EV

out = df[["mol_id", "canonical_smiles", "n_heavy"] + [t + "_ev" for t in TARGETS]]
out = out.reset_index(drop=True)
out.to_parquet(DATA_DIR / "qm9_clean.parquet", index=False)

print(f"\nfinal: {len(out)} molecules -> {DATA_DIR / 'qm9_clean.parquet'}")
for t in TARGETS:
    v = out[t + "_ev"].to_numpy()
    print(f"{t}: mean={v.mean():.4f} std={v.std():.4f} "
          f"baseline MAE={np.abs(v - v.mean()).mean():.4f} eV")

RDLogger.EnableLog("rdApp.*")