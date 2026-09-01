#read-only inspection of the raw QM9 file.
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from config import DATA_RAW, HARTREE_TO_EV, TARGETS

RDLogger.DisableLog("rdApp.*")

df = pd.read_csv(DATA_RAW)
print(f"rows: {len(df)}")
print(f"columns: {list(df.columns)}\n")

# --- units ---
for t in TARGETS:
    v = df[t]
    print(f"{t}: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
    print(f"  -> as eV: min={v.min()*HARTREE_TO_EV:.3f} max={v.max()*HARTREE_TO_EV:.3f}")
print()

# --- parsing ---
mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
bad = [i for i, m in enumerate(mols) if m is None]
print(f"unparseable SMILES: {len(bad)}")
if bad:
    print(df["smiles"].iloc[bad[:10]].tolist(), "\n")

ok = np.array([m is not None for m in mols])
canon = pd.Series(
    [Chem.MolToSmiles(m) if m is not None else None for m in mols],
    index=df.index,
)

# --- duplicates ---
dup_mask = canon[ok].duplicated(keep=False)
n_dup_rows = int(dup_mask.sum())
groups = canon[ok][dup_mask].groupby(canon[ok][dup_mask])
print(f"rows in duplicate groups: {n_dup_rows} across {groups.ngroups} structures")

if groups.ngroups:
    spreads = {t: [] for t in TARGETS}
    for _, idx in groups.groups.items():
        for t in TARGETS:
            vals = df.loc[idx, t].to_numpy() * HARTREE_TO_EV
            spreads[t].append(vals.max() - vals.min())
    for t in TARGETS:
        s = np.array(spreads[t])
        print(f"  {t} label spread within duplicates (eV): "
              f"median={np.median(s):.4f} p95={np.percentile(s, 95):.4f} max={s.max():.4f}")
print()

# --- composition ---
elements, heavy, charged, frags = set(), [], 0, 0
for m in mols:
    if m is None:
        continue
    elements.update(a.GetSymbol() for a in m.GetAtoms())
    heavy.append(m.GetNumHeavyAtoms())
    if Chem.GetFormalCharge(m) != 0:
        charged += 1
    if len(Chem.GetMolFrags(m)) > 1:
        frags += 1

heavy = np.array(heavy)
print(f"elements present: {sorted(elements)}")
print(f"heavy atoms: min={heavy.min()} max={heavy.max()} mean={heavy.mean():.2f}")
print(f"charged molecules: {charged}")
print(f"multi-fragment molecules: {frags}\n")

# --- trivial baseline ---
for t in TARGETS:
    v = df[t].to_numpy() * HARTREE_TO_EV
    print(f"{t}: std={v.std():.4f} eV | mean-predictor MAE={np.abs(v - v.mean()).mean():.4f} eV")

RDLogger.EnableLog("rdApp.*")