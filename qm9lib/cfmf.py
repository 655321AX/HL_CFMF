# qm9lib/cfmf.py — collision-free Morgan fingerprints via unfolded identifiers.
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
import json

from config import DATA_DIR, TARGETS
from qm9lib.data import load_clean, load_splits


def _morgan_one(smiles, radius):
    """Unfolded count Morgan FP for one molecule, as compact arrays."""
    mol = Chem.MolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = gen.GetSparseCountFingerprint(mol, additionalOutput=ao)

    counts = fp.GetNonzeroElements()          # {identifier: count}
    info = ao.GetBitInfoMap()                 # {identifier: ((atom, radius), ...)}
    ids = np.fromiter(counts.keys(), dtype=np.int64)
    return (
        ids,
        np.array([counts[i] for i in ids], dtype=np.float32),
        np.array([info[i][0][1] for i in ids], dtype=np.int8),   # radius
        np.array([info[i][0][0] for i in ids], dtype=np.int16),  # centre atom
    )


def compute_morgan(smiles_list, radius=2, n_jobs=-1):
    return Parallel(n_jobs=n_jobs, batch_size=500)(
        delayed(_morgan_one)(s, radius) for s in smiles_list
    )


def build_matrix(results):
    """Stack per-molecule arrays into one sparse matrix, one column per identifier."""
    lengths = [len(r[0]) for r in results]
    rows = np.repeat(np.arange(len(results)), lengths)
    ids = np.concatenate([r[0] for r in results])
    cnt = np.concatenate([r[1] for r in results])
    rad = np.concatenate([r[2] for r in results])
    ctr = np.concatenate([r[3] for r in results])

    uniq, first, cols = np.unique(ids, return_index=True, return_inverse=True)
    X = sparse.csr_matrix((cnt, (rows, cols)), shape=(len(results), len(uniq)))

    meta = pd.DataFrame({
        "identifier": uniq,
        "radius": rad[first],
        "example_row": rows[first],
        "example_center": ctr[first],
    })

    # partial pre-hash collision check: one identifier seen at two radii
    n_pairs = len(np.unique(np.stack([cols, rad]), axis=1)[0])
    n_multi_radius = n_pairs - len(uniq)
    return X, meta, n_multi_radius


def select_bits(X_train, min_count=1):
    """Fit-on-train: columns present in at least min_count training molecules."""
    doc_freq = X_train.getnnz(axis=0)
    return np.flatnonzero(doc_freq >= min_count)

def load_cfmf(kind="random", min_count=10, max_radius=2, binary=False):
    """Train/val/test matrices + targets for one fingerprint configuration.

    Returns:
        (X_train, X_val, X_test), y, col_meta
        y[target] = (y_train, y_val, y_test)
        col_meta = metadata rows for the selected columns, in column order
    """
    df = load_clean()
    X = sparse.load_npz(DATA_DIR / "morgan_r2_counts.npz")
    meta = pd.read_parquet(DATA_DIR / "morgan_r2_meta.parquet")

    with open(DATA_DIR / "morgan_selected.json") as f:
        cols = np.array(json.load(f)[kind][f"min{min_count}"])
    cols = cols[meta["radius"].to_numpy()[cols] <= max_radius]

    X = X[:, cols]
    if binary:
        X = (X > 0).astype(np.float32)

    split_ids = load_splits(kind)
    Xs = tuple(X[df.index.get_indexer(ids)] for ids in split_ids)
    y = {t: tuple(df.loc[ids, t + "_ev"].to_numpy() for ids in split_ids)
         for t in TARGETS}
    return Xs, y, meta.iloc[cols].reset_index(drop=True)