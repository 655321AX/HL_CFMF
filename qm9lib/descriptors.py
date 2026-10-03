# qm9lib/descriptors.py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
import json

from config import DATA_DIR, TARGETS
from qm9lib.data import load_clean, load_splits

def _calc_one(smiles):
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.CalcMolDescriptors(mol)   # dict: name -> value


def compute_descriptors(smiles_list, n_jobs=-1):
    """All RDKit 2D descriptors, one row per molecule. Stateless per molecule."""
    rows = Parallel(n_jobs=n_jobs, batch_size=500)(
        delayed(_calc_one)(s) for s in smiles_list
    )
    return pd.DataFrame(rows)


def select_descriptors(X_train, corr_threshold=0.95):
    """Fit-on-train filter: finite -> non-constant -> non-redundant."""
    X = X_train.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().all()]
    X = X.loc[:, X.std() > 0]

    corr = X.corr().abs().to_numpy()
    kept = []
    for j in range(corr.shape[0]):
        if all(corr[j, k] <= corr_threshold for k in kept):
            kept.append(j)
    return X.columns[kept].tolist()

def load_descriptors(kind="random"):
    """Train/val/test descriptor tables (train-selected columns) + targets."""
    df = load_clean()
    X = pd.read_parquet(DATA_DIR / "desc_raw.parquet")
    with open(DATA_DIR / "desc_selected.json") as f:
        cols = json.load(f)[kind]

    X = X[cols].replace([np.inf, -np.inf], np.nan)
    split_ids = load_splits(kind)
    Xs = tuple(X.loc[ids] for ids in split_ids)
    y = {t: tuple(df.loc[ids, t + "_ev"].to_numpy() for ids in split_ids)
         for t in TARGETS}
    return Xs, y, cols