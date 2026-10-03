# qm9lib/data.py — loaders shared by every later stage.
import json

import pandas as pd

from config import DATA_DIR


def load_clean():
    """Clean dataset indexed by mol_id."""
    return pd.read_parquet(DATA_DIR / "qm9_clean.parquet").set_index("mol_id")


def load_splits(kind="random"):
    """Return (train_ids, val_ids, test_ids) for 'random' or 'scaffold'."""
    with open(DATA_DIR / "splits.json") as f:
        s = json.load(f)[kind]
    return s["train"], s["val"], s["test"]

PARAMS_FILE = DATA_DIR / "best_params.json"


def save_params(key, params):
    """Store params under key, keeping entries already in the file."""
    allp = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    allp[key] = params
    PARAMS_FILE.write_text(json.dumps(allp, indent=1))


def load_params(key):
    return json.loads(PARAMS_FILE.read_text())[key]