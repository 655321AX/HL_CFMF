# paths, seeds, constants
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "qm9.csv"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"

for d in (DATA_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

SEED = 42
HARTREE_TO_EV = 27.211386245988

TARGETS = ["homo", "lumo"]