import random
from collections import defaultdict

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(ids, fracs, seed):
    """Shuffle ids and cut into train/val/test by fractions."""
    ids = np.asarray(ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_train = int(fracs[0] * len(ids))
    n_val = int(fracs[1] * len(ids))
    return (ids[perm[:n_train]].tolist(),
            ids[perm[n_train:n_train + n_val]].tolist(),
            ids[perm[n_train + n_val:]].tolist())


def murcko_scaffold(smiles):
    """Bemis-Murcko ring framework; '' for acyclic molecules."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)


def scaffold_split(ids, smiles, fracs, seed):
    """Balanced scaffold split: no scaffold appears in more than one set."""
    groups = defaultdict(list)
    for i, s in zip(ids, smiles):
        groups[murcko_scaffold(s)].append(i)

    n = len(ids)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)

    # groups too large for val/test go to train first
    big = [g for g in groups.values() if len(g) > n_val / 2]
    small = [g for g in groups.values() if len(g) <= n_val / 2]
    big.sort(key=len, reverse=True)
    random.Random(seed).shuffle(small)

    train, val, test = [], [], []
    for g in big + small:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            test += g
    return train, val, test, groups