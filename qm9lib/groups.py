# qm9lib/groups.py — functional group definitions and column assignment.
import numpy as np
import pandas as pd
from rdkit import Chem

# Ordered most-specific first: a column is assigned to the first group it matches.
# Only groups possible in QM9 (C, H, N, O, F; <= 9 heavy atoms).
GROUP_SMARTS = [
    ("amide",            "[CX3](=[OX1])[NX3]"),
    ("ester",            "[CX3](=[OX1])[OX2][#6]"),
    ("carboxylic acid",  "[CX3](=[OX1])[OX2H1]"),
    ("anhydride",        "[CX3](=[OX1])[OX2][CX3]=[OX1]"),
    ("aldehyde",         "[CX3H1](=[OX1])"),
    ("ketone",           "[#6][CX3](=[OX1])[#6]"),
    ("nitrile",          "[NX1]#[CX2]"),
    ("imine",            "[CX3]=[NX2]"),
    ("nitro",            "[NX3](=[OX1])=[OX1]"),
    ("epoxide",          "[OX2r3]1[#6r3][#6r3]1"),
    ("aziridine",        "[NX3r3]1[#6r3][#6r3]1"),
    ("furan-type O",     "[oX2]"),
    ("pyrrole-type NH",  "[nX3H1]"),
    ("pyridine-type N",  "[nX2]"),
    ("phenol OH",        "[OX2H1][c]"),
    ("alcohol OH",       "[OX2H1][#6]"),
    ("ether O",          "[OX2]([#6])[#6]"),
    ("primary amine",    "[NX3H2][#6]"),
    ("secondary amine",  "[NX3H1]([#6])[#6]"),
    ("tertiary amine",   "[NX3H0]([#6])([#6])[#6]"),
    ("alkyne",           "[CX2]#[CX2]"),
    ("alkene",           "[CX3]=[CX3]"),
    ("fluoride",         "[F]"),
    ("carbon skeleton",  "[#6]"),
]

_PATTERNS = [(name, Chem.MolFromSmarts(s)) for name, s in GROUP_SMARTS]


def groups_in(mol):
    """{group_name: set of atom indices} for every group present in mol."""
    found = {}
    for name, patt in _PATTERNS:
        hits = mol.GetSubstructMatches(patt)
        if hits:
            found[name] = [set(h) for h in hits]
    return found


def assign_columns(frags):
    """Assign each column to a group by its CENTRE atom.

    A Morgan identifier is defined by the atom it is centred on; the
    surrounding environment is context. So a column belongs to group G
    when its centre atom is one of G's core atoms.
    """
    from qm9lib.data import load_clean

    smiles_by_id = load_clean()["canonical_smiles"]
    mol_cache, atom_groups = {}, {}
    order = {name: i for i, (name, _) in enumerate(GROUP_SMARTS)}
    rows = []

    for r in frags.itertuples():
        mid = r.example_mol_id
        if mid not in mol_cache:
            mol = Chem.MolFromSmiles(smiles_by_id[mid])
            mol_cache[mid] = mol
            # atom index -> groups whose core contains that atom
            per_atom = {}
            for name, occ in groups_in(mol).items():
                for atoms in occ:
                    for a in atoms:
                        per_atom.setdefault(a, set()).add(name)
            atom_groups[mid] = per_atom

        matched = sorted(atom_groups[mid].get(int(r.example_center), ()))
        primary = min(matched, key=lambda g: order[g]) if matched else "unassigned"
        rows.append({"col": r.col, "group": primary,
                     "n_groups_matched": len(matched),
                     "all_groups": "|".join(matched)})

    return pd.DataFrame(rows)