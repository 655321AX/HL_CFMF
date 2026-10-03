# qm9lib/fragments.py — map fingerprint columns back to real substructures.
import pandas as pd
from rdkit import Chem

from qm9lib.data import load_clean


def _atom_label(atom):
    """Readable label for a radius-0 environment (single atom)."""
    arom = "aromatic " if atom.GetIsAromatic() else ""
    return (f"[{arom}{atom.GetSymbol()}; deg {atom.GetDegree()}; "
            f"{atom.GetTotalNumHs()}H; chg {atom.GetFormalCharge()}]")


def fragment_of(mol, center, radius):
    """Return (smiles, atom_indices) for the circular environment.

    H counts are written explicitly, since Morgan invariants encode them:
    an aldehyde's [CH1]=O and a ketone's [CH0]=O are different identifiers.
    """
    if radius == 0:
        return _atom_label(mol.GetAtomWithIdx(center)), (center,)

    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    if not env:
        return None, ()

    atoms = {center}
    for b in env:
        bond = mol.GetBondWithIdx(b)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    atoms = tuple(sorted(atoms))

    try:
        smi = Chem.MolFragmentToSmiles(
            mol, atomsToUse=list(atoms), bondsToUse=list(env),
            rootedAtAtom=center, canonical=False, allHsExplicit=True,
        )
    except Exception:
        smi = None
    return smi, atoms


def describe_columns(col_meta):
    """One row per fingerprint column: its substructure and where it was seen.

    col_meta comes from load_cfmf(); rows must stay in matrix column order.
    """
    smiles_by_id = load_clean()["canonical_smiles"]
    rows = []
    for col, m in enumerate(col_meta.itertuples()):
        mol = Chem.MolFromSmiles(smiles_by_id[m.example_mol_id])
        smi, atoms = fragment_of(mol, int(m.example_center), int(m.radius))
        rows.append({
            "col": col,
            "identifier": m.identifier,
            "radius": m.radius,
            "fragment": smi,
            "center_symbol": mol.GetAtomWithIdx(int(m.example_center)).GetSymbol(),
            "n_env_atoms": len(atoms),
            "example_mol_id": m.example_mol_id,
            "example_center": m.example_center,
            "example_smiles": smiles_by_id[m.example_mol_id],
        })
    return pd.DataFrame(rows)

def _atom_label(atom):
    """Readable label for a radius-0 environment (single atom)."""
    arom = "aromatic " if atom.GetIsAromatic() else ""
    ring = "; ring" if atom.IsInRing() else ""
    return (f"[{arom}{atom.GetSymbol()}; deg {atom.GetDegree()}; "
            f"{atom.GetTotalNumHs()}H; chg {atom.GetFormalCharge()}{ring}]")