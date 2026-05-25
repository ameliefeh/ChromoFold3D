import json
from pathlib import Path

import numpy as np
import requests
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from fp_gnn.pdb_io import get_chromophore_code, get_chromophore_pdb_block

DEFAULT_CCD_CACHE = Path("data/ccd_cache")


def get_ccd_smiles_cached(residue_code: str, cache_dir: Path = DEFAULT_CCD_CACHE) -> str:
    """Return canonical SMILES for `residue_code` from the RCSB Chemical
    Component Dictionary, using a local JSON cache.

    We use it to retrieve the canonical smile of each of the ligand (chromophore) contained in each protein. Each protein can have
    a different chromophore. With the smile, we build then a Mol object with RDKIT for subsequent chromophore graph features.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{residue_code}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())["smiles"]

    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{residue_code}"
    data = requests.get(url).json()
    smiles = data["rcsb_chem_comp_descriptor"]["SMILES"]
    cache_file.write_text(json.dumps({"smiles": smiles}))
    return smiles


def mol_to_graph(mol):
    """Convert a sanitized RDKit Mol (heavy atoms only, with a 3D
    conformer) to a PyG Data object using OGB-compatible features. We improve the features used in the exercice session
    of the lecture by adding more atom feature with OGB. """
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    positions = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(n_atoms)],
        dtype=np.float32,
    )

    # OGB atom feature schema (9 ints per atom; used by AtomEncoder in model.py):
    #   [0] atomic_num            (C, N, O, S, ...)
    #   [1] chirality             (CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW/CCW, OTHER)
    #   [2] degree                (number of bonded heavy-atom neighbors)
    #   [3] formal_charge         (integer charge on the atom)
    #   [4] num_hs                (implicit H count)
    #   [5] num_radical_electrons (usually 0)
    #   [6] hybridization         (sp / sp2 / sp3 / sp3d / sp3d2)
    #   [7] is_aromatic           (0/1)
    #   [8] is_in_ring            (0/1)
    atom_feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.long)

    # OGB bond chemistry schema (3 ints per bond; used by BondEncoder in model.py):
    #   [0] bond_type   (single, double, triple, aromatic, misc)
    #   [1] bond_stereo (none, Z, E, cis, trans, any)
    #   [2] is_conjugated (0/1)
    # Plus a separate continuous distance feature (`edge_attr_dist`), which is first
    # projected with nn.Linear(1, H) and then added into the bond embedding inside ChromMPNN (chromophore message passing graph).
    src, dst = [], []
    chem_feats = []
    distances = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_to_feature_vector(bond)
        d = float(np.linalg.norm(positions[i] - positions[j]))
        # Both directions
        src.extend([i, j])
        dst.extend([j, i])
        chem_feats.extend([bf, bf])
        distances.extend([d, d])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_chem = torch.tensor(chem_feats, dtype=torch.long)
    edge_attr_dist = torch.tensor(distances, dtype=torch.float).unsqueeze(-1)
    pos = torch.tensor(positions, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr_chem=edge_attr_chem,
        edge_attr_dist=edge_attr_dist,
        pos=pos,
    )


def build_chromophore_graph(pdb_path: str | Path) -> Data:
    """Read the chromophore from a PDB file, build a sanitized 3D RDKit
    Mol via the CCD SMILES template, and return its PyG Data graph."""
    code = get_chromophore_code(pdb_path)
    block = get_chromophore_pdb_block(pdb_path, code)

    smiles = get_ccd_smiles_cached(code)
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise RuntimeError(f"RDKit could not parse SMILES for {code}: {smiles}")

    raw_mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if raw_mol is None:
        raise RuntimeError(f"RDKit could not parse PDB block for {code} in {pdb_path}")

    # Some PDBs omit the terminal -OH (OXT) of the chromophore's C-terminal
    # carboxylate; we strip it from the template so the heavy-atom counts match.
    # The SMARTS picks that specific -OH (the one off a CH2 bonded to a ring N),
    # not other carboxylates like the Asp side chain in DYG.
    n_heavy_pdb = sum(1 for a in raw_mol.GetAtoms() if a.GetAtomicNum() != 1)
    n_heavy_tmpl = template.GetNumAtoms()
    if n_heavy_tmpl > n_heavy_pdb:
        patt = Chem.MolFromSmarts("[#7&R][CH2][CX3](=[OX1])[OX2H]")
        matches = template.GetSubstructMatches(patt)
        if matches:
            oh_idx = matches[0][4]  # terminal -OH oxygen of the chromophore C-term
            rw = Chem.RWMol(template)
            rw.RemoveAtom(oh_idx)
            template = rw.GetMol()
            Chem.SanitizeMol(template)

    mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)

    return mol_to_graph(mol)
