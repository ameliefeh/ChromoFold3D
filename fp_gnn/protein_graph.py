import numpy as np
import torch
from torch_geometric.data import Data

# 20 standard amino acids in fixed alphabetical order; index = one-hot column
STANDARD_AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_INDEX = {name: i for i, name in enumerate(STANDARD_AA)}


def _one_hot_residues(residue_names: list[str]) -> torch.Tensor:
    N = len(residue_names)
    x = torch.zeros((N, 20), dtype=torch.float)
    for i, name in enumerate(residue_names):
        idx = AA_INDEX.get(name)
        if idx is None:
            print(f"[WARN] non-standard residue '{name}' at index {i}; encoding as zero vector")
            continue
        x[i, idx] = 1.0
    return x


def build_protein_graph(
    residue_names: list[str], ca_coords: np.ndarray, cutoff: float = 8.0
) -> Data:
    """Residue contact graph. Nodes = residues with 20-D one-hot;
    edges = pairs whose Cα-Cα distance <= cutoff Å, in both directions,
    no self-loops; edge_attr = scalar distance."""
    N = len(residue_names)
    assert ca_coords.shape == (N, 3), f"ca_coords shape {ca_coords.shape} != ({N}, 3)"

    coords = np.asarray(ca_coords, dtype=np.float32)
    # Pairwise Euclidean distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    mask = (dist <= cutoff) & (dist > 0.0)  # exclude self-loops
    src, dst = np.where(mask)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(dist[src, dst], dtype=torch.float).unsqueeze(-1)

    x = _one_hot_residues(residue_names)
    pos = torch.tensor(coords, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
