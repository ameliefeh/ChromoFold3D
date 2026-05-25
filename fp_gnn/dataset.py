from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from fp_gnn.chromophore_graph import build_chromophore_graph
from fp_gnn.pdb_io import get_protein_residue_ca
from fp_gnn.protein_graph import build_protein_graph

# PDB entries whose chromophore atoms are incomplete to build a full chromophore graph
# The process.py is sorting out the proteins in the dataset and keeping only those that have the correct number of atoms for the
# chromophore and the protein (Calpha atoms of each residue/amino acid)
SKIP_PDB_CODES = {"1HUY"}

# 20 canonical amino acids in alphabetical one-letter order.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
SEQ_FEAT_DIM = len(AA_ALPHABET) + 1  # 20 composition fractions + log1p(length)


def sequence_to_features(seq: str) -> torch.Tensor:
    """Encode an amino-acid sequence into a fixed 21-dimensional feature vector.

    [0..19]  = relative frequency of each canonical amino acid (sums to ≤ 1).
               Non-canonical amino acids are ignored, similar to standard
               composition-based approaches.

    [20]     = log1p(sequence_length), used as a simple proxy for sequence size.
               This complements kDa (measured size/mass of the protein) since the two are related but
               not identical—e.g., fusion tags in protein can change kDa without affecting the
               chromophore region.

    Returns a tensor of shape [1, 21], which batches to [B, 21] in PyG.
    """
    seq = (seq or "").upper()
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float32)
    for ch in seq:
        idx = AA_INDEX.get(ch)
        if idx is not None:
            counts[idx] += 1.0
    n = max(len(seq), 1)
    composition = counts / n
    log_length = np.log1p(n).astype(np.float32)
    feat = np.concatenate([composition, [log_length]])
    return torch.from_numpy(feat).unsqueeze(0)  # [1, 21]


class FPData(Data):
    """Holds protein graph + chromophore graph in one object."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "chrom_edge_index":
            return self.chrom_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "chrom_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class FluorProteinDataset(InMemoryDataset):
    def __init__(self, root, labels_csv="data/labels.csv", repo_root="."):
        self._labels_csv = Path(repo_root) / labels_csv
        self._repo_root = Path(repo_root)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self._labels_csv)
        data_list = []
        for _, row in df.iterrows():
            if row["pdb_code"] in SKIP_PDB_CODES:
                print(
                    f"[WARN] skipping {row['pdb_code']}: chromophore atoms "
                    f"incomplete in PDB (see docs/methodology.md)"
                )
                continue
            pdb_path = self._repo_root / row["pdb_path"]

            res_names, ca = get_protein_residue_ca(pdb_path)
            prot = build_protein_graph(res_names, ca, cutoff=8.0)

            chrom = build_chromophore_graph(pdb_path)

            item = FPData(
                # Protein graph
                x=prot.x,
                edge_index=prot.edge_index,
                edge_attr=prot.edge_attr,
                # Chromophore graph (chrom_ prefix)
                chrom_x=chrom.x,
                chrom_edge_index=chrom.edge_index,
                chrom_edge_attr_chem=chrom.edge_attr_chem,
                chrom_edge_attr_dist=chrom.edge_attr_dist,
                # Scalars
                kda=torch.tensor([row["kDa"]], dtype=torch.float),
                y=torch.tensor([[row["brightness"], row["emission"]]], dtype=torch.float),
                pdb_code=row["pdb_code"],
                # Sequence features ([1, 21]): AA composition + log-length.
                seq_feat=sequence_to_features(row["sequence"]),
            )
            data_list.append(item)

        torch.save(self.collate(data_list), self.processed_paths[0])
