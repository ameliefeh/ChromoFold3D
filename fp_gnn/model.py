import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MLP, NNConv, global_add_pool


def _make_edge_network(emb_dim: int, hidden: int | None = None) -> MLP:
    h = hidden or 2 * emb_dim
    return MLP([emb_dim, h, emb_dim * emb_dim], norm=None)


class ChromMPNN(nn.Module):
    """MPNN over the chromophore graph (atoms + bonds + bond distance)."""

    def __init__(self, node_embedding_dim: int = 64, num_message_steps: int = 3):
        super().__init__()
        H = node_embedding_dim
        self.num_message_steps = num_message_steps

        self.atom_emb = AtomEncoder(emb_dim=H)
        self.bond_chem_emb = BondEncoder(emb_dim=H)
        self.dist_proj = nn.Linear(1, H)

        self.message_layer = NNConv(H, H, nn=_make_edge_network(H), aggr="mean")
        self.gru = nn.GRU(input_size=H, hidden_size=H)

    def forward(self, batch):
        x = self.atom_emb(batch.chrom_x)
        e = self.bond_chem_emb(batch.chrom_edge_attr_chem) + self.dist_proj(
            batch.chrom_edge_attr_dist
        )

        h = x.unsqueeze(0)  # GRU expects [1, N, H]
        node_state = x
        for _ in range(self.num_message_steps):
            m = self.message_layer(node_state, batch.chrom_edge_index, e)
            m = F.relu(m)
            node_state, h = self.gru(m.unsqueeze(0), h)
            node_state = node_state.squeeze(0)

        # PyG doesn't automatically create chrom_x_batch for custom node features.
        # If we're batching graphs, we rebuild it using _slice_dict['chrom_x'].
        # If there's only one graph, we just assign all atoms to graph 0.
        chrom_batch = getattr(batch, "chrom_x_batch", None)
        if chrom_batch is None:
            slice_dict = getattr(batch, "_slice_dict", None)
            if slice_dict is not None and "chrom_x" in slice_dict:
                slices = slice_dict["chrom_x"].to(node_state.device)
                sizes = slices[1:] - slices[:-1]
                num_graphs = len(sizes)
                chrom_batch = torch.repeat_interleave(
                    torch.arange(num_graphs, device=node_state.device), sizes
                )
            else:
                # Single-graph (non-batched) case
                chrom_batch = torch.zeros(
                    node_state.shape[0], dtype=torch.long, device=node_state.device
                )
        return global_add_pool(node_state, chrom_batch)


class ProteinMPNN(nn.Module):
    """MPNN over the protein residue contact graph."""

    def __init__(
        self,
        node_embedding_dim: int = 64,
        num_message_steps: int = 3,
        num_residue_types: int = 20,
    ):
        super().__init__()
        H = node_embedding_dim
        self.num_message_steps = num_message_steps

        self.node_proj = nn.Linear(num_residue_types, H)
        self.edge_proj = nn.Linear(1, H)

        self.message_layer = NNConv(H, H, nn=_make_edge_network(H), aggr="mean")
        self.gru = nn.GRU(input_size=H, hidden_size=H)

    def forward(self, batch):
        x = self.node_proj(batch.x)
        e = self.edge_proj(batch.edge_attr)

        h = x.unsqueeze(0)
        node_state = x
        for _ in range(self.num_message_steps):
            m = self.message_layer(node_state, batch.edge_index, e)
            m = F.relu(m)
            node_state, h = self.gru(m.unsqueeze(0), h)
            node_state = node_state.squeeze(0)

        prot_batch = getattr(batch, "batch", None)
        if prot_batch is None:
            prot_batch = torch.zeros(
                node_state.shape[0], dtype=torch.long, device=node_state.device
            )
        return global_add_pool(node_state, prot_batch)


class FPNet(nn.Module):
    """Fluorescent-protein property network.

    Three feature streams are pooled to a per-protein vector and used as input to a
    small MLP head that predicts (brightness, emission):

      - Protein residue-contact graph (Cα within 8 Å)        -> MPNN -> H
      - Chromophore atom graph (CCD-typed bonds + distances) -> MPNN -> H
      - Sequence features (AA composition + log-length)      -> MLP  -> H
      - Standardised molecular weight (kDa_z)                -> 1

    Final feature vector is formed by concatenating prot, chrom, seq, and kDa_z (3H + 1)
    """

    def __init__(
        self,
        node_embedding_dim: int = 64,
        num_message_steps: int = 3,
        seq_input_dim: int = 21,
    ):
        super().__init__()
        H = node_embedding_dim
        self.protein_mpnn = ProteinMPNN(H, num_message_steps)
        self.chrom_mpnn = ChromMPNN(H, num_message_steps)
        self.seq_encoder = MLP([seq_input_dim, H, H], norm=None)
        self.head = MLP([3 * H + 1, H, 2], norm=None)

    def forward(self, batch):
        prot_emb = self.protein_mpnn(batch)
        chrom_emb = self.chrom_mpnn(batch)
        seq_emb = self.seq_encoder(batch.seq_feat)
        fused = torch.cat([prot_emb, chrom_emb, seq_emb, batch.kda_z.view(-1, 1)], dim=-1)
        return self.head(fused)
