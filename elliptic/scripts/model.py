"""
Encode-Process-Decode graph neural network (MeshGraphNets-style) for the
elliptic stage. This is deliberately mesh-agnostic: nothing here assumes a
regular grid, so the same architecture applies to an unstructured CFD mesh
later - only the graph-building step (common/grid.py) would change.

Architecture:
  - node encoder:  per-node features -> hidden state
  - edge encoder:  per-edge features -> hidden state
  - K rounds of message passing (an EdgeConv-like update): each edge computes
    a message from its two endpoint states + its own state, edges into a
    node are summed, and the node state is updated with a residual MLP
  - decoder: hidden state -> predicted scalar field u
  - hard Dirichlet BC: boundary nodes are overwritten with their known
    boundary value, so the boundary condition is satisfied exactly rather
    than penalized in the loss
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def mlp(in_dim, hidden_dim, out_dim, n_hidden=1):
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class MessagePassingLayer(MessagePassing):
    def __init__(self, hidden_dim):
        # mean (not sum) aggregation: the multiscale graph gives interior
        # nodes far more edges than nodes near the boundary/corners (long
        # strides get clipped there), so summing messages would make a
        # node's activation scale depend on its position/degree rather than
        # on the PDE data - mean keeps every node's update on the same scale.
        super().__init__(aggr="mean")
        self.edge_mlp = mlp(3 * hidden_dim, hidden_dim, hidden_dim, n_hidden=2)
        self.node_mlp = mlp(2 * hidden_dim, hidden_dim, hidden_dim, n_hidden=2)

    def forward(self, h, edge_index, e):
        agg = self.propagate(edge_index, h=h, e=e)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_new

    def message(self, h_i, h_j, e):
        return self.edge_mlp(torch.cat([h_i, h_j, e], dim=-1))


class EllipticGNN(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=3, hidden_dim=64, n_layers=6):
        super().__init__()
        self.node_encoder = mlp(node_in_dim, hidden_dim, hidden_dim, n_hidden=2)
        self.edge_encoder = mlp(edge_in_dim, hidden_dim, hidden_dim, n_hidden=2)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim) for _ in range(n_layers)])
        self.decoder = mlp(hidden_dim, hidden_dim, 1, n_hidden=2)

    def forward(self, x, edge_index, edge_attr, boundary_mask, bc_value):
        # rescale the source term column (raw amplitudes up to ~14) to O(1)
        # before it hits the encoder; everything else (boundary flag/value,
        # x, y in [0, 1]) is already reasonably scaled.
        x = torch.cat([x[:, :1] / 8.0, x[:, 1:]], dim=-1)
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        u_pred = self.decoder(h)
        # hard-enforce the Dirichlet boundary condition
        u_pred = torch.where(boundary_mask.unsqueeze(-1), bc_value.unsqueeze(-1), u_pred)
        return u_pred
