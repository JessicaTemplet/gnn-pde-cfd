"""
Encode-Process-Decode GNN for the parabolic stage: a learned one-step
time integrator for the heat equation, u_t -> u_{t+dt}.

Identical architecture to the hyperbolic stage: the same message-passing
family applies to any equation where the state at each node depends on
the state of its neighbors at the previous timestep. The key design
choice that carries over unchanged:
  - Predict the increment du = u_{t+dt} - u_t rather than the absolute
    next state. This ensures a zero-output (untrained) network is the
    identity map, a safe failure mode when later rolled out autoregressively.
  - Periodic domain, so no boundary enforcement needed.

The main hypothesis for this stage (see README): because diffusion is
dissipative, a naive one-step trained version of this model should already
be rollout-stable -- prediction errors are damped rather than advected and
amplified, unlike Burgers'. This makes parabolic the natural contrast case
for the hyperbolic instability story.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def mlp(in_dim, hidden_dim, out_dim, n_hidden=2):
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class MessagePassingLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="mean")
        self.edge_mlp = mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_mlp = mlp(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, h, edge_index, e):
        agg = self.propagate(edge_index, h=h, e=e)
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))

    def message(self, h_i, h_j, e):
        return self.edge_mlp(torch.cat([h_i, h_j, e], dim=-1))


class HeatStepGNN(nn.Module):
    def __init__(self, node_in_dim=2, edge_in_dim=2, hidden_dim=32, n_layers=4):
        super().__init__()
        self.node_encoder = mlp(node_in_dim, hidden_dim, hidden_dim)
        self.edge_encoder = mlp(edge_in_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.decoder = mlp(hidden_dim, hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        """x: (N, 2) = [u_t, x_pos]. Returns predicted u_{t+dt}: (N, 1)."""
        u_t = x[:, :1]
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        du = self.decoder(h)
        return u_t + du

    def rollout(self, u0, pos, edge_index, edge_attr, n_steps):
        """Autoregressive rollout from an initial condition.

        Args:
            u0: (N, 1) initial field values
            pos: (N, 1) node positions
            edge_index: (2, E) graph connectivity
            edge_attr: (E, 2) edge features
            n_steps: number of model steps to roll out

        Returns:
            trajectory: (n_steps + 1, N) tensor, first slice is u0
        """
        u = u0
        traj = [u.squeeze(-1).clone()]
        for _ in range(n_steps):
            x = torch.cat([u, pos], dim=-1)
            u = self.forward(x, edge_index, edge_attr)
            traj.append(u.squeeze(-1).clone())
        return torch.stack(traj)
