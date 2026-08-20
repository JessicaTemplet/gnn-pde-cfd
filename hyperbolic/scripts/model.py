"""
Encode-Process-Decode GNN for the hyperbolic stage: a learned one-step
time integrator for Burgers' equation, u_t -> u_{t+dt}.

Same family of architecture as the elliptic stage (message passing on a
multiscale graph), two differences that matter for a time-stepper:
  - the network predicts the *increment* du = u_{t+dt} - u_t rather than the
    absolute next state. Per model step du is small compared to u itself, so
    this is an easier target and (importantly for the rollout story) it
    means an untrained/zero-output network defaults to the identity map
    u_{t+dt} = u_t, i.e. "nothing changes", rather than to some arbitrary
    absolute field - a much safer failure mode when the model is later run
    autoregressively for hundreds of steps.
  - no boundary handling: the domain is periodic, so there's nothing to
    hard-enforce.
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


class BurgersStepGNN(nn.Module):
    def __init__(self, node_in_dim=2, edge_in_dim=2, hidden_dim=32, n_layers=4):
        super().__init__()
        self.node_encoder = mlp(node_in_dim, hidden_dim, hidden_dim)
        self.edge_encoder = mlp(edge_in_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim) for _ in range(n_layers)])
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
        """Autoregressive rollout from an initial condition. u0: (N, 1).
        Returns trajectory (n_steps + 1, N)."""
        u = u0
        traj = [u.squeeze(-1).clone()]
        for _ in range(n_steps):
            x = torch.cat([u, pos], dim=-1)
            u = self.forward(x, edge_index, edge_attr)
            traj.append(u.squeeze(-1).clone())
        return torch.stack(traj)
