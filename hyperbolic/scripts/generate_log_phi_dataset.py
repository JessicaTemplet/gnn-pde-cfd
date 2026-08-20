"""
Generate the log-phi dataset for Cole-Hopf v2 training.

Applies the Cole-Hopf forward transform to the existing Burgers' trajectories
to produce psi = log(phi) trajectories. Training a GNN on (psi_t, psi_{t+dt})
pairs lets us roll out Burgers' via Cole-Hopf without the positivity constraint
that broke v1:

    u = -2*nu * psi_x          (inverse: spectral derivative only, no division)

This is exactly the Cole-Hopf forward map: phi = exp(psi), u = -2*nu*phi_x/phi
= -2*nu*(exp(psi))'_x / exp(psi) = -2*nu*psi_x. So recovering u from psi is
a simple spectral derivative — no positivity enforcement, no division hazard.

psi dynamics: psi_t = nu*(psi_xx + psi_x^2)  (nonlinear, but bounded)

Run after generate_dataset.py.
"""
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

NU = 0.02
L  = 1.0
NX = 64


def spectral_antideriv(u, L):
    """Antiderivative integral_0^x u(x') dx', spectrally. Assumes zero-mean u."""
    nx = len(u)
    u_hat = np.fft.rfft(u)
    k = np.arange(len(u_hat), dtype=np.float64)
    int_hat = np.zeros_like(u_hat, dtype=complex)
    int_hat[1:] = u_hat[1:] / (1j * 2.0 * np.pi * k[1:] / L)
    return np.fft.irfft(int_hat, n=nx).real


def to_psi(u, nu, L):
    """Cole-Hopf forward: Burgers' u -> psi = log(phi), mean-centered.
    Centering keeps mean(psi)=0 across the domain; the constant cancels
    in u = -2*nu*psi_x so it has no effect on the recovered field.
    """
    psi = -1.0 / (2.0 * nu) * spectral_antideriv(u, L)
    return (psi - psi.mean()).astype(np.float32)


def traj_to_psi(traj, nu, L):
    """Cole-Hopf every snapshot of a trajectory. (T, nx) -> (T, nx)."""
    return np.stack([to_psi(traj[t], nu, L) for t in range(traj.shape[0])])


def main():
    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]

    train_traj = torch.load(os.path.join(DATA_DIR, "train_traj.pt"), weights_only=False).numpy()
    val_traj   = torch.load(os.path.join(DATA_DIR, "val_traj.pt"),   weights_only=False).numpy()
    test_traj  = torch.load(os.path.join(DATA_DIR, "test_traj.pt"),  weights_only=False).numpy()

    print("converting trajectories to psi = log(phi)...")
    train_psi = np.stack([traj_to_psi(t, NU, L) for t in train_traj])  # (80, 101, 64)
    val_psi   = np.stack([traj_to_psi(t, NU, L) for t in val_traj])    # (15, 101, 64)
    test_psi  = np.stack([traj_to_psi(t, NU, L) for t in test_traj])   # (15, 101, 64)

    print(f"psi range: [{train_psi.min():.2f}, {train_psi.max():.2f}]  "
          f"std={train_psi.std():.3f}")

    # test psi traj for rollout evaluation (keep alongside test_traj.pt for u)
    torch.save(torch.tensor(test_psi), os.path.join(DATA_DIR, "test_psi_traj.pt"))

    # single-step (psi_t, psi_{t+dt}) pairs
    def make_pairs(psi_trajs):
        samples = []
        for psi in psi_trajs:
            for t in range(psi.shape[0] - 1):
                psi_t    = torch.tensor(psi[t],   dtype=torch.float32).unsqueeze(-1)
                psi_next = torch.tensor(psi[t+1], dtype=torch.float32).unsqueeze(-1)
                x = torch.cat([psi_t, pos], dim=-1)
                samples.append(Data(x=x, edge_index=edge_index,
                                    edge_attr=edge_attr, y=psi_next))
        return samples

    train_pairs = make_pairs(train_psi)
    val_pairs   = make_pairs(val_psi)
    torch.save(train_pairs, os.path.join(DATA_DIR, "psi_train_pairs.pt"))
    torch.save(val_pairs,   os.path.join(DATA_DIR, "psi_val_pairs.pt"))
    print(f"psi pairs: train={len(train_pairs)}  val={len(val_pairs)}")


if __name__ == "__main__":
    main()
