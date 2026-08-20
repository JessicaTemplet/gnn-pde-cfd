"""
Generate the hyperbolic-stage dataset: 1D periodic viscous Burgers'
trajectories from random smooth initial conditions, via the spectral
reference solver (spectral_solver.py).

Two things are saved:
  - single-step (u_t -> u_{t+dt}) training pairs, sampled from train/val
    trajectories, for training the GNN as a next-step predictor
  - full trajectories for train/val/test, for evaluating autoregressive
    rollout (the actual point of this stage - one-step accuracy and
    rollout stability are different things)

Run:
    python generate_dataset.py
"""
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from grid import build_periodic_chain_graph              # noqa: E402
from spectral_solver import solve_burgers, random_initial_condition  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

NX = 64
L = 1.0
NU = 0.02
DT_SOLVER = 0.0005
N_SOLVER_STEPS = 1000
SAVE_EVERY = 10          # -> dt_model = 0.005, 101 snapshots per trajectory (t=0..0.5)

N_TRAIN_TRAJ = 80
N_VAL_TRAJ = 15
N_TEST_TRAJ = 15
SEED = 0


def make_trajectories(n: int, rng: np.random.Generator):
    trajs = []
    for _ in range(n):
        u0 = random_initial_condition(NX, L, rng)
        traj = solve_burgers(u0, L, NU, DT_SOLVER, N_SOLVER_STEPS, save_every=SAVE_EVERY)
        trajs.append(traj.astype(np.float32))
    return np.stack(trajs)  # (n, n_snapshots, nx)


def make_pairs(trajs: np.ndarray, pos, edge_index, edge_attr):
    """Every consecutive (u_t, u_{t+1}) transition in every trajectory."""
    samples = []
    for traj in trajs:
        for t in range(traj.shape[0] - 1):
            u_t = torch.tensor(traj[t], dtype=torch.float32).unsqueeze(-1)
            u_next = torch.tensor(traj[t + 1], dtype=torch.float32).unsqueeze(-1)
            x = torch.cat([u_t, pos], dim=-1)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=u_next)
            samples.append(data)
    return samples


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pos, edge_index, edge_attr, dx = build_periodic_chain_graph(NX, L)
    dt_model = DT_SOLVER * SAVE_EVERY
    meta = dict(nx=NX, L=L, nu=NU, dx=dx, dt_model=dt_model,
                n_snapshots=N_SOLVER_STEPS // SAVE_EVERY + 1)
    torch.save(meta, os.path.join(DATA_DIR, "meta.pt"))
    torch.save(dict(pos=pos, edge_index=edge_index, edge_attr=edge_attr),
               os.path.join(DATA_DIR, "graph.pt"))

    train_traj = make_trajectories(N_TRAIN_TRAJ, rng)
    val_traj = make_trajectories(N_VAL_TRAJ, rng)
    test_traj = make_trajectories(N_TEST_TRAJ, rng)

    torch.save(torch.tensor(train_traj), os.path.join(DATA_DIR, "train_traj.pt"))
    torch.save(torch.tensor(val_traj), os.path.join(DATA_DIR, "val_traj.pt"))
    torch.save(torch.tensor(test_traj), os.path.join(DATA_DIR, "test_traj.pt"))
    print(f"trajectories: train {train_traj.shape} val {val_traj.shape} test {test_traj.shape}")

    train_pairs = make_pairs(train_traj, pos, edge_index, edge_attr)
    val_pairs = make_pairs(val_traj, pos, edge_index, edge_attr)
    torch.save(train_pairs, os.path.join(DATA_DIR, "train_pairs.pt"))
    torch.save(val_pairs, os.path.join(DATA_DIR, "val_pairs.pt"))
    print(f"single-step pairs: train {len(train_pairs)} val {len(val_pairs)}")


if __name__ == "__main__":
    main()
