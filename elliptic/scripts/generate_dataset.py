"""
Generate a supervised dataset for the elliptic stage: 2D Poisson equation

    -Laplacian(u) = f  in the unit square
    u = 0               on the boundary  (homogeneous Dirichlet)

Each sample draws a random smooth source field f (a sum of a few Gaussian
bumps with random center/amplitude/width - a common synthetic-forcing recipe
for benchmarking neural PDE solvers). The reference solution is computed with
a direct sparse finite-difference solve, which is exact up to discretization
error and is what the GNN is trained to reproduce.

Run:
    python generate_dataset.py
"""
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from grid import build_grid_graph                  # noqa: E402
from fd_operators import build_poisson_operator    # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

NX, NY = 32, 32
N_TRAIN = 240
N_VAL = 30
N_TEST = 30
SEED = 0


def random_source(xx: np.ndarray, yy: np.ndarray, rng: np.random.Generator,
                   n_bumps_range=(2, 5)) -> np.ndarray:
    """Sum of random Gaussian bumps, a standard synthetic forcing term."""
    f = np.zeros_like(xx)
    n_bumps = rng.integers(*n_bumps_range, endpoint=True)
    for _ in range(n_bumps):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        amp = rng.uniform(-8.0, 8.0)
        width = rng.uniform(0.04, 0.12)
        f += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width**2))
    return f


def make_split(n_samples: int, pos, edge_index, edge_attr, stencil_edge_index, boundary_mask,
                dx, dy, shape, rng: np.random.Generator):
    ny, nx = shape
    xx = pos[:, 0].numpy().reshape(ny, nx)
    yy = pos[:, 1].numpy().reshape(ny, nx)
    b_mask_np = boundary_mask.numpy()
    bc_value = np.zeros(nx * ny, dtype=np.float64)  # homogeneous Dirichlet
    solve = build_poisson_operator(nx, ny, dx, dy, b_mask_np)

    samples = []
    for _ in range(n_samples):
        f = random_source(xx, yy, rng).ravel()
        u = solve(f, bc_value)

        node_x = np.stack([
            f,
            b_mask_np.astype(np.float64),
            bc_value,
            pos[:, 0].numpy(),
            pos[:, 1].numpy(),
        ], axis=1)

        data = Data(
            x=torch.tensor(node_x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            stencil_edge_index=stencil_edge_index,
            y=torch.tensor(u, dtype=torch.float32).unsqueeze(-1),
            pos=pos,
            boundary_mask=boundary_mask,
        )
        samples.append(data)
    return samples


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pos, edge_index, edge_attr, stencil_edge_index, boundary_mask, dx, dy, shape = build_grid_graph(NX, NY)
    print(f"grid: {NX}x{NY} nodes, {edge_index.shape[1]} multiscale edges, "
          f"{stencil_edge_index.shape[1]} stencil edges")

    meta = dict(nx=NX, ny=NY, dx=dx, dy=dy, shape=shape)
    torch.save(meta, os.path.join(DATA_DIR, "grid_meta.pt"))
    torch.save(
        dict(pos=pos, edge_index=edge_index, edge_attr=edge_attr,
             stencil_edge_index=stencil_edge_index, boundary_mask=boundary_mask),
        os.path.join(DATA_DIR, "grid_graph.pt"),
    )

    for split_name, n in [("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)]:
        samples = make_split(n, pos, edge_index, edge_attr, stencil_edge_index, boundary_mask,
                              dx, dy, shape, rng)
        torch.save(samples, os.path.join(DATA_DIR, f"{split_name}.pt"))
        print(f"{split_name}: {n} samples saved")


if __name__ == "__main__":
    main()
