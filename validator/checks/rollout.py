"""Run an autoregressive rollout for every test trajectory, using either a
model's own .rollout(...) method or an external rollout-adapter function
with the same signature (see model_loader.load_rollout_fn / default_rollout_fn).
"""
from __future__ import annotations

from typing import Callable

import torch


def run_rollout(rollout_fn: Callable, model, test_traj: torch.Tensor,
                 pos: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                 u0_shape: str = "N1") -> torch.Tensor:
    """test_traj: (n_traj, n_snap, nx) scalar field per node. Returns
    predictions of the same shape, one rollout per trajectory's own initial
    condition.

    u0_shape controls whether the per-trajectory initial condition is
    handed to rollout_fn as (nx, 1) ("N1", the convention every model's own
    .rollout() uses, matching pos's (nx, 1) shape for concatenation) or
    plain (nx,) ("N", needed by external adapters that do 1D signal
    processing on u0 directly -- e.g. the Cole-Hopf adapter's spectral
    antiderivative/derivative, which needs a genuinely 1D tensor for
    torch.fft.rfft along the last axis).
    """
    n_traj, n_snap, nx = test_traj.shape
    preds = torch.zeros_like(test_traj)
    for i in range(n_traj):
        u0 = test_traj[i, 0]
        if u0_shape == "N1":
            u0 = u0.unsqueeze(-1)
        elif u0_shape != "N":
            raise ValueError(f"unknown u0_shape: {u0_shape!r} (expected 'N1' or 'N')")
        with torch.no_grad():
            traj = rollout_fn(model, u0, pos, edge_index, edge_attr, n_snap - 1)
        preds[i] = traj
    return preds
