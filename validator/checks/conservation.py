"""Physics-conservation checks, one per PDE class.

Only PDE classes that actually have a working stage in this repo get a
real implementation -- the registry below is deliberately the extension
point for future stages, not a set of speculative checks with no data to
run them against yet. Adding real CFD (incompressible Navier-Stokes:
divergence-free velocity; compressible: mass/energy conservation; RANS:
k >= 0 / epsilon,omega > 0 realizability) is meant to be a new function
plus one registry entry, not a redesign.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from fd_operators import poisson_residual_scaled  # noqa: E402


@dataclass
class ConservationResult:
    name: str
    residual_mean_abs: float
    residual_max_abs: float
    passed: bool


def check_poisson_residual(u_pred: torch.Tensor, stencil_edge_index: torch.Tensor,
                            f: torch.Tensor, dx: float, dy: float,
                            max_mean_abs_residual: float = 0.05) -> ConservationResult:
    """Does the predicted field satisfy the discrete Poisson equation
    -Laplacian(u) = f, in the same scaled residual form used as the physics
    loss during training (fd_operators.poisson_residual_scaled, using the
    plain 4-connected stencil graph, not the multiscale message-passing
    graph)? This is independent of the accuracy-vs-FD-reference check: a
    field can match the reference well pointwise while still violating the
    PDE locally, or vice versa.
    """
    residual = poisson_residual_scaled(u_pred, stencil_edge_index, f, dx, dy)
    mean_abs = float(residual.abs().mean())
    max_abs = float(residual.abs().max())
    return ConservationResult(
        name="poisson_residual",
        residual_mean_abs=mean_abs,
        residual_max_abs=max_abs,
        passed=mean_abs <= max_mean_abs_residual,
    )


CONSERVATION_CHECKS: dict[str, Callable] = {
    "elliptic_poisson": check_poisson_residual,
}
