"""Rollout-stability checks: NaN/Inf detection and blow-up detection.

Blow-up is broader than a literal NaN/Inf -- float32 has headroom into the
1e30s, so a rollout can reach physically nonsensical magnitudes while every
value stays technically finite (this is exactly what happened in the
hyperbolic stage's v3 checkpoint, see the top-level README). A trajectory
is flagged diverged if its final-step magnitude exceeds
`divergence_scale_factor` times the true field's max magnitude -- the same
convention already used by this repo's per-stage evaluate_rollout.py scripts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StabilityResult:
    n_trajectories: int
    n_nan_inf: int
    n_diverged: int
    true_scale: float
    max_abs_val_per_step: np.ndarray   # (n_snap,)
    diverged_mask: np.ndarray          # (n_traj,) bool

    @property
    def passed(self) -> bool:
        return self.n_nan_inf == 0 and self.n_diverged == 0


def check_stability(preds: torch.Tensor, reference: torch.Tensor,
                     divergence_scale_factor: float = 20.0) -> StabilityResult:
    finite = torch.isfinite(preds)
    n_nan_inf = int((~finite.all(dim=(1, 2))).sum())
    preds_safe = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)

    true_scale = float(reference.abs().max())
    final_mag_per_traj = preds_safe[:, -1, :].abs().amax(dim=-1)
    diverged_mask = (final_mag_per_traj > divergence_scale_factor * true_scale).numpy()
    max_abs_val_per_step = preds_safe.abs().amax(dim=(0, 2)).numpy()

    return StabilityResult(
        n_trajectories=preds.shape[0],
        n_nan_inf=n_nan_inf,
        n_diverged=int(diverged_mask.sum()),
        true_scale=true_scale,
        max_abs_val_per_step=max_abs_val_per_step,
        diverged_mask=diverged_mask,
    )
