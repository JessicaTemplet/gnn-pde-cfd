"""Accuracy-vs-ground-truth checks: per-step relative L2 error over a
rollout, plus optional pass/fail thresholds at specific times."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class AccuracyResult:
    t: np.ndarray                      # (n_snap,) rollout time
    mean_rel_err_per_step: np.ndarray  # (n_snap,)
    final_rel_err: float
    threshold_failures: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.threshold_failures) == 0


def check_accuracy(preds: torch.Tensor, reference: torch.Tensor, dt: float,
                    max_rel_l2_final: Optional[float] = None,
                    max_rel_l2_at: Optional[dict] = None) -> AccuracyResult:
    preds_safe = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)
    err = (preds_safe - reference).pow(2).sum(dim=-1).sqrt()
    scale = reference.pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)
    rel_err = (err / scale).mean(dim=0).numpy()

    n_snap = reference.shape[1]
    t = np.arange(n_snap) * dt
    final_rel_err = float(rel_err[-1])

    failures = []
    if max_rel_l2_final is not None and final_rel_err > max_rel_l2_final:
        failures.append(
            f"final relative L2 error {final_rel_err:.4f} exceeds threshold {max_rel_l2_final}"
        )

    for t_check, max_err in (max_rel_l2_at or {}).items():
        idx = min(int(round(t_check / dt)), n_snap - 1)
        if rel_err[idx] > max_err:
            failures.append(
                f"relative L2 error at t={t_check} ({rel_err[idx]:.4f}) exceeds threshold {max_err}"
            )

    return AccuracyResult(t=t, mean_rel_err_per_step=rel_err, final_rel_err=final_rel_err,
                           threshold_failures=failures)
