"""Generalization checks: re-run the rollout+accuracy+stability pipeline
against tagged held-out datasets (unseen initial conditions, viscosity,
geometry, boundary conditions, ...) and compare against the primary
in-distribution test set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .accuracy import AccuracyResult, check_accuracy
from .rollout import run_rollout
from .stability import StabilityResult, check_stability


@dataclass
class GeneralizationSetResult:
    tag: str
    stability: StabilityResult
    accuracy: AccuracyResult
    degraded: bool


@dataclass
class GeneralizationResult:
    sets: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(not s.degraded for s in self.sets)


def check_generalization(rollout_fn: Callable, model,
                          primary_final_err: float,
                          degradation_factor: Optional[float],
                          divergence_scale_factor: float, dt: float,
                          gen_sets: list, u0_shape: str = "N1") -> GeneralizationResult:
    """gen_sets: list of dicts with resolved paths + tag, i.e.
    {"tag": ..., "test_traj": Path, "graph": Path, "meta": Path}."""
    results = []
    for gs in gen_sets:
        test_traj = torch.load(gs["test_traj"], weights_only=False)
        graph = torch.load(gs["graph"], weights_only=False)
        pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]

        preds = run_rollout(rollout_fn, model, test_traj, pos, edge_index, edge_attr, u0_shape=u0_shape)
        stability = check_stability(preds, test_traj, divergence_scale_factor)
        accuracy = check_accuracy(preds, test_traj, dt)

        degraded = not stability.passed
        if degradation_factor is not None and primary_final_err is not None:
            degraded = degraded or accuracy.final_rel_err > degradation_factor * primary_final_err

        results.append(GeneralizationSetResult(tag=gs["tag"], stability=stability,
                                                accuracy=accuracy, degraded=degraded))

    return GeneralizationResult(sets=results)
