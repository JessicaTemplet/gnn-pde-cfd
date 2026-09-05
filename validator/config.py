"""Config schema and YAML loader for the GNN-PDE validator.

Paths inside a config file are resolved relative to the GNNPDE_CFD repo
root (the parent of this package), not the config file's own location --
every existing per-stage script already refers to data/models this way
(e.g. "hyperbolic/data/graph.pt"), so configs read the same as the
existing per-stage READMEs and CLI scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


@dataclass
class ModelSpec:
    module: str          # .py file defining the model class, relative to repo root
    class_name: str
    checkpoint: str      # torch state_dict checkpoint, relative to repo root
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def module_path(self) -> Path:
        return resolve(self.module)

    @property
    def checkpoint_path(self) -> Path:
        return resolve(self.checkpoint)


@dataclass
class FunctionSpec:
    """Points at an external `fn(model, u0, pos, edge_index, edge_attr,
    n_steps) -> trajectory` adapter, for stages where the model's own
    .rollout() isn't the right entry point -- e.g. Cole-Hopf, which rolls
    out in a transformed coordinate and converts back to u each step."""
    module: str
    function: str

    @property
    def module_path(self) -> Path:
        return resolve(self.module)


@dataclass
class GeneralizationSet:
    tag: str
    test_traj: str
    graph: Optional[str] = None
    meta: Optional[str] = None

    def resolve_paths(self, primary_graph: str, primary_meta: str) -> dict[str, Path]:
        return {
            "test_traj": resolve(self.test_traj),
            "graph": resolve(self.graph or primary_graph),
            "meta": resolve(self.meta or primary_meta),
        }


@dataclass
class Thresholds:
    divergence_scale_factor: float = 20.0
    max_rel_l2_final: Optional[float] = None
    max_rel_l2_at: dict[float, float] = field(default_factory=dict)
    # e.g. 2.0 = "a generalization set's final error may not exceed 2x the
    # primary (in-distribution) test set's final error"
    generalization_degradation_factor: Optional[float] = None
    max_mean_conservation_residual: float = 0.05


@dataclass
class ValidationConfig:
    name: str
    stage: str
    problem_type: str          # "rollout" | "steady_state"
    pde_class: str
    model: ModelSpec
    data: dict[str, str]
    thresholds: Thresholds = field(default_factory=Thresholds)
    rollout_fn: Optional[FunctionSpec] = None
    # "N1" (default, matches every model's own .rollout()) or "N" (needed by
    # adapters that do 1D signal processing on u0, e.g. Cole-Hopf's FFT)
    rollout_u0_shape: str = "N1"
    generalization_sets: list[GeneralizationSet] = field(default_factory=list)


def load_config(path: str | Path) -> ValidationConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    model = ModelSpec(**raw["model"])
    thresholds = Thresholds(**raw.get("thresholds", {}))
    rollout_fn = FunctionSpec(**raw["rollout_fn"]) if raw.get("rollout_fn") else None
    gen_sets = [GeneralizationSet(**g) for g in raw.get("generalization_sets", [])]

    return ValidationConfig(
        name=raw["name"],
        stage=raw["stage"],
        problem_type=raw["problem_type"],
        pde_class=raw["pde_class"],
        model=model,
        data=raw["data"],
        thresholds=thresholds,
        rollout_fn=rollout_fn,
        rollout_u0_shape=raw.get("rollout_u0_shape", "N1"),
        generalization_sets=gen_sets,
    )
