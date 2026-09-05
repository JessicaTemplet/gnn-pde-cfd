"""Dynamic loading of a model class + checkpoint from file paths, and of
optional external rollout-adapter functions (e.g. Cole-Hopf's coordinate
transform) -- both by file path rather than dotted module name.

Every PDE stage in this repo has its own scripts/model.py, and none of the
stage directories are real Python packages (each is a standalone scripts/
folder meant to be run with its own cwd, per the top-level README). Loading
by file path with a unique synthetic module name sidesteps that: two stages
can both define scripts/model.py without colliding in sys.modules.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Callable

import torch

from .config import FunctionSpec, ModelSpec


def _load_module(file_path: Path) -> types.ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"module file not found: {file_path}")
    name = f"_gnncfd_validator_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    # PyTorch Geometric's MessagePassing.__init__ introspects `message()`'s
    # type hints via sys.modules[module.__name__].__dict__, so the module
    # must be registered there before exec -- a plain module_from_spec
    # without this line loads fine but breaks the first MessagePassing
    # subclass instantiated from it with a KeyError.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model(spec: ModelSpec) -> torch.nn.Module:
    module = _load_module(spec.module_path)
    cls = getattr(module, spec.class_name)
    model = cls(**spec.init_kwargs)
    state_dict = torch.load(spec.checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_rollout_fn(spec: FunctionSpec) -> Callable:
    module = _load_module(spec.module_path)
    return getattr(module, spec.function)


def default_rollout_fn(model: torch.nn.Module) -> Callable:
    """Wraps model.rollout(...) so it matches the uniform
    fn(model, u0, pos, edge_index, edge_attr, n_steps) signature used by
    external rollout adapters (see FunctionSpec) -- callers don't need to
    know whether a given config uses the model's own method or an adapter.
    """
    def _fn(m, u0, pos, edge_index, edge_attr, n_steps):
        return m.rollout(u0, pos, edge_index, edge_attr, n_steps)
    return _fn
