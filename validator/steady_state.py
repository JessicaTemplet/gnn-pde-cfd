"""Validation pipeline for steady_state problems (currently: elliptic).

Different shape from the rollout pipeline: no time axis, one solve per
test sample, and the model's forward signature takes extra per-sample
tensors (boundary_mask, bc_value) rather than a single autoregressive step
-- so this isn't unified with checks/rollout.py, it's a parallel pipeline
that reuses the same conservation-check registry.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .checks.conservation import CONSERVATION_CHECKS
from .config import resolve
from .model_loader import load_model


def run_steady_state(config) -> dict[str, Any]:
    model = load_model(config.model)
    test_set = torch.load(resolve(config.data["test_set"]), weights_only=False)
    grid_meta = torch.load(resolve(config.data["grid_meta"]), weights_only=False)
    dx, dy = grid_meta["dx"], grid_meta["dy"]

    conservation_check = CONSERVATION_CHECKS.get(config.pde_class)
    max_mean_residual = config.thresholds.max_mean_conservation_residual

    rel_errs = []
    residual_means = []
    for data in test_set:
        bc_value = data.x[:, 2]
        with torch.no_grad():
            u_pred = model(data.x, data.edge_index, data.edge_attr, data.boundary_mask, bc_value)
        u_true = data.y.squeeze(-1)
        rel_err = float((u_pred.squeeze(-1) - u_true).norm() / (u_true.norm() + 1e-12))
        rel_errs.append(rel_err)

        if conservation_check is not None:
            f = data.x[:, 0]
            result = conservation_check(u_pred.squeeze(-1), data.stencil_edge_index, f, dx, dy,
                                         max_mean_abs_residual=max_mean_residual)
            residual_means.append(result.residual_mean_abs)

    rel_errs = np.array(rel_errs)
    n_nan = int(np.sum(~np.isfinite(rel_errs)))

    status = "pass"
    checks = [{"label": "No NaN/Inf in predictions", "pass": n_nan == 0}]
    if n_nan > 0:
        status = "fail"

    threshold_failures = []
    if config.thresholds.max_rel_l2_final is not None and float(rel_errs.mean()) > config.thresholds.max_rel_l2_final:
        threshold_failures.append(
            f"mean relative L2 error {rel_errs.mean():.4f} exceeds threshold {config.thresholds.max_rel_l2_final}"
        )
    if threshold_failures:
        for msg in threshold_failures:
            checks.append({"label": msg, "pass": False})
        status = "warn" if status != "fail" else status
    else:
        checks.append({"label": "Accuracy thresholds met", "pass": True})

    metrics = [
        {"label": "test rel. L2 (mean)", "value": f"{rel_errs.mean():.3f}"},
        {"label": "test rel. L2 (median)", "value": f"{np.median(rel_errs):.3f}"},
        {"label": "test rel. L2 (range)", "value": f"{rel_errs.min():.2f}–{rel_errs.max():.2f}"},
        {"label": "n_test", "value": str(len(test_set))},
    ]

    if residual_means:
        residual_arr = np.array(residual_means)
        metrics.append({"label": "PDE residual mean |.|", "value": f"{residual_arr.mean():.4g}"})
        residual_ok = bool(residual_arr.mean() <= max_mean_residual)
        checks.append({"label": "PDE residual within tolerance", "pass": residual_ok})
        if not residual_ok:
            status = "fail"

    return {
        "id": config.name,
        "name": config.name,
        "stage": config.stage,
        "status": status,
        "metrics": metrics,
        "checks": checks,
        "series": None,
        "raw": {
            "rel_l2_mean": float(rel_errs.mean()),
            "rel_l2_median": float(np.median(rel_errs)),
            "rel_l2_max": float(rel_errs.max()),
            "rel_l2_min": float(rel_errs.min()),
        },
    }
