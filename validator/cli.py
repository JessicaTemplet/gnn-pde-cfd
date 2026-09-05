"""Command-line entry point: run all applicable checks for one config and
report the result.

    python -m validator.cli --config validator/configs/hyperbolic_v4.yaml
    python -m validator.cli --config validator/configs/elliptic.yaml --out validator/results/elliptic.json
    python -m validator.cli --config validator/configs/hyperbolic_v4.yaml --json

Default output is a plain-English pass/fail summary with an explanation of
any failing check (see report.format_summary) -- meant to be read directly
by whoever just trained the model, not parsed. Pass --json for the raw
report instead (what the dashboard aggregator consumes). --out always
writes the JSON report to a file regardless of which one printed to stdout.

Exit code is 1 if the report status is "fail", 0 otherwise (including "warn"),
so this can be dropped into a CI gate on the hard failures (divergence,
NaN, conservation violation) without blocking on softer accuracy warnings.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from .checks.accuracy import check_accuracy
from .checks.generalization import check_generalization
from .checks.rollout import run_rollout
from .checks.stability import check_stability
from .config import load_config, resolve
from .model_loader import default_rollout_fn, load_model, load_rollout_fn
from .report import build_rollout_report, format_summary, write_report
from .steady_state import run_steady_state


def run_rollout_problem(config) -> dict:
    model = load_model(config.model)
    rollout_fn = load_rollout_fn(config.rollout_fn) if config.rollout_fn else default_rollout_fn(model)

    graph = torch.load(resolve(config.data["graph"]), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    test_traj = torch.load(resolve(config.data["test_traj"]), weights_only=False)
    meta = torch.load(resolve(config.data["meta"]), weights_only=False)
    dt = meta["dt_model"]

    preds = run_rollout(rollout_fn, model, test_traj, pos, edge_index, edge_attr,
                         u0_shape=config.rollout_u0_shape)
    stability = check_stability(preds, test_traj, config.thresholds.divergence_scale_factor)
    accuracy = check_accuracy(preds, test_traj, dt,
                               config.thresholds.max_rel_l2_final,
                               config.thresholds.max_rel_l2_at)

    generalization = None
    if config.generalization_sets:
        gen_sets = []
        for gs in config.generalization_sets:
            paths = gs.resolve_paths(config.data["graph"], config.data["meta"])
            gen_sets.append({"tag": gs.tag, **paths})
        generalization = check_generalization(
            rollout_fn, model, accuracy.final_rel_err,
            config.thresholds.generalization_degradation_factor,
            config.thresholds.divergence_scale_factor, dt, gen_sets,
            u0_shape=config.rollout_u0_shape,
        )

    return build_rollout_report(config, stability, accuracy, generalization=generalization)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a trained GNN-PDE model against a config.")
    parser.add_argument("--config", required=True, help="path to a validator YAML config")
    parser.add_argument("--out", default=None, help="optional path to write the JSON report")
    parser.add_argument("--json", action="store_true", help="print the raw JSON report instead of a plain-English summary")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if config.problem_type == "rollout":
        report = run_rollout_problem(config)
    elif config.problem_type == "steady_state":
        report = run_steady_state(config)
    else:
        raise ValueError(f"unknown problem_type: {config.problem_type!r} (expected 'rollout' or 'steady_state')")

    print(json.dumps(report, indent=2) if args.json else format_summary(report))
    if args.out:
        write_report(report, args.out)
        print(f"\nreport written to {args.out}", file=sys.stderr)

    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
