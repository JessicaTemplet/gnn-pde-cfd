"""Run every config in validator/configs/ through the validator and merge
the computed reports with dashboard_meta.yaml's human-authored narrative,
producing one JSON file in the shape GNNPDEValidationUI.jsx expects.

This is what makes the dashboard show *computed* results instead of
hand-typed numbers: run this after training or re-validating anything, then
reload the dashboard (or re-publish it, if it's served as a static file).

Run from the repo root:
    python -m validator.build_dashboard_data
    python -m validator.build_dashboard_data --out validator/results/dashboard_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .cli import run_rollout_problem
from .config import REPO_ROOT, load_config
from .steady_state import run_steady_state

CONFIGS_DIR = REPO_ROOT / "validator" / "configs"
META_PATH = CONFIGS_DIR / "dashboard_meta.yaml"
DEFAULT_OUT = REPO_ROOT / "validator" / "results" / "dashboard_data.json"


def run_config(path: Path) -> dict:
    config = load_config(path)
    if config.problem_type == "rollout":
        return run_rollout_problem(config)
    elif config.problem_type == "steady_state":
        return run_steady_state(config)
    raise ValueError(f"{path}: unknown problem_type {config.problem_type!r}")


def build(configs_dir: Path = CONFIGS_DIR, meta_path: Path = META_PATH) -> list[dict]:
    with open(meta_path, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}

    entries = []
    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        if cfg_path == meta_path:
            continue
        print(f"running {cfg_path.name}...", file=sys.stderr)
        report = run_config(cfg_path)
        cp_id = report["id"]
        cp_meta = meta.get(cp_id, {})
        if cp_id not in meta:
            print(f"  warning: no dashboard_meta.yaml entry for {cp_id!r} -- "
                  f"tagline/note will be blank", file=sys.stderr)

        entries.append({
            "id": cp_id,
            "name": cp_meta.get("name", report["name"]),
            "stage": report["stage"],
            "tagline": cp_meta.get("tagline", ""),
            "status": report["status"],
            "metrics": report["metrics"],
            "checks": report["checks"],
            "note": cp_meta.get("note", ""),
            "series": report["series"],
            "image": None,
            "compareId": cp_meta.get("compareId"),
        })

    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-dir", default=str(CONFIGS_DIR))
    parser.add_argument("--meta", default=str(META_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    entries = build(Path(args.configs_dir), Path(args.meta))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)

    n_fail = sum(1 for e in entries if e["status"] == "fail")
    n_warn = sum(1 for e in entries if e["status"] == "warn")
    n_pass = sum(1 for e in entries if e["status"] == "pass")
    print(f"\nwrote {len(entries)} checkpoints to {out_path} "
          f"({n_pass} pass, {n_warn} warn, {n_fail} fail)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
