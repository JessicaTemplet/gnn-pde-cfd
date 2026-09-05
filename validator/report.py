"""Build a JSON-serializable report from check results, in the same shape
as the hand-curated CHECKPOINTS entries in ../GNNPDEValidationUI.jsx --
so that dashboard can eventually be pointed at computed output instead of
hand-typed numbers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _status(stability, accuracy) -> str:
    if not stability.passed:
        return "fail"
    if not accuracy.passed:
        return "warn"
    return "pass"


def build_rollout_report(config, stability, accuracy, generalization=None) -> dict[str, Any]:
    checks = [
        {"label": "No NaN/Inf over the rollout", "pass": stability.n_nan_inf == 0},
        {"label": f"No divergence (>{config.thresholds.divergence_scale_factor:g}x true scale)",
         "pass": stability.n_diverged == 0},
    ]
    if accuracy.threshold_failures:
        for msg in accuracy.threshold_failures:
            checks.append({"label": msg, "pass": False})
    else:
        checks.append({"label": "Accuracy thresholds met", "pass": True})

    metrics = [
        {"label": "final relative L2", "value": f"{accuracy.final_rel_err:.4f}"},
        {"label": "trajectories", "value": str(stability.n_trajectories)},
        {"label": "diverged", "value": f"{stability.n_diverged} / {stability.n_trajectories}"},
        {"label": "max |pred|", "value": f"{stability.max_abs_val_per_step.max():.3g}"},
    ]

    series = [[float(t), float(e)] for t, e in zip(accuracy.t, accuracy.mean_rel_err_per_step)]

    status = _status(stability, accuracy)

    report: dict[str, Any] = {
        "id": config.name,
        "name": config.name,
        "stage": config.stage,
        "status": status,
        "metrics": metrics,
        "checks": checks,
        "series": series,
        "raw": {
            "final_rel_err": accuracy.final_rel_err,
            "n_diverged": stability.n_diverged,
            "n_nan_inf": stability.n_nan_inf,
            "true_scale": stability.true_scale,
        },
    }

    if generalization is not None:
        report["generalization"] = [
            {
                "tag": s.tag,
                "final_rel_err": s.accuracy.final_rel_err,
                "n_diverged": s.stability.n_diverged,
                "degraded": s.degraded,
            }
            for s in generalization.sets
        ]
        if not generalization.passed:
            report["status"] = "fail" if status == "pass" else status
            report["checks"].append(
                {"label": "Generalization sets within degradation tolerance", "pass": False}
            )

    return report


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# What a failing check probably means, matched by substring against the
# check's label -- these are meant for someone new to this repo (or to GNN
# PDE surrogates generally) who doesn't yet know what "divergence" or
# "conservation residual" implies about the model's training.
_HINTS = [
    ("NaN/Inf", "The rollout produced non-finite values -- almost always a "
                "training instability (exploding gradients, too-high a learning "
                "rate, or too many autoregressive steps without an unrolled/"
                "curriculum loss). See the hyperbolic v3 story in the README "
                "for a worked example of exactly this failure."),
    ("divergence", "The rollout stayed finite but blew up far past the "
                    "reference field's scale -- same underlying cause as a NaN "
                    "(an unstable autoregressive step map), just slower to show "
                    "up. Check whether training used a fixed-horizon unrolled "
                    "loss (a fixed k can hide a per-step blow-up that only "
                    "shows up over more steps than it was trained on)."),
    ("relative L2 error", "The model is stable but not accurate enough against "
                           "the reference trajectory/solution to meet this "
                           "config's threshold -- not necessarily broken, just "
                           "under-trained or architecturally too small for this "
                           "problem. More capacity, more training, or (per the "
                           "Cole-Hopf result in the README) a better coordinate "
                           "representation of the target field can all help."),
    ("residual", "The predicted field doesn't satisfy the PDE itself (checked "
                 "independently of the reference solution) -- this can happen "
                 "even when accuracy-vs-reference looks fine, since matching "
                 "the reference pointwise doesn't guarantee the physics holds "
                 "locally. Consider adding or increasing a physics-residual "
                 "loss term during training."),
    ("Generalization", "The model's error grows too much on a held-out "
                        "dataset (different geometry/viscosity/boundary "
                        "conditions/etc. than training) relative to its "
                        "in-distribution test set -- a sign of overfitting to "
                        "the specific training distribution rather than "
                        "learning the underlying dynamics."),
]


def _hint_for(label: str) -> str | None:
    for needle, hint in _HINTS:
        if needle in label:
            return hint
    return None


def format_summary(report: dict) -> str:
    """Plain-English report for a human running this from the command line --
    the JSON report is for tooling (the dashboard, CI); this is for someone
    who just wants to know what happened and, if something's wrong, why.
    """
    lines = []
    lines.append(f"=== {report['name']} ({report['stage']}) ===")
    lines.append(f"Status: {report['status'].upper()}")
    lines.append("")

    lines.append("Metrics:")
    for m in report["metrics"]:
        lines.append(f"  {m['label']}: {m['value']}")
    lines.append("")

    lines.append("Checks:")
    failed = []
    for ck in report["checks"]:
        mark = "PASS" if ck["pass"] else "FAIL"
        lines.append(f"  [{mark}] {ck['label']}")
        if not ck["pass"]:
            failed.append(ck["label"])

    if report.get("generalization"):
        lines.append("")
        lines.append("Generalization sets:")
        for gs in report["generalization"]:
            mark = "DEGRADED" if gs["degraded"] else "OK"
            lines.append(f"  [{mark}] {gs['tag']}: final rel. L2 = {gs['final_rel_err']:.4f}, "
                          f"diverged {gs['n_diverged']}")
            if gs["degraded"]:
                failed.append(f"generalization set {gs['tag']!r} degraded")

    if failed:
        lines.append("")
        lines.append("Issues found:")
        seen_hints = set()
        for label in failed:
            hint = _hint_for(label)
            lines.append(f"  - {label}")
            if hint and hint not in seen_hints:
                lines.append(f"    -> {hint}")
                seen_hints.add(hint)
    else:
        lines.append("")
        lines.append("No issues found.")

    return "\n".join(lines)
