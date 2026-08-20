"""
Evaluate the trained heat-equation GNN via autoregressive rollout.

Same evaluation structure as the hyperbolic stage: start from test trajectory
initial conditions, run the model autoregressively for the full t=0..0.5
window, and compare against the reference solver.

The key question this evaluation answers for the parabolic hypothesis: does
one-step supervised training produce a rollout-stable model for the heat
equation, or does it fail the way hyperbolic v1 did? Because diffusion damps
errors rather than advecting them, the expectation is that the rollout is
stable (relative L2 stays bounded and may even decrease over the trajectory
as diffusion relaxes the field toward zero). This contrasts with Burgers'
v1 where the shock stalled and error climbed monotonically.

A stable rollout here would be strong evidence that the instability in the
hyperbolic case was specific to hyperbolic propagation, not a fundamental
problem with one-step GNN training.

Run:
    python evaluate_rollout.py
    python evaluate_rollout.py --model-path ../models/checkpoint.pt
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from model import HeatStepGNN  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")


def rollout_all(model, test_traj, pos, edge_index, edge_attr):
    """Rollout all test trajectories. Returns predictions (n_traj, n_snap, nx)."""
    n_traj, n_snap, nx = test_traj.shape
    preds = torch.zeros_like(test_traj)
    model.eval()
    for i in range(n_traj):
        u0 = test_traj[i, 0].unsqueeze(-1)
        with torch.no_grad():
            traj = model.rollout(u0, pos, edge_index, edge_attr, n_snap - 1)
        preds[i] = traj
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="path to model weights (default: models/best_model.pt)")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()
    model_path = args.model_path or os.path.join(MODEL_DIR, "best_model.pt")

    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    test_traj = torch.load(os.path.join(DATA_DIR, "test_traj.pt"), weights_only=False)
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)

    model = HeatStepGNN(
        node_in_dim=2, edge_in_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))

    preds = rollout_all(model, test_traj, pos, edge_index, edge_attr)

    finite = torch.isfinite(preds)
    n_nan_inf = int((~finite.all(dim=(1, 2))).sum())
    preds_safe = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)
    true_scale = float(test_traj.abs().max())
    final_mag_per_traj = preds_safe[:, -1, :].abs().amax(dim=-1)
    n_diverged = int((final_mag_per_traj > 20 * true_scale).sum())

    err = (preds_safe - test_traj).pow(2).sum(dim=-1).sqrt()
    scale = test_traj.pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)
    rel_err = err / scale
    mean_rel_err_per_step = rel_err.mean(dim=0).numpy()
    max_abs_val_per_step = preds_safe.abs().amax(dim=(0, 2)).numpy()

    dt_model = meta["dt_model"]
    t = np.arange(test_traj.shape[1]) * dt_model

    # --- plot: rollout error + magnitude ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, mean_rel_err_per_step)
    axes[0].set_xlabel("rollout time")
    axes[0].set_ylabel("mean relative L2 error vs. reference")
    axes[0].set_title("rollout error over time")
    # note: no log scale here by default - diffusion should keep error bounded,
    # and a log scale would compress what may be an interesting flat/declining curve

    axes[1].plot(t, max_abs_val_per_step, label="max |u_pred|")
    axes[1].axhline(true_scale, color="gray", linestyle="--", label="max |u_true| (all traj)")
    axes[1].set_xlabel("rollout time")
    axes[1].set_ylabel("max |u|")
    axes[1].set_title("prediction magnitude (blow-up check)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rollout_error.png"), dpi=140)
    plt.close(fig)

    # --- plot: space-time comparison for 2 test trajectories ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, i in enumerate([0, 1]):
        vmax = float(test_traj[i].abs().max())
        im0 = axes[row, 0].imshow(
            test_traj[i].numpy().T, origin="lower", aspect="auto",
            extent=[t[0], t[-1], 0, meta["L"]], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        axes[row, 0].set_title(f"traj {i}: reference")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(
            preds_safe[i].numpy().T, origin="lower", aspect="auto",
            extent=[t[0], t[-1], 0, meta["L"]], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        axes[row, 1].set_title("GNN rollout")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        errmap = (preds_safe[i] - test_traj[i]).abs().numpy().T
        im2 = axes[row, 2].imshow(
            errmap, origin="lower", aspect="auto",
            extent=[t[0], t[-1], 0, meta["L"]], cmap="inferno",
        )
        axes[row, 2].set_title("|error|")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        for ax in axes[row]:
            ax.set_xlabel("t")
            ax.set_ylabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rollout_spacetime.png"), dpi=140)
    plt.close(fig)

    summary = {
        "n_test_traj": test_traj.shape[0],
        "n_nan_inf": n_nan_inf,
        "n_diverged_gt_20x_true_scale": n_diverged,
        "rel_err_at_25pct": float(mean_rel_err_per_step[len(t) // 4]),
        "rel_err_at_50pct": float(mean_rel_err_per_step[len(t) // 2]),
        "rel_err_at_final": float(mean_rel_err_per_step[-1]),
        "max_pred_magnitude": float(max_abs_val_per_step.max()),
        "max_true_magnitude": true_scale,
    }
    with open(os.path.join(RESULTS_DIR, "rollout_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
