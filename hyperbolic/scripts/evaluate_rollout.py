"""
Evaluate a trained Burgers' stepper via autoregressive rollout: start from
the test trajectories' initial conditions, run the model for as many steps
as the reference trajectory has, and compare. This is the metric that
actually matters for this stage - one-step MSE (train.py's validation loss)
says nothing about whether errors compound and the rollout stays stable.

Run:
    python evaluate_rollout.py [--tag v1] [--model-path ../models/best_model_v1.pt]
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
from model import BurgersStepGNN  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")


def rollout_all(model, test_traj, pos, edge_index, edge_attr):
    """test_traj: (n_traj, n_snap, nx). Returns predicted rollouts, same shape."""
    n_traj, n_snap, nx = test_traj.shape
    preds = torch.zeros_like(test_traj)
    for i in range(n_traj):
        u0 = test_traj[i, 0].unsqueeze(-1)
        with torch.no_grad():
            traj = model.rollout(u0, pos, edge_index, edge_attr, n_snap - 1)
        preds[i] = traj
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()
    model_path = args.model_path or os.path.join(MODEL_DIR, f"best_model_{args.tag}.pt")

    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    test_traj = torch.load(os.path.join(DATA_DIR, "test_traj.pt"), weights_only=False)
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)

    model = BurgersStepGNN(node_in_dim=2, edge_in_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    preds = rollout_all(model, test_traj, pos, edge_index, edge_attr)

    finite = torch.isfinite(preds)
    n_nan_inf = int((~finite.all(dim=(1, 2))).sum())
    preds_safe = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)
    # "diverged" is broader than NaN/Inf: float32 has headroom up into the
    # 1e30s, so a rollout can reach physically nonsensical magnitudes (1e5,
    # 1e6, ...) while every value is still technically finite. Flag anything
    # that ends up >20x the true field's max magnitude as diverged in
    # practice, even if it never hits a literal NaN.
    true_scale = float(test_traj.abs().max())
    final_mag_per_traj = preds_safe[:, -1, :].abs().amax(dim=-1)
    n_diverged = int((final_mag_per_traj > 20 * true_scale).sum())

    # per-step relative L2 error, averaged over trajectories (that didn't blow up)
    err = (preds_safe - test_traj).pow(2).sum(dim=-1).sqrt()          # (n_traj, n_snap)
    scale = test_traj.pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)       # (n_traj, n_snap)
    rel_err = err / scale
    mean_rel_err_per_step = rel_err.mean(dim=0).numpy()
    max_abs_val_per_step = preds_safe.abs().amax(dim=(0, 2)).numpy()

    dt_model = meta["dt_model"]
    t = np.arange(test_traj.shape[1]) * dt_model

    # --- plot: rollout error growth over time ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, mean_rel_err_per_step)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rollout time")
    axes[0].set_ylabel("mean relative L2 error vs. reference")
    axes[0].set_title(f"[{args.tag}] rollout error growth")

    axes[1].plot(t, max_abs_val_per_step, label="max |u_pred|")
    axes[1].axhline(float(test_traj.abs().max()), color="gray", linestyle="--", label="max |u_true| (all traj)")
    axes[1].set_xlabel("rollout time")
    axes[1].set_ylabel("max |u|")
    axes[1].set_title(f"[{args.tag}] prediction magnitude (blow-up check)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"rollout_error_{args.tag}.png"), dpi=140)
    plt.close(fig)

    # --- plot: space-time comparison for 2 test trajectories ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, i in enumerate([0, 1]):
        vmax = float(test_traj[i].abs().max())
        im0 = axes[row, 0].imshow(test_traj[i].numpy().T, origin="lower", aspect="auto",
                                   extent=[t[0], t[-1], 0, meta["L"]], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"traj {i}: reference")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(preds_safe[i].numpy().T, origin="lower", aspect="auto",
                                   extent=[t[0], t[-1], 0, meta["L"]], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f"[{args.tag}] GNN rollout")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        errmap = (preds_safe[i] - test_traj[i]).abs().numpy().T
        im2 = axes[row, 2].imshow(errmap, origin="lower", aspect="auto",
                                   extent=[t[0], t[-1], 0, meta["L"]], cmap="inferno")
        axes[row, 2].set_title("|error|")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        for ax in axes[row]:
            ax.set_xlabel("t")
            ax.set_ylabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"rollout_spacetime_{args.tag}.png"), dpi=140)
    plt.close(fig)

    summary = {
        "tag": args.tag,
        "n_test_traj": test_traj.shape[0],
        "n_nan_inf": n_nan_inf,
        "n_diverged_gt_20x_true_scale": n_diverged,
        "rel_err_at_25pct": float(mean_rel_err_per_step[len(t) // 4]),
        "rel_err_at_50pct": float(mean_rel_err_per_step[len(t) // 2]),
        "rel_err_at_final": float(mean_rel_err_per_step[-1]),
        "max_pred_magnitude": float(max_abs_val_per_step.max()),
        "max_true_magnitude": float(test_traj.abs().max()),
    }
    with open(os.path.join(RESULTS_DIR, f"rollout_summary_{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
