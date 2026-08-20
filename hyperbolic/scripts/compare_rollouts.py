"""
Overlay autoregressive rollout error curves for multiple model versions on
the same axes. Runs the full test-set rollout for each requested tag and
produces a single comparison PNG in results/.

Why not just read the existing rollout_summary JSON files? Those only store
3 scalar snapshots (25%, 50%, final). This script reruns the rollout to get
the full per-step curve, which shows *when* divergence starts -- critical for
distinguishing the v1 stall (error plateaus early) from the v3 blow-up
(error tracks v1/v2 for the first ~0.05 time units then diverges exponentially).

Run:
    python compare_rollouts.py v1 v2 v4
    python compare_rollouts.py v1 v2 v3 v4    # include the cautionary tale
    python compare_rollouts.py v2 v4           # just the comparison that matters
"""
import argparse
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

# Architecture must match every model being loaded.
# v1/v2/v3/v4 all use the same hidden_dim/n_layers -- if you add a model
# with different architecture, pass --hidden-dim and --n-layers explicitly.
DEFAULT_HIDDEN_DIM = 16
DEFAULT_N_LAYERS = 2


def rollout_all(model, test_traj, pos, edge_index, edge_attr):
    """Run full autoregressive rollout for every test trajectory.
    Returns predictions of shape (n_traj, n_snap, nx), clamped to finite."""
    n_traj, n_snap, nx = test_traj.shape
    preds = torch.zeros_like(test_traj)
    model.eval()
    for i in range(n_traj):
        u0 = test_traj[i, 0].unsqueeze(-1)
        with torch.no_grad():
            traj = model.rollout(u0, pos, edge_index, edge_attr, n_snap - 1)
        preds[i] = traj
    # replace NaN/Inf with a large sentinel so they show up on the magnitude plot
    preds = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)
    return preds


def compute_rel_err_curve(preds, test_traj):
    """Mean per-step relative L2 error, averaged over trajectories (n_snap,)."""
    err = (preds - test_traj).pow(2).sum(dim=-1).sqrt()           # (n_traj, n_snap)
    scale = test_traj.pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)   # (n_traj, n_snap)
    return (err / scale).mean(dim=0).numpy()                       # (n_snap,)


def compute_max_mag_curve(preds):
    """Max |u_pred| at each timestep across all trajectories (n_snap,)."""
    return preds.abs().amax(dim=(0, 2)).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="model version tags, e.g. v1 v2 v4")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--out", default=None,
                        help="output PNG path (default: results/compare_<tags>.png)")
    args = parser.parse_args()

    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    test_traj = torch.load(os.path.join(DATA_DIR, "test_traj.pt"), weights_only=False)
    meta = torch.load(os.path.join(DATA_DIR, "meta.pt"), weights_only=False)

    dt_model = meta["dt_model"]
    t = np.arange(test_traj.shape[1]) * dt_model
    true_scale = float(test_traj.abs().max())

    results = {}
    for tag in args.tags:
        model_path = os.path.join(MODEL_DIR, f"best_model_{tag}.pt")
        if not os.path.exists(model_path):
            print(f"[{tag}] model not found at {model_path}, skipping")
            continue
        model = BurgersStepGNN(
            node_in_dim=2, edge_in_dim=2,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        preds = rollout_all(model, test_traj, pos, edge_index, edge_attr)
        results[tag] = {
            "rel_err": compute_rel_err_curve(preds, test_traj),
            "max_mag": compute_max_mag_curve(preds),
        }
        final_err = results[tag]["rel_err"][-1]
        max_mag = results[tag]["max_mag"].max()
        print(f"[{tag}]  final_rel_L2={final_err:.4f}  max|pred|={max_mag:.2f}")

    if not results:
        print("no models found, nothing to plot")
        return

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for tag, d in results.items():
        axes[0].plot(t, d["rel_err"], label=tag)
        axes[1].plot(t, d["max_mag"], label=tag)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("rollout time")
    axes[0].set_ylabel("mean relative L2 error")
    axes[0].set_title("rollout error growth (lower is better)")
    axes[0].legend()

    axes[1].axhline(true_scale, color="gray", linestyle="--", linewidth=1,
                    label="max |u_true|")
    axes[1].set_xlabel("rollout time")
    axes[1].set_ylabel("max |u_pred|")
    axes[1].set_title("prediction magnitude (blow-up check)")
    axes[1].legend()

    fig.tight_layout()
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"compare_{'_'.join(args.tags)}.png"
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
