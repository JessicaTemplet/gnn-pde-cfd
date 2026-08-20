"""
Cole-Hopf v2 rollout: Burgers' dynamics via log-phi GNN.

The fix for v1's positivity failure: instead of training the heat GNN to
predict phi (which must be positive) and then clamping/shifting to enforce
it, we train a GNN directly in log-phi space (psi = log(phi)) and roll out
there. Recovering u from psi is:

    u = -2*nu * psi_x          (spectral derivative, no division, no positivity constraint)

because phi = exp(psi), so phi_x = exp(psi)*psi_x, and phi_x/phi = psi_x.

The model was trained on Cole-Hopf transforms of actual Burgers' trajectories
at nu=0.02, so the nu mismatch from v1 is also gone.

Run:
    python cole_hopf_v2_rollout.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import BurgersStepGNN  # noqa: E402

HYPER_DATA  = os.path.join(HERE, "..", "data")
MODEL_PATH  = os.path.join(HERE, "..", "models", "best_log_phi_model.pt")
RESULTS_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NU         = 0.02
L          = 1.0
HIDDEN_DIM = 16
N_LAYERS   = 2


def spectral_antideriv(u, L):
    """Spectral antiderivative integral_0^x u(x') dx'. Assumes zero-mean u."""
    nx = u.shape[0]
    u_hat = torch.fft.rfft(u.double())
    k = torch.arange(u_hat.shape[0], dtype=torch.float64, device=u.device)
    int_hat = torch.zeros_like(u_hat)
    int_hat[1:] = u_hat[1:] / (1j * 2.0 * torch.pi * k[1:] / L)
    return torch.fft.irfft(int_hat, n=nx).real.float()


def spectral_deriv(psi, L):
    """Spectral d/dx psi."""
    nx = psi.shape[0]
    psi_hat = torch.fft.rfft(psi.double())
    k = torch.arange(psi_hat.shape[0], dtype=torch.float64, device=psi.device)
    psi_x_hat = 1j * 2.0 * torch.pi * k / L * psi_hat
    return torch.fft.irfft(psi_x_hat, n=nx).real.float()


def to_psi(u, nu, L):
    """Cole-Hopf forward: u -> psi = log(phi), mean-centered."""
    psi = -1.0 / (2.0 * nu) * spectral_antideriv(u, L)
    return psi - psi.mean()


def to_u(psi, nu, L):
    """Cole-Hopf inverse: psi -> u via spectral derivative. No division."""
    return -2.0 * nu * spectral_deriv(psi, L)


def rollout(model, u0, pos, edge_index, edge_attr, n_steps):
    """Roll out n_steps of Burgers' via Cole-Hopf + log-phi GNN.
    Returns (n_steps+1, NX) tensor of u values.
    """
    psi = to_psi(u0, NU, L)
    traj = [u0.clone()]
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            x   = torch.cat([psi.unsqueeze(-1), pos], dim=-1)
            psi = model(x, edge_index, edge_attr).squeeze(-1)
            # no positivity enforcement needed -- psi is unconstrained
            # re-center each step to prevent slow drift in the mean
            psi = psi - psi.mean()
            u   = to_u(psi, NU, L)
            traj.append(u.clone())
    return torch.stack(traj)


def main():
    graph = torch.load(os.path.join(HYPER_DATA, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    test_traj = torch.load(os.path.join(HYPER_DATA, "test_traj.pt"), weights_only=False)
    meta      = torch.load(os.path.join(HYPER_DATA, "meta.pt"),      weights_only=False)

    model = BurgersStepGNN(node_in_dim=2, edge_in_dim=2,
                           hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    n_traj, n_snap, nx = test_traj.shape
    n_steps = n_snap - 1
    preds = torch.zeros_like(test_traj)

    for i in range(n_traj):
        preds[i] = rollout(model, test_traj[i, 0], pos, edge_index, edge_attr, n_steps)

    # metrics
    finite    = torch.isfinite(preds)
    n_nan_inf = int((~finite.all(dim=(1, 2))).sum())
    preds_s   = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)
    true_scale = float(test_traj.abs().max())
    n_diverged = int((preds_s[:, -1].abs().amax(dim=-1) > 20 * true_scale).sum())

    err     = (preds_s - test_traj).pow(2).sum(dim=-1).sqrt()
    scale   = test_traj.pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)
    rel_err = (err / scale).mean(dim=0).numpy()
    mag     = preds_s.abs().amax(dim=(0, 2)).numpy()

    dt_model = meta["dt_model"]
    t = np.arange(n_snap) * dt_model

    # plot: error + magnitude
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, rel_err, color="tab:purple", label="Cole-Hopf v2 (log-phi GNN)")
    axes[0].set_xlabel("rollout time")
    axes[0].set_ylabel("mean relative L2 error")
    axes[0].set_title("rollout error (Cole-Hopf v2)")
    axes[0].legend()

    axes[1].plot(t, mag, color="tab:purple", label="max |u_pred|")
    axes[1].axhline(true_scale, color="gray", linestyle="--", label="max |u_true|")
    axes[1].set_xlabel("rollout time")
    axes[1].set_ylabel("max |u|")
    axes[1].set_title("magnitude check")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rollout_error_cole_hopf_v2.png"), dpi=140)
    plt.close(fig)

    # space-time comparison for 2 test trajectories
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, i in enumerate([0, 1]):
        vmax = float(test_traj[i].abs().max())
        panels = [
            (test_traj[i].numpy(), "reference",          "RdBu_r"),
            (preds_s[i].numpy(),   "Cole-Hopf v2 GNN",  "RdBu_r"),
            (np.abs(preds_s[i].numpy() - test_traj[i].numpy()), "|error|", "inferno"),
        ]
        for col, (data, title, cmap) in enumerate(panels):
            kw = dict(origin="lower", aspect="auto", extent=[t[0], t[-1], 0, L])
            im = axes[row, col].imshow(
                data.T, cmap=cmap,
                **({"vmin": -vmax, "vmax": vmax} if col < 2 else {}), **kw)
            axes[row, col].set_title(f"traj {i}: {title}")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            axes[row, col].set_xlabel("t")
            axes[row, col].set_ylabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rollout_spacetime_cole_hopf_v2.png"), dpi=140)
    plt.close(fig)

    summary = {
        "method":                      "cole_hopf_v2_log_phi_gnn",
        "nu":                          NU,
        "nu_mismatch":                 "none -- model trained directly on psi from Burgers' trajectories",
        "n_test_traj":                 n_traj,
        "n_nan_inf":                   n_nan_inf,
        "n_diverged_gt_20x_true_scale": n_diverged,
        "rel_err_at_25pct":            float(rel_err[len(t) // 4]),
        "rel_err_at_50pct":            float(rel_err[len(t) // 2]),
        "rel_err_at_final":            float(rel_err[-1]),
        "max_pred_magnitude":          float(mag.max()),
        "max_true_magnitude":          true_scale,
    }
    with open(os.path.join(RESULTS_DIR, "rollout_summary_cole_hopf_v2.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
