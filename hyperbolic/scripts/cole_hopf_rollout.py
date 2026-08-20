"""
Experiment: roll out Burgers' dynamics via Cole-Hopf + trained heat GNN.

The Cole-Hopf transformation is an exact mapping from viscous Burgers' to
the linear heat equation: if phi solves phi_t = nu*phi_xx, then
u = -2*nu * phi_x / phi solves u_t + u*u_x = nu*u_xx (Burgers'). So in
principle, if you have a good solver for the heat equation, you can solve
Burgers' for free by transforming forward, stepping, transforming back.

We have a trained heat GNN (parabolic stage). This script asks: does plugging
it into the Cole-Hopf pipeline produce a usable Burgers' rollout?

Known approximation sources going in:
  1. nu mismatch: heat GNN trained at nu=0.01; Burgers' stage uses nu=0.02.
     Cole-Hopf requires phi to evolve at the same nu as Burgers'. The model
     will under-decay high modes in phi-space by a factor of ~exp(0.01*k^2*dt)
     per step. For k=1 over 100 steps: ~0.5% underdecay total. For k=8: ~27%
     underdecay total. Low-mode dynamics should survive; high-frequency
     content will accumulate.
  2. GNN approximation error: the heat model isn't a perfect heat solver.
     Each step introduces a small error in phi; Cole-Hopf inverse amplifies
     errors near zero crossings of phi (where phi is small and phi_x/phi blows).
  3. phi positivity: the GNN outputs unconstrained values; we enforce
     positivity by shifting (subtract min + epsilon) before normalizing.
     This preserves the spatial gradient structure that matters for to_u.

The initial conditions are zero-mean by construction (sum of sine modes),
so Cole-Hopf is exact on the periodic domain without any mean correction.

Run:
    python cole_hopf_rollout.py
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
# import HeatStepGNN from parabolic stage
sys.path.insert(0, os.path.join(HERE, "..", "..", "parabolic", "scripts"))
from model import HeatStepGNN  # noqa: E402

HYPER_DATA  = os.path.join(HERE, "..", "data")
PARA_MODEL  = os.path.join(HERE, "..", "..", "parabolic", "models", "best_model.pt")
RESULTS_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NU_BURGERS    = 0.02   # Burgers' viscosity -- used in both transforms
NU_HEAT_MODEL = 0.01   # what the heat GNN was actually trained at (documented mismatch)
L             = 1.0
NX            = 64
HIDDEN_DIM    = 16
N_LAYERS      = 2


# ---- Cole-Hopf transforms (spectral, double precision for numerical stability) ----

def spectral_antideriv(u, L):
    """Antiderivative int_0^x u(x') dx' via spectral division by ik.
    Assumes zero-mean u (k=0 component is zero). Returns real-valued result."""
    nx = u.shape[0]
    u_hat = torch.fft.rfft(u.double())
    k = torch.arange(u_hat.shape[0], dtype=torch.float64, device=u.device)
    int_hat = torch.zeros_like(u_hat)
    int_hat[1:] = u_hat[1:] / (1j * 2.0 * torch.pi * k[1:] / L)
    return torch.fft.irfft(int_hat, n=nx).real.float()


def spectral_deriv(u, L):
    """Spectral d/dx u."""
    nx = u.shape[0]
    u_hat = torch.fft.rfft(u.double())
    k = torch.arange(u_hat.shape[0], dtype=torch.float64, device=u.device)
    ux_hat = 1j * 2.0 * torch.pi * k / L * u_hat
    return torch.fft.irfft(ux_hat, n=nx).real.float()


def to_phi(u, nu, L):
    """Cole-Hopf forward: Burgers' field u -> heat field phi.

    phi(x) = exp(-1/(2*nu) * integral_0^x u(x') dx')
    Normalized so geometric mean of phi = 1 (scale-invariant for to_u).
    """
    psi = -1.0 / (2.0 * nu) * spectral_antideriv(u, L)
    psi = psi - psi.mean()       # log-center: geometric mean of phi = 1
    return torch.exp(psi)


def to_u(phi, nu, L):
    """Cole-Hopf inverse: heat field phi -> Burgers' field u.

    u = -2*nu * phi_x / phi  (spectral derivative, double precision)
    """
    phi_x = spectral_deriv(phi, L)
    return -2.0 * nu * phi_x / phi.clamp(min=1e-15)


def normalize_phi(phi):
    """Log-center phi so geometric mean = 1. Scale-invariant: to_u is unaffected.
    Also handles non-positive values from the GNN by shifting first."""
    # shift to strictly positive (preserves spatial gradient, constant cancels in to_u)
    phi_pos = phi - phi.min() + 1e-4
    log_phi = torch.log(phi_pos)
    return torch.exp(log_phi - log_phi.mean())


# ---- Rollout ----

def rollout_cole_hopf(model, u0, pos, edge_index, edge_attr, n_steps):
    """Roll out n_steps of Burgers' via Cole-Hopf + heat GNN.

    Args:
        u0: (NX,) initial Burgers' field (zero-mean)
    Returns:
        traj: (n_steps+1, NX) predicted u values
    """
    phi = to_phi(u0, NU_BURGERS, L)
    traj = [u0.clone()]

    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            # heat GNN step in phi-space
            x = torch.cat([phi.unsqueeze(-1), pos], dim=-1)
            phi_raw = model(x, edge_index, edge_attr).squeeze(-1)
            # enforce positivity + normalize (scale cancels in to_u)
            phi = normalize_phi(phi_raw)
            # recover Burgers' field
            u = to_u(phi, NU_BURGERS, L)
            traj.append(u.clone())

    return torch.stack(traj)   # (n_steps+1, NX)


def main():
    graph = torch.load(os.path.join(HYPER_DATA, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    test_traj = torch.load(os.path.join(HYPER_DATA, "test_traj.pt"), weights_only=False)
    meta = torch.load(os.path.join(HYPER_DATA, "meta.pt"), weights_only=False)

    model = HeatStepGNN(
        node_in_dim=2, edge_in_dim=2,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
    )
    model.load_state_dict(torch.load(PARA_MODEL, weights_only=True))

    n_traj, n_snap, nx = test_traj.shape
    n_steps = n_snap - 1
    preds = torch.zeros_like(test_traj)

    for i in range(n_traj):
        u0 = test_traj[i, 0]
        preds[i] = rollout_cole_hopf(model, u0, pos, edge_index, edge_attr, n_steps)

    # ---- metrics ----
    finite = torch.isfinite(preds)
    n_nan_inf = int((~finite.all(dim=(1, 2))).sum())
    preds_safe = torch.nan_to_num(preds, nan=1e8, posinf=1e8, neginf=-1e8)
    true_scale = float(test_traj.abs().max())
    n_diverged = int((preds_safe[:, -1].abs().amax(dim=-1) > 20 * true_scale).sum())

    err = (preds_safe - test_traj).pow(2).sum(dim=-1).sqrt()
    scale = test_traj.pow(2).sum(dim=-1).sqrt().clamp_min(1e-6)
    rel_err = (err / scale).mean(dim=0).numpy()
    mag = preds_safe.abs().amax(dim=(0, 2)).numpy()

    dt_model = meta["dt_model"]
    t = np.arange(n_snap) * dt_model

    # ---- plot: error curve + magnitude ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, rel_err, label="Cole-Hopf + heat GNN", color="tab:green")
    axes[0].set_xlabel("rollout time")
    axes[0].set_ylabel("mean relative L2 error")
    axes[0].set_title("rollout error (Cole-Hopf)")
    axes[0].legend()

    axes[1].plot(t, mag, label="max |u_pred|", color="tab:green")
    axes[1].axhline(true_scale, color="gray", linestyle="--", label="max |u_true|")
    axes[1].set_xlabel("rollout time")
    axes[1].set_ylabel("max |u|")
    axes[1].set_title("prediction magnitude")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rollout_error_cole_hopf.png"), dpi=140)
    plt.close(fig)

    # ---- plot: space-time for 2 test trajectories ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, i in enumerate([0, 1]):
        vmax = float(test_traj[i].abs().max())
        panels = [
            (test_traj[i].numpy(), "reference", "RdBu_r"),
            (preds_safe[i].numpy(), "Cole-Hopf GNN", "RdBu_r"),
            (np.abs(preds_safe[i].numpy() - test_traj[i].numpy()), "|error|", "inferno"),
        ]
        for col, (data, title, cmap) in enumerate(panels):
            kw = dict(origin="lower", aspect="auto", extent=[t[0], t[-1], 0, L])
            if col < 2:
                im = axes[row, col].imshow(data.T, cmap=cmap,
                                           vmin=-vmax, vmax=vmax, **kw)
            else:
                im = axes[row, col].imshow(data.T, cmap=cmap, **kw)
            axes[row, col].set_title(f"traj {i}: {title}")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            axes[row, col].set_xlabel("t")
            axes[row, col].set_ylabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rollout_spacetime_cole_hopf.png"), dpi=140)
    plt.close(fig)

    # ---- summary ----
    summary = {
        "method": "cole_hopf_heat_gnn",
        "nu_burgers": NU_BURGERS,
        "nu_heat_model": NU_HEAT_MODEL,
        "nu_mismatch": (
            "heat GNN trained at nu=0.01; Cole-Hopf requires nu=0.02. "
            "High modes will under-decay in phi-space -- low-mode dynamics "
            "should be less affected."
        ),
        "n_test_traj": n_traj,
        "n_nan_inf": n_nan_inf,
        "n_diverged_gt_20x_true_scale": n_diverged,
        "rel_err_at_25pct": float(rel_err[len(t) // 4]),
        "rel_err_at_50pct": float(rel_err[len(t) // 2]),
        "rel_err_at_final": float(rel_err[-1]),
        "max_pred_magnitude": float(mag.max()),
        "max_true_magnitude": true_scale,
    }
    with open(os.path.join(RESULTS_DIR, "rollout_summary_cole_hopf.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
