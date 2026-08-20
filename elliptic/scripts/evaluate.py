"""
Evaluate the trained elliptic GNN against the held-out test set and the
finite-difference reference, and produce the "simulation" plots: field
comparisons, error maps, error distribution, and the training loss curve.

Run:
    python evaluate.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from model import EllipticGNN  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")


def relative_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def plot_loss_curve(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(history["train_data"]) + 1)

    axes[0].plot(epochs, history["train_data"], label="train")
    axes[0].plot(epochs, history["val_data"], label="val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE(u_pred, u_fd)")
    axes[0].set_title("Supervised data loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_phys"], label="train")
    axes[1].plot(epochs, history["val_phys"], label="val")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE(-Laplacian(u_pred) - f, 0)")
    axes[1].set_title("PDE residual (physics) loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=140)
    plt.close(fig)


def plot_field_comparisons(model, test_set, shape, n_show=4):
    ny, nx = shape
    idxs = np.linspace(0, len(test_set) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, i in enumerate(idxs):
        data = test_set[i]
        bc_value = data.x[:, 2]
        with torch.no_grad():
            u_pred = model(data.x, data.edge_index, data.edge_attr, data.boundary_mask, bc_value)

        u_true = data.y.squeeze(-1).numpy().reshape(ny, nx)
        u_hat = u_pred.squeeze(-1).numpy().reshape(ny, nx)
        err = np.abs(u_hat - u_true)

        vmax = max(abs(u_true.min()), abs(u_true.max()))
        im0 = axes[row, 0].imshow(u_true, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"sample {i}: FD reference")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(u_hat, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title("GNN prediction")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(err, origin="lower", cmap="inferno")
        rel_err = relative_l2(u_hat.ravel(), u_true.ravel())
        axes[row, 2].set_title(f"|error|  (rel L2={rel_err:.3f})")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "field_comparison.png"), dpi=140)
    plt.close(fig)


def plot_error_histogram(model, test_set):
    rel_errs = []
    for data in test_set:
        bc_value = data.x[:, 2]
        with torch.no_grad():
            u_pred = model(data.x, data.edge_index, data.edge_attr, data.boundary_mask, bc_value)
        u_true = data.y.squeeze(-1).numpy()
        u_hat = u_pred.squeeze(-1).numpy()
        rel_errs.append(relative_l2(u_hat, u_true))

    rel_errs = np.array(rel_errs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rel_errs, bins=15, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("relative L2 error vs finite-difference solution")
    ax.set_ylabel("count")
    ax.set_title(f"Test set (n={len(test_set)}): mean={rel_errs.mean():.3f}, "
                 f"median={np.median(rel_errs):.3f}, max={rel_errs.max():.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "error_histogram.png"), dpi=140)
    plt.close(fig)
    return rel_errs


def main():
    meta = torch.load(os.path.join(DATA_DIR, "grid_meta.pt"), weights_only=False)
    test_set = torch.load(os.path.join(DATA_DIR, "test.pt"), weights_only=False)

    with open(os.path.join(RESULTS_DIR, "history.json")) as fh:
        history = json.load(fh)

    model = EllipticGNN(node_in_dim=5, edge_in_dim=3, hidden_dim=20, n_layers=2)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), weights_only=True))
    model.eval()

    plot_loss_curve(history)
    plot_field_comparisons(model, test_set, meta["shape"], n_show=4)
    rel_errs = plot_error_histogram(model, test_set)

    summary = {
        "n_test": len(test_set),
        "rel_l2_mean": float(rel_errs.mean()),
        "rel_l2_median": float(np.median(rel_errs)),
        "rel_l2_max": float(rel_errs.max()),
        "rel_l2_min": float(rel_errs.min()),
    }
    with open(os.path.join(RESULTS_DIR, "test_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print("plots written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
