"""
Train the elliptic-stage GNN to imitate the finite-difference Poisson solver.

Loss = supervised MSE(u_pred, u_fd) on interior nodes
     + lambda_phys * MSE(-Laplacian(u_pred) - f, 0) on interior nodes

The second term is a physics-informed residual: it uses the exact discrete
Laplacian stencil (common/fd_operators.py) applied to the network's own
prediction, so the network is pushed toward PDE-consistent solutions, not
just toward matching the reference pointwise.

Checkpointed / resumable so it can be run in short chunks:
    python train.py --epochs 10 --total-epochs 80
Each call trains `--epochs` more epochs (resuming from the last checkpoint
if one exists) until `--total-epochs` is reached.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from fd_operators import poisson_residual_scaled  # noqa: E402
from model import EllipticGNN                      # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")

BATCH_SIZE = 12
LR = 2e-3
LAMBDA_PHYS = 0.2
SEED = 0
HIDDEN_DIM = 20
N_LAYERS = 2


def run_epoch(model, loader, meta, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_data_loss, total_phys_loss, n_batches = 0.0, 0.0, 0

    for batch in loader:
        interior = ~batch.boundary_mask
        bc_value = batch.x[:, 2]

        u_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.boundary_mask, bc_value)
        data_loss = nn.functional.mse_loss(u_pred[interior], batch.y[interior])

        f = batch.x[:, 0]
        residual = poisson_residual_scaled(u_pred.squeeze(-1), batch.stencil_edge_index, f, meta["dx"], meta["dy"])
        phys_loss = (residual[interior] ** 2).mean()

        loss = data_loss + LAMBDA_PHYS * phys_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_data_loss += data_loss.item()
        total_phys_loss += phys_loss.item()
        n_batches += 1

    return total_data_loss / n_batches, total_phys_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="epochs to run this invocation")
    parser.add_argument("--total-epochs", type=int, default=80, help="target total epochs (for LR schedule)")
    parser.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint and start over")
    parser.add_argument("--warm-restart", action="store_true",
                         help="keep model+optimizer weights but reset the LR schedule "
                              "(use when a cosine schedule has bottomed out and progress has stalled)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    meta = torch.load(os.path.join(DATA_DIR, "grid_meta.pt"), weights_only=False)
    train_set = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=False)
    val_set = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = EllipticGNN(node_in_dim=5, edge_in_dim=3, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    start_epoch = 0
    best_val = float("inf")
    history = {"train_data": [], "train_phys": [], "val_data": [], "val_phys": []}

    if os.path.exists(CKPT_PATH) and not args.fresh:
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if not args.warm_restart:
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
        else:
            # keep weights + Adam moment estimates, but let the cosine
            # schedule restart from LR_max over the new total-epochs window
            start_epoch = 0
            print("warm restart: LR schedule reset, epoch counter reset for scheduling purposes")
        best_val = ckpt["best_val"]
        with open(HISTORY_PATH) as fh:
            history = json.load(fh)
        print(f"resumed from epoch {ckpt['epoch']}, best_val={best_val:.6f}")

    end_epoch = min(start_epoch + args.epochs, args.total_epochs)
    t0 = time.time()

    for epoch in range(start_epoch + 1, end_epoch + 1):
        tr_data, tr_phys = run_epoch(model, train_loader, meta, optimizer)
        with torch.no_grad():
            va_data, va_phys = run_epoch(model, val_loader, meta, optimizer=None)
        scheduler.step()

        history["train_data"].append(tr_data)
        history["train_phys"].append(tr_phys)
        history["val_data"].append(va_data)
        history["val_phys"].append(va_phys)

        if va_data < best_val:
            best_val = va_data
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

        print(f"epoch {epoch:4d}/{args.total_epochs}  train_data={tr_data:.6f}  train_phys={tr_phys:.6f}  "
              f"val_data={va_data:.6f}  val_phys={va_phys:.6f}", flush=True)

        # checkpoint after every epoch so a killed/interrupted run loses at
        # most one epoch of progress and can always be resumed
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }, CKPT_PATH)
        with open(HISTORY_PATH, "w") as fh:
            json.dump(history, fh)

    elapsed = time.time() - t0
    print(f"chunk done in {elapsed:.1f}s, now at epoch {end_epoch}/{args.total_epochs}, best val_data={best_val:.6f}")


if __name__ == "__main__":
    main()
