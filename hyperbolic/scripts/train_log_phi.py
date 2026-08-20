"""
Train a GNN to advance psi = log(phi) one step in time.

psi is the Cole-Hopf log variable derived from Burgers' trajectories.
The model learns psi_t -> psi_{t+dt}. Recovering Burgers' u from psi is then
just u = -2*nu * spectral_deriv(psi) -- no division, no positivity constraint.

psi dynamics are nonlinear (psi_t = nu*(psi_xx + psi_x^2)) but bounded, so
one-step training is expected to converge. If the rollout turns out unstable,
the same curriculum-unrolling fix from v4 is directly applicable here.

Same architecture as v4 (BurgersStepGNN, increment prediction), same training
hyperparameters, same gradient clipping. Saves to best_log_phi_model.pt.

Run:
    python train_log_phi.py --fresh --epochs 30 --total-epochs 30
    python train_log_phi.py --epochs 10           # resume
"""
import argparse
import json
import os
import sys
import time

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import BurgersStepGNN  # same architecture, different variable

HERE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(HERE, "..", "data")
MODEL_DIR   = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH   = os.path.join(MODEL_DIR, "log_phi_checkpoint.pt")
BEST_PATH   = os.path.join(MODEL_DIR, "best_log_phi_model.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history_log_phi.json")

BATCH_SIZE = 96
LR         = 1.5e-3
HIDDEN_DIM = 16
N_LAYERS   = 2
SEED       = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--fresh",        action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_pairs = torch.load(os.path.join(DATA_DIR, "psi_train_pairs.pt"), weights_only=False)
    val_pairs   = torch.load(os.path.join(DATA_DIR, "psi_val_pairs.pt"),   weights_only=False)
    train_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_pairs,   batch_size=BATCH_SIZE, shuffle=False)

    model     = BurgersStepGNN(node_in_dim=2, edge_in_dim=2,
                               hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_epochs)

    start_epoch = 0
    best_val    = float("inf")
    history     = {"train": [], "val": []}

    if os.path.exists(CKPT_PATH) and not args.fresh:
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        with open(HISTORY_PATH) as fh:
            history = json.load(fh)
        print(f"resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    end_epoch = min(start_epoch + args.epochs, args.total_epochs)
    t0 = time.time()

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        tr = 0.0
        for batch in train_loader:
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr += loss.item()
        tr /= len(train_loader)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                va  += torch.nn.functional.mse_loss(pred, batch.y).item()
        va /= len(val_loader)
        scheduler.step()

        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), BEST_PATH)

        print(f"epoch {epoch:4d}/{args.total_epochs}  train={tr:.6f}  val={va:.6f}",
              flush=True)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val": best_val}, CKPT_PATH)
        with open(HISTORY_PATH, "w") as fh:
            json.dump(history, fh)

    print(f"done in {time.time()-t0:.1f}s  epoch {end_epoch}/{args.total_epochs}"
          f"  best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
