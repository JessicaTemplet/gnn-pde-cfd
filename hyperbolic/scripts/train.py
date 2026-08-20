"""
Train the hyperbolic-stage GNN as a one-step Burgers' time integrator.

This is deliberately the *naive* setup: supervised single-step MSE only,
trained on (u_t, u_{t+dt}) pairs drawn independently from across many
trajectories. It has no idea, at training time, that its own predictions
will later be fed back in as input hundreds of times in a row - which is
exactly the setup this stage is designed to stress-test (see evaluate.py
and train_v2.py).

Checkpointed / resumable, same pattern as the elliptic stage:
    python train.py --epochs 5 --total-epochs 30
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
from model import BurgersStepGNN  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_v1.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history_v1.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v1.pt")

BATCH_SIZE = 512
LR = 2e-3
SEED = 0
HIDDEN_DIM = 16
N_LAYERS = 2


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = nn.functional.mse_loss(pred, batch.y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_pairs = torch.load(os.path.join(DATA_DIR, "train_pairs.pt"), weights_only=False)
    val_pairs = torch.load(os.path.join(DATA_DIR, "val_pairs.pt"), weights_only=False)
    train_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False)

    model = BurgersStepGNN(node_in_dim=2, edge_in_dim=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    start_epoch, best_val = 0, float("inf")
    history = {"train": [], "val": []}

    if os.path.exists(CKPT_PATH) and not args.fresh:
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt["best_val"]
        with open(HISTORY_PATH) as fh:
            history = json.load(fh)
        print(f"resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    end_epoch = min(start_epoch + args.epochs, args.total_epochs)
    t0 = time.time()
    for epoch in range(start_epoch + 1, end_epoch + 1):
        tr = run_epoch(model, train_loader, optimizer)
        with torch.no_grad():
            va = run_epoch(model, val_loader, optimizer=None)
        scheduler.step()
        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"epoch {epoch:4d}/{args.total_epochs}  train={tr:.6f}  val={va:.6f}", flush=True)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val": best_val,
        }, CKPT_PATH)
        with open(HISTORY_PATH, "w") as fh:
            json.dump(history, fh)

    print(f"chunk done in {time.time()-t0:.1f}s, now at epoch {end_epoch}/{args.total_epochs}, "
          f"best val={best_val:.6f}")


if __name__ == "__main__":
    main()
