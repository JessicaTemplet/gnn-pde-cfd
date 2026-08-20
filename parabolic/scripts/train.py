"""
Train the parabolic-stage GNN as a supervised one-step predictor.

This is the v1 (naive one-step) training from the hyperbolic playbook,
applied to the heat equation. Based on the parabolic hypothesis -- that
diffusion damps errors rather than amplifying them -- we expect this simpler
training recipe to already produce a rollout-stable model, in contrast to
the hyperbolic v1 where one-step training was necessary but not sufficient.

If evaluate_rollout.py shows the rollout is stable (relative L2 stays bounded
or decreases over the trajectory), the hypothesis is confirmed and there is
no need for a v2 unrolled fine-tuning pass. If the rollout is surprisingly
unstable, a v2 pass using train_v2.py from the hyperbolic stage (adapted for
the parabolic data and model) would be the natural next step.

Checkpointed and resumable: run in short chunks the same way as the
hyperbolic stage.

Run:
    python train.py --fresh --epochs 10 --total-epochs 30
    python train.py --epochs 10          # resumes from checkpoint
"""
import argparse
import json
import os
import sys
import time

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from model import HeatStepGNN  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

BATCH_SIZE = 64
LR = 1e-3
HIDDEN_DIM = 16
N_LAYERS = 2
SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="epochs to run in this chunk")
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore checkpoint and train from scratch")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_pairs = torch.load(os.path.join(DATA_DIR, "train_pairs.pt"), weights_only=False)
    val_pairs = torch.load(os.path.join(DATA_DIR, "val_pairs.pt"), weights_only=False)
    train_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False)

    model = HeatStepGNN(
        node_in_dim=2, edge_in_dim=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_epochs
    )

    start_epoch = 0
    best_val = float("inf")
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
        model.train()
        tr_total = 0.0
        for batch in train_loader:
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += loss.item()
        tr = tr_total / len(train_loader)

        model.eval()
        va_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                va_total += torch.nn.functional.mse_loss(pred, batch.y).item()
        va = va_total / len(val_loader)
        scheduler.step()

        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(f"epoch {epoch:4d}/{args.total_epochs}  train={tr:.6f}  val={va:.6f}",
              flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            },
            CKPT_PATH,
        )
        with open(HISTORY_PATH, "w") as fh:
            json.dump(history, fh)

    print(f"chunk done in {time.time()-t0:.1f}s, now at epoch {end_epoch}/"
          f"{args.total_epochs}, best val={best_val:.6f}")


if __name__ == "__main__":
    main()
