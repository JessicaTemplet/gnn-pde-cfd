"""
v3: continue v2's unrolled fine-tuning further, with the val-loss noise bug
fixed.

Why v2's val curve was noisy (0.0035 -> 0.0048 -> 0.0033 -> 0.0044 ...):
train_v2.py recreated `rng = torch.Generator().manual_seed(SEED)` fresh at
the top of *every process invocation* - and because this whole stage trains
in short resumable chunks (this sandbox has no persistent background
process and a hard ~45s wall-clock limit per call), that means every
resumed chunk replayed the *exact same* sequence of "random" validation
windows from the start, on only ~5 batches (480 of the 1,275 possible
windows). It's not noise in the conventional epoch-to-epoch sense - it's a
handful of chunk-boundary-aligned re-draws of a small, non-fixed validation
sample. Confirmed by reading the code, not just inferred from the curve.

Two fixes here:
  1. a validation set sampled ONCE with a fixed seed and reused identically
     every epoch - same idea as a normal held-out val split, which the
     resample-every-call version in v2 wasn't actually giving us.
  2. training randomness seeded from (SEED, start_epoch) instead of a fixed
     constant, so each resumed chunk sees a fresh stretch of random training
     windows instead of replaying the same ones.

Warm-starts from the v2 checkpoint (not v1) - this is a continuation of the
same unrolled fine-tune, run longer, to test the "more training closes more
of the gap" hypothesis directly rather than assert it.

Run:
    python train_v3.py --epochs 5 --total-epochs 40 --unroll-k 5
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from model import BurgersStepGNN  # noqa: E402
from train_v2 import tile_graph, sample_windows, unrolled_loss  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_v3.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history_v3.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v3.pt")
V2_CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_v2.pt")
V2_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v2.pt")

BATCH_SIZE = 96
LR = 1.5e-3
SEED = 1
HIDDEN_DIM = 16
N_LAYERS = 2
VAL_SEED = 999          # fixed - the val set itself never changes across epochs/chunks
N_VAL_BATCHES = 8        # fixed count too, larger than v2's ~5 for a steadier estimate


def build_fixed_val_set(val_traj, k, n_batches, batch_size, seed):
    """Sample the validation windows once, with a fixed seed, and return
    them concatenated - reused identically for every epoch's val loss."""
    gen = torch.Generator().manual_seed(seed)
    u0s, targets = [], []
    for _ in range(n_batches):
        u0, tgt = sample_windows(val_traj, batch_size, k, gen)
        u0s.append(u0)
        targets.append(tgt)
    return torch.cat(u0s, dim=0), torch.cat(targets, dim=0)


def eval_fixed_val(model, u0_all, targets_all, pos, edge_index, edge_attr,
                    n_nodes, k, batch_size):
    pos_b, ei_b, ea_b = tile_graph(pos, edge_index, edge_attr, batch_size, n_nodes)
    n = u0_all.shape[0]
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, n - batch_size + 1, batch_size):
            u0 = u0_all[start:start + batch_size]
            targets = targets_all[start:start + batch_size]
            loss = unrolled_loss(model, u0, targets, pos_b, ei_b, ea_b, batch_size, n_nodes, k)
            total += loss.item()
            n_batches += 1
    return total / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--total-epochs", type=int, default=40)
    parser.add_argument("--unroll-k", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=20)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    n_nodes = pos.shape[0]
    train_traj = torch.load(os.path.join(DATA_DIR, "train_traj.pt"), weights_only=False)
    val_traj = torch.load(os.path.join(DATA_DIR, "val_traj.pt"), weights_only=False)

    val_u0, val_targets = build_fixed_val_set(val_traj, args.unroll_k, N_VAL_BATCHES, BATCH_SIZE, VAL_SEED)

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
    else:
        model.load_state_dict(torch.load(V2_MODEL_PATH, weights_only=True))
        # deliberately NOT loading v2's optimizer state: v2's cosine schedule
        # ran to completion (T_max=15) and ended at LR=0, and CosineAnnealingLR's
        # step() is recursive - it multiplies the *previous* LR, not an
        # absolute function of epoch - so restoring an optimizer whose LR is
        # already 0 makes every subsequent step() multiply zero by zero
        # forever. (Found this by noticing v3's val loss was bit-for-bit
        # identical across epochs, which for different model weights on a
        # fixed validation set should be essentially impossible - traced it
        # to optimizer.param_groups[0]['lr'] being silently pinned at 0.0.)
        # A fresh optimizer/scheduler avoids the whole bug class; only the
        # model weights carry over from v2.
        print("warm-started from v2 model weights (fresh optimizer + LR schedule)")

    end_epoch = min(start_epoch + args.epochs, args.total_epochs)
    t0 = time.time()
    for epoch in range(start_epoch + 1, end_epoch + 1):
        # training rng seeded from (SEED, epoch): deterministic per run, but
        # a fresh stretch of random windows every epoch/chunk instead of
        # replaying the same ~20 batches every time the process restarts
        train_rng = torch.Generator().manual_seed(SEED * 100_000 + epoch)
        model.train()
        pos_b, ei_b, ea_b = tile_graph(pos, edge_index, edge_attr, BATCH_SIZE, n_nodes)
        tr_total = 0.0
        for _ in range(args.batches_per_epoch):
            u0, targets = sample_windows(train_traj, BATCH_SIZE, args.unroll_k, train_rng)
            loss = unrolled_loss(model, u0, targets, pos_b, ei_b, ea_b, BATCH_SIZE, n_nodes, args.unroll_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += loss.item()
        tr = tr_total / args.batches_per_epoch

        model.eval()
        va = eval_fixed_val(model, val_u0, val_targets, pos, edge_index, edge_attr,
                             n_nodes, args.unroll_k, BATCH_SIZE)
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
