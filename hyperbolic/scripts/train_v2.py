"""
v2: fix the rollout instability diagnosed from v1.

v1's diagnosis (see results/rollout_spacetime_v1.png): one-step MSE looked
fine (close to, but meaningfully below, the "predict no change" baseline),
but under autoregressive rollout the shock front stalls and the field
magnitude decays - the model never sees its own compounding error at
training time, so it has no incentive to get the *propagation speed* right
over many steps, only the single-step increment.

The fix: warm-start from the v1 weights (it already learned reasonable local
dynamics, no need to relearn that), then fine-tune with a short *unrolled*
rollout loss - k steps of the model's own predictions fed back as input,
backpropagated through the whole chain. This directly penalizes exactly the
failure mode observed: if the shock stalls over k steps, the unrolled
prediction diverges from the true trajectory and the loss sees it, which a
single-step loss structurally cannot.

Run:
    python train_v2.py --epochs 5 --total-epochs 20 --unroll-k 5
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from model import BurgersStepGNN  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_v2.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history_v2.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v2.pt")
V1_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v1.pt")

BATCH_SIZE = 96
LR = 1.5e-3
SEED = 1
HIDDEN_DIM = 16
N_LAYERS = 2


def tile_graph(pos, edge_index, edge_attr, batch_size, n_nodes):
    """Replicate a single fixed graph batch_size times into one big
    disconnected graph, so one message-passing forward call processes the
    whole batch at once (same trick torch_geometric's Batch does, done by
    hand here since we're not building Data objects for this loop)."""
    pos_b = pos.repeat(batch_size, 1)
    offsets = (torch.arange(batch_size) * n_nodes).repeat_interleave(edge_index.shape[1])
    ei_b = edge_index.repeat(1, batch_size) + torch.stack([offsets, offsets])
    ea_b = edge_attr.repeat(batch_size, 1)
    return pos_b, ei_b, ea_b


def sample_windows(traj: torch.Tensor, batch_size: int, k: int, rng: torch.Generator):
    """traj: (n_traj, n_snap, nx). Returns u0 (B, nx) and targets (B, k, nx)."""
    n_traj, n_snap, nx = traj.shape
    traj_idx = torch.randint(0, n_traj, (batch_size,), generator=rng)
    start_idx = torch.randint(0, n_snap - k, (batch_size,), generator=rng)
    u0 = traj[traj_idx, start_idx]                                    # (B, nx)
    targets = torch.stack([traj[traj_idx, start_idx + j + 1] for j in range(k)], dim=1)  # (B, k, nx)
    return u0, targets


def unrolled_loss(model, u0, targets, pos_b, ei_b, ea_b, batch_size, n_nodes, k):
    u = u0.reshape(-1, 1)  # (B*N, 1)
    total = 0.0
    for j in range(k):
        x = torch.cat([u, pos_b], dim=-1)
        u = model(x, ei_b, ea_b)
        target_j = targets[:, j, :].reshape(-1, 1)
        total = total + torch.nn.functional.mse_loss(u, target_j)
    return total / k


def run_epoch(model, traj, pos, edge_index, edge_attr, n_nodes, k, batch_size,
              n_batches, rng, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    pos_b, ei_b, ea_b = tile_graph(pos, edge_index, edge_attr, batch_size, n_nodes)
    total = 0.0
    for _ in range(n_batches):
        u0, targets = sample_windows(traj, batch_size, k, rng)
        loss = unrolled_loss(model, u0, targets, pos_b, ei_b, ea_b, batch_size, n_nodes, k)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--total-epochs", type=int, default=20)
    parser.add_argument("--unroll-k", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=20)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    n_nodes = pos.shape[0]
    train_traj = torch.load(os.path.join(DATA_DIR, "train_traj.pt"), weights_only=False)
    val_traj = torch.load(os.path.join(DATA_DIR, "val_traj.pt"), weights_only=False)

    model = BurgersStepGNN(node_in_dim=2, edge_in_dim=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)
    rng = torch.Generator().manual_seed(SEED)

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
        model.load_state_dict(torch.load(V1_MODEL_PATH, weights_only=True))
        print("warm-started from v1 one-step model")

    end_epoch = min(start_epoch + args.epochs, args.total_epochs)
    t0 = time.time()
    for epoch in range(start_epoch + 1, end_epoch + 1):
        tr = run_epoch(model, train_traj, pos, edge_index, edge_attr, n_nodes, args.unroll_k,
                        BATCH_SIZE, args.batches_per_epoch, rng, optimizer)
        with torch.no_grad():
            va = run_epoch(model, val_traj, pos, edge_index, edge_attr, n_nodes, args.unroll_k,
                            BATCH_SIZE, max(1, args.batches_per_epoch // 4), rng, optimizer=None)
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
