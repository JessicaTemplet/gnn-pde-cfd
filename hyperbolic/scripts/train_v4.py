"""
v4: curriculum unrolling -- the principled fix for the overfit-to-horizon
instability that destroyed v3.

v3's exact failure mode (see README and results/rollout_error_v1_v2_v3.png):
fixed-k=5 unrolled training let the model develop a per-step amplification
factor invisibly close to 1 (measured 1.17x/step). Over k=5 training steps
that looks like ~2.2x growth -- consistent with noise in the loss. Over k=100
rollout steps it compounds to 1.17^100 ~= 10^7. The training loss improved
monotonically while the full rollout blew up.

The fix: instead of fixing k, grow k from 1 to K_MAX over training.

At any epoch e, the current k(e) is large enough to penalize amplification
factors that slipped past at k(e-1). A model that begins to develop a
per-step factor of 1.1x gets caught when k grows to 8 (1.1^8 = 2.1x,
clearly penalized), then 12, then 15. By K_MAX the horizon is long enough
that even small per-step factors accumulate visibly in the loss.

Why this works where v3 didn't: v3's failure was that a model fine-tuned to
a fixed k=5 never had any loss signal that could see behavior at k>5.
Curriculum training exposes the model to progressively longer horizons, so
each increase in k provides a direct signal about the failure modes that were
invisible before.

Two additional safety measures learned from v3:
  1. Gradient clipping (norm 1.0): long unrolled gradients can spike,
     especially as k grows; clipping avoids the optimizer making large,
     destabilizing updates in response to rare large-gradient batches.
  2. In-chunk rollout spot-check: at the end of every training chunk, one
     full 100-step rollout is run and the final relative L2 and max magnitude
     are printed. This is exactly the check that would have caught v3's
     blowup immediately, rather than only when evaluate_rollout.py was run
     explicitly. Spot-check results are also appended to history_v4.json so
     you can track whether rollout quality is improving alongside val loss.

Architecture: same hidden_dim=16, n_layers=2 as v2/v3 -- required to
warm-start from v2 weights. Warm-starts from v2 (best stable rollout model),
not v3.

Validation loss is computed at fixed k=K_MAX throughout, so it is comparable
across epochs and not inflated by the early low-k training phases.

Run:
    python train_v4.py --epochs 5 --total-epochs 20
    python train_v4.py --epochs 5               # resumes if checkpoint exists
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from model import BurgersStepGNN  # noqa: E402
from train_v2 import tile_graph, sample_windows, unrolled_loss  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
RESULTS_DIR = os.path.join(HERE, "..", "results")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoint_v4.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history_v4.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v4.pt")
V2_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v2.pt")

BATCH_SIZE = 96
LR = 1.5e-3
SEED = 1
HIDDEN_DIM = 16
N_LAYERS = 2
K_MIN = 1
K_MAX = 15          # max rollout horizon seen during training
VAL_SEED = 999      # fixed -- val set is sampled once and reused every epoch
N_VAL_BATCHES = 8
GRAD_CLIP_NORM = 1.0


def curriculum_k(epoch: int, total_epochs: int) -> int:
    """Linear schedule: k = K_MIN at epoch 1, K_MAX at total_epochs."""
    if total_epochs <= 1:
        return K_MAX
    progress = (epoch - 1) / (total_epochs - 1)
    return max(K_MIN, min(K_MAX, round(K_MIN + (K_MAX - K_MIN) * progress)))


def build_fixed_val_set(val_traj, k, n_batches, batch_size, seed):
    """Sample the validation windows once with a fixed seed and concatenate.
    Reused identically for every epoch, giving a stable val-loss estimate."""
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
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, n - batch_size + 1, batch_size):
            u0 = u0_all[start:start + batch_size]
            targets = targets_all[start:start + batch_size]
            loss = unrolled_loss(model, u0, targets, pos_b, ei_b, ea_b,
                                 batch_size, n_nodes, k)
            total += loss.item()
            count += 1
    return total / count if count else float("inf")


def rollout_spot_check(model, test_traj, pos, edge_index, edge_attr):
    """One full trajectory rollout -- fast divergence check at end of each chunk.

    Returns (final_rel_L2, max_magnitude_over_full_rollout). A healthy model
    should have max_magnitude close to the true field scale (~3.0 for this
    dataset); anything >20x that is the same divergence signature as v3.
    """
    model.eval()
    with torch.no_grad():
        u0 = test_traj[0, 0].unsqueeze(-1)   # (nx, 1)
        n_steps = test_traj.shape[1] - 1
        pred = model.rollout(u0, pos, edge_index, edge_attr, n_steps)  # (n_steps+1, nx)
        ref = test_traj[0]                    # (n_steps+1, nx)
        final_err = (
            (pred[-1] - ref[-1]).pow(2).sum().sqrt()
            / ref[-1].pow(2).sum().sqrt().clamp_min(1e-6)
        )
        max_mag = float(pred.abs().max())
    return float(final_err), max_mag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5,
                        help="epochs to run in this chunk")
    parser.add_argument("--total-epochs", type=int, default=20,
                        help="total training budget (sets the curriculum endpoint)")
    parser.add_argument("--batches-per-epoch", type=int, default=15)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoint, warm-start from v2")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    graph = torch.load(os.path.join(DATA_DIR, "graph.pt"), weights_only=False)
    pos, edge_index, edge_attr = graph["pos"], graph["edge_index"], graph["edge_attr"]
    n_nodes = pos.shape[0]
    train_traj = torch.load(os.path.join(DATA_DIR, "train_traj.pt"), weights_only=False)
    val_traj = torch.load(os.path.join(DATA_DIR, "val_traj.pt"), weights_only=False)
    test_traj = torch.load(os.path.join(DATA_DIR, "test_traj.pt"), weights_only=False)

    # val at fixed K_MAX so the metric is comparable across all training epochs,
    # independent of the curriculum's current k
    val_u0, val_targets = build_fixed_val_set(
        val_traj, K_MAX, N_VAL_BATCHES, BATCH_SIZE, VAL_SEED
    )

    model = BurgersStepGNN(
        node_in_dim=2, edge_in_dim=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_epochs
    )

    start_epoch = 0
    best_val = float("inf")
    history = {
        "train": [], "val": [], "k_used": [],
        "spot_check_final_rel_l2": [], "spot_check_max_mag": [],
    }

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
        # deliberately NOT loading v2's optimizer state: see train_v3.py for
        # why restoring a cosine schedule that reached LR=0 pins the LR at
        # zero forever. only the model weights carry over.
        model.load_state_dict(torch.load(V2_MODEL_PATH, weights_only=True))
        print("warm-started from v2 model weights (fresh optimizer + LR schedule)")

    end_epoch = min(start_epoch + args.epochs, args.total_epochs)
    t0 = time.time()

    for epoch in range(start_epoch + 1, end_epoch + 1):
        k = curriculum_k(epoch, args.total_epochs)
        # seed from (SEED, epoch) so each chunk sees a fresh stretch of
        # random training windows rather than replaying the same ones
        train_rng = torch.Generator().manual_seed(SEED * 100_000 + epoch)

        model.train()
        pos_b, ei_b, ea_b = tile_graph(pos, edge_index, edge_attr, BATCH_SIZE, n_nodes)
        tr_total = 0.0
        for _ in range(args.batches_per_epoch):
            u0, targets = sample_windows(train_traj, BATCH_SIZE, k, train_rng)
            loss = unrolled_loss(
                model, u0, targets, pos_b, ei_b, ea_b, BATCH_SIZE, n_nodes, k
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            tr_total += loss.item()
        tr = tr_total / args.batches_per_epoch

        model.eval()
        va = eval_fixed_val(
            model, val_u0, val_targets, pos, edge_index, edge_attr,
            n_nodes, K_MAX, BATCH_SIZE
        )
        scheduler.step()

        history["train"].append(tr)
        history["val"].append(va)
        history["k_used"].append(k)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(
            f"epoch {epoch:4d}/{args.total_epochs}  k={k:2d}  "
            f"train={tr:.6f}  val(k={K_MAX})={va:.6f}",
            flush=True,
        )
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

    # --- end-of-chunk rollout spot check ---
    # Runs a full 100-step rollout on one test trajectory and prints the
    # final relative L2 and max field magnitude. This is the check that
    # would have immediately caught v3's blowup (max|pred| growing to 10^7
    # while training loss looked fine). If the status here shows DIVERGED,
    # do not continue training -- examine evaluate_rollout.py output first.
    spot_err, spot_mag = rollout_spot_check(
        model, test_traj, pos, edge_index, edge_attr
    )
    true_scale = float(test_traj.abs().max())
    diverged = spot_mag > 20 * true_scale
    status = "DIVERGED" if diverged else "OK"
    history["spot_check_final_rel_l2"].append(spot_err)
    history["spot_check_max_mag"].append(spot_mag)
    with open(HISTORY_PATH, "w") as fh:
        json.dump(history, fh)

    print(
        f"\nspot check [{status}]  final_rel_L2={spot_err:.4f}"
        f"  max|pred|={spot_mag:.2f}  (true scale={true_scale:.2f})"
    )
    print(
        f"chunk done in {time.time()-t0:.1f}s, now at epoch {end_epoch}/"
        f"{args.total_epochs}, best val(k={K_MAX})={best_val:.6f}"
    )
    if diverged:
        print(
            "WARNING: rollout is diverging. Stop training and run "
            "evaluate_rollout.py --tag v4 to diagnose before continuing."
        )


if __name__ == "__main__":
    main()
