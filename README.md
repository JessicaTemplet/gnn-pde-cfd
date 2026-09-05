# GNN-PDE CFD workflow

Training untrained graph neural networks to solve PDEs, one class at a time:
elliptic, then parabolic, then hyperbolic. Each stage trains a GNN to imitate
a classical numerical solver on that PDE class, then validates the trained
network against the classical "simulation" (finite-difference reference).

## Stack decision

Rust was the first choice, but was dropped for this project after checking
the ecosystem: there is no Rust equivalent of PyTorch Geometric. `burn` has
autograd but admits it isn't GPU-mature yet; the only GNN-on-Rust project
found (`candle-gnn`) is a small, non-production library; `petgraph` is a graph
*data structure* crate with no neural network layer on top. Building
message-passing layers, batching, and scatter/gather ops from scratch in Rust
(a language you're still ramping up on, vs. Go being suggested as a better
fit) was judged too much risk for too little payoff. Went with **PyTorch +
PyTorch Geometric** instead.

## Layout

```
common/             shared grid->graph utilities and finite-difference operators
elliptic/           stage 1: steady-state Poisson equation      (done, see below)
parabolic/          stage 2: heat equation                      (done -- one-step training confirmed rollout-stable)
hyperbolic/         stage 3: 1D viscous Burgers' equation        (done -- Cole-Hopf log-phi is the shipped model)
```

Stage order in this repo is elliptic -> hyperbolic -> parabolic (parabolic was
skipped for now and hyperbolic pulled forward, at your call - see the
hyperbolic section for why Burgers' specifically).

Each stage follows the same pattern: `scripts/generate_dataset.py` builds
training data with a classical solver, `scripts/model.py` defines the GNN,
`scripts/train.py` trains it (checkpointed/resumable), `scripts/evaluate.py`
produces the validation plots.

## Elliptic stage: what's built and what it shows

**Problem**: 2D Poisson equation `-Laplacian(u) = f` on a 32x32 grid, unit
square, homogeneous Dirichlet boundary (`u = 0` on the edges). `f` is a random
sum of 2-5 Gaussian bumps per sample (240 train / 30 val / 30 test).

**Reference solver**: direct sparse LU solve of the discretized system
(`common/fd_operators.py`) - exact up to discretization error, used both to
generate training targets and as the "simulation" the GNN is validated
against.

**Model** (`elliptic/scripts/model.py`): an encode-process-decode
message-passing GNN, PyTorch Geometric `MessagePassing` layers, mean
aggregation, hard-enforced Dirichlet BC (boundary nodes are overwritten with
the known value rather than penalized). Trained with a supervised MSE loss
against the FD solution, plus a small PDE-residual term computed from the
network's own prediction using the exact discrete Laplacian stencil.

**The graph isn't a plain grid.** A regular 4-connected grid graph needs as
many message-passing layers as the grid's diameter for information to reach
across the domain, because Poisson's equation is genuinely non-local - every
point's value depends on the source term everywhere. The first version of
this used a plain grid with 3-6 local layers and it visibly failed (see
"what didn't work" below). The fix, used in the current version, is a
multiscale graph (`common/grid.py`): shortcut edges at strides 2, 4, 8, 16 on
top of the base grid, the same idea GraphCast uses for its multi-mesh. That
gives a shallow (2-layer) network a full-domain receptive field in a couple
of hops.

### Results

Trained ~24 epochs (small model: hidden dim 20, 2 message-passing layers,
batch size 12 - sized to fit this sandbox's CPU-only, no-persistent-background
execution environment). Train/val loss converge together with no overfitting.

| metric | value |
|---|---|
| test relative L2 error | mean 0.74, median 0.68, range [0.57, 1.42] |

See `elliptic/results/`: `loss_curve.png`, `field_comparison.png` (FD
reference vs. GNN prediction vs. error map for 4 test samples),
`error_histogram.png`, `test_summary.json`.

**Honest read of the result**: the network gets the sign and approximate
location of each source bump right, but under-predicts how far the true
solution spreads out - the FD reference has a smooth, long-range halo around
each bump (Poisson's Green's function decays slowly), while the GNN's
prediction is comparatively localized. That's visible directly in
`field_comparison.png`. This is a genuine, verified result - not a bug I
missed - and it's consistent with the network still being under-resourced
for full nonlocal coupling: 2 message-passing layers plus multiscale
shortcuts covers the domain in a few hops, but a couple of hops through a
small (hidden dim 20) network is still a coarse approximation of the true
Green's function.

**What didn't work, briefly, since it's instructive**: the first working
version trained with plain grid connectivity and converged to something
barely better than predicting zero everywhere (relative L2 ~0.9-1.5).
Two compounding bugs, both fixed in the current code:
1. the physics-residual loss was computed with a raw `1/dx^2` stencil, which
   on a 32x32 grid multiplies by ~960 - it dwarfed the data loss and
   destabilized training before it was reformulated in the scaled form
   (`common/fd_operators.py: poisson_residual_scaled`).
2. training used batch size 60 (only 4 gradient steps/epoch on 240 samples) -
   dropping to batch size 12 (20 steps/epoch) is most of why the model
   started learning structure instead of the mean field.

**To push accuracy further** (didn't do this here - each of these multiplies
training time, and this environment is CPU-only with no persistent background
processes, so every training run has to fit in short, resumable chunks):
more message-passing layers or a proper hierarchical/pooled multiscale
architecture (graph U-Net style, closer to what production mesh-based PDE
solvers use), a bigger hidden dimension, more training data, more epochs,
and - the single biggest lever - running on a GPU instead of a 2-core CPU
sandbox.

### Running it

```
cd elliptic/scripts
python generate_dataset.py
python train.py --fresh --epochs <n> --total-epochs <target>   # resumable, checkpoints every epoch
python evaluate.py
```

## Hyperbolic stage: 1D viscous Burgers' equation

Went with Burgers' (`u_t + u u_x = nu u_xx`) instead of the linear wave
equation originally sketched in the roadmap below. It's the standard choice
in the neural-PDE-surrogate literature for a specific reason: it's nonlinear
and forms shocks from smooth initial conditions, so it's genuinely hard, but
it's still a 1D scalar equation, cheap enough to solve and train on in this
CPU sandbox. And critically, naive training setups on it *visibly*
misbehave under autoregressive rollout - which is exactly what happened
here, giving a real (not staged) instability to diagnose and fix.

**Problem**: periodic 1D domain `x in [0,1)`, viscosity `nu=0.02`, smooth
random initial conditions (truncated random Fourier series, amplitude
decaying with mode number - same recipe used in the FNO Burgers'
benchmark). 80 train / 15 val / 15 test trajectories, each rolled out to
`t=0.5`, 101 snapshots, `dt_model=0.005`.

**Reference solver** (`hyperbolic/scripts/spectral_solver.py`): Fourier
pseudo-spectral in space, integrating-factor + RK4 in time (diffusion
handled exactly, advection explicit), 2/3-rule dealiasing. `nu=0.02` was
chosen empirically (`results/_diag_snapshots.png`) as the largest viscosity
that still forms a clean, visible shock without under-resolved grid noise at
`nx=64` - smaller `nu` shocks up faster but starts aliasing at this
resolution.

**Model** (`hyperbolic/scripts/model.py`): same encode-process-decode
message-passing family as the elliptic stage, but predicts the *increment*
`u_{t+dt} - u_t` rather than the absolute next state - a zero-output network
defaults to "nothing changes," a much safer failure mode for something that
gets run autoregressively for 100 steps. Periodic chain graph with stride
1/2/4/8 shortcut edges (`common/grid.py: build_periodic_chain_graph`).

### v1: the naive baseline, and what actually broke

`train.py` trains the model as a plain one-step regressor: supervised MSE
on independently-sampled `(u_t, u_{t+dt})` pairs pooled across all
trajectories, no notion that the model will later feed its own output back
in as input. One-step validation loss converged fine, meaningfully below
the "predict no change" baseline (0.00048 vs. 0.00057 MSE) - by that metric
it looks like a working model.

**Then the rollout told a different story.** Feeding the trained model its
own predictions for 100 steps (`evaluate_rollout.py`, `results/
rollout_spacetime_v1.png`): it doesn't blow up to infinity, but the shock
*stalls*. The reference solution's shock front visibly advects across the
domain over `t=0..0.5`; the v1 rollout captures the initial steepening and
then the whole profile essentially freezes in place a fraction of the way
through, decaying in amplitude instead of continuing to propagate. Mean
relative L2 error against the reference climbs from near 0 to 0.80 within
the first fifth of the trajectory and plateaus there (`results/
rollout_error_v1.png`).

This is a legitimate, verified instability, not a numerical blow-up but the
more common failure mode for one-step-trained neural time-steppers:
minimizing single-step error has no incentive to get the *propagation
speed* right, and a network that just predicts "small change" is already
close to a one-step optimum, so nothing in training pushes it to correct
that over many steps.

### v2: the fix

`train_v2.py` warm-starts from the v1 weights (the local one-step dynamics
were fine, no need to relearn them) and fine-tunes with an **unrolled
multi-step loss**: 5 steps of the model's own predictions fed back in as
input, backpropagated through the whole chain, loss averaged over all 5
steps against the true trajectory. This directly penalizes the observed
failure mode - a stalled shock diverges from the true trajectory over those
5 steps, and the single-step loss structurally could never see that.

Before trusting a full training run, I verified the unrolled-training code
itself was correct by overfitting one fixed batch (loss 0.087 -> 0.0024
over 80 iterations) - confirms gradients flow properly through the
backprop-through-time chain before spending a training budget on it.

**Result**: real, measurable improvement, not a full fix. Final-time
rollout relative L2 error dropped from 0.80 (v1) to 0.65 (v2); the
improvement is consistent across the whole trajectory, not just at one
point in time (`results/rollout_error_v1_vs_v2.png` overlays both curves on
the same axes). The shock still stalls visibly in `results/
rollout_spacetime_v2.png` - 15 epochs of fine-tuning on a 16-hidden-unit,
2-layer model wasn't enough training to fully relearn the correct
propagation speed, only enough to measurably dent the bias toward it.

| | v1 (naive) | v2 (warm-start + 5-step unrolled, 15 ep) - **final model** |
|---|---|---|
| rollout relative L2 @ t=0.125 (25%) | 0.67 | 0.58 |
| rollout relative L2 @ t=0.5 (final) | 0.80 | 0.65 |
| blew up | no | no |

### v3: continuing training further - a second instability, caught before being reported

The obvious next question was "if 15 epochs of unrolled fine-tuning helped,
does more help further?" `train_v3.py` continues v2's training - same
`unroll-k 5`, warm-started from v2's weights - for 13 more epochs. The
5-step unrolled *training and validation* loss kept improving the entire
time (0.0038 -> 0.0025, smoothly, no noise - see below for why). By that
metric it looked like a clean continued win.

**Full-rollout evaluation said otherwise.** At epoch 10 the rollout was
already worse than v2 (final relative L2 0.84 vs. v2's 0.65). By epoch 13 it
was a genuine blow-up: max field magnitude reached 8.3 million against a
true scale of ~3. Checked whether that was three different things it could
plausibly have been - an LR-restart transient that would recover, the old
stall reappearing in a new shape, or the model overfitting to the 5-step
horizon specifically - by looking at the raw (unclipped) prediction
magnitude at every step, not just the plotted spacetime comparison (which,
worth flagging, was *itself* misleading here: matplotlib had silently
clamped its color scale to the reference trajectory's range, so the plot
visually looked like an ordinary stall until the raw numbers were checked
directly). The magnitude trace ruled two of the three out immediately:

```
t=0.000  max|pred|=2.97      t=0.200  max|pred|=642
t=0.025  max|pred|=3.42      t=0.300  max|pred|=14,833
t=0.050  max|pred|=5.85      t=0.400  max|pred|=348,316
t=0.100  max|pred|=29.3      t=0.500  max|pred|=8,283,525
```

That's a constant ~2.17x growth every 5 model-steps from the very first
step - not a delayed onset (rules out a transient), not amplitude decay
(rules out the old stall - this is the opposite sign), just clean
exponential growth (~1.17x per single step) present from t=0. That's the
signature of the third option: the model overfit to the unroll horizon it
was trained on. A per-step amplification of 1.17x compounds to only ~2.2x
over a 5-step training window - indistinguishable from noise to that loss -
but to `1.17^100 ~ 10^7` over the 100 steps a real rollout needs. Nothing in
a fixed-k unrolled loss constrains the *spectral radius* of the learned
step map for k beyond what it was trained on, and v3 walked straight past
the point where that radius crossed 1.

**v2 remains the delivered model for this stage precisely because this
check was done before reporting v3 as an improvement.** `results/
rollout_error_v1_v2_v3.png` plots all three: v1 and v2 both plateau in the
0.6-0.8 range, v3 tracks them closely for the first ~0.05 time units and
then diverges exponentially off the top of a log-scale axis.

**A second, unrelated bug found and fixed along the way**: v2's own loss
curve had looked noisy epoch-to-epoch (0.0035 -> 0.0048 -> 0.0033 -> ...).
Traced it to a real cause, not just "unrolled loss is noisier by nature" -
`train_v2.py` recreates `torch.Generator().manual_seed(SEED)` fresh at the
top of every process invocation, and because this whole stage trains in
short resumable chunks (no persistent background process, ~45s wall-clock
limit per call), every resumed chunk was replaying the exact same small
random validation sample from the start rather than continuing a evolving
sequence. `train_v3.py` fixes this with a validation set sampled once with
a fixed seed and reused identically every epoch, plus training randomness
seeded from `(SEED, epoch)` so resumed chunks see fresh windows instead of
replaying old ones. v2's *reported numbers are unaffected* by this -  v2
warm-started from v1 using only model weights, no optimizer state, so its
own training ran a normal, undisturbed cosine LR schedule start to finish.

That bug fix surfaced a second, sharper one: building v3, the first attempt
warm-started from v2 by loading v2's *saved optimizer state* too (to carry
over Adam's momentum), which reintroduced v2's exact final learning rate -
0.0, since v2's cosine schedule had annealed fully to its minimum by epoch
15. `CosineAnnealingLR.step()` is recursive - it multiplies the *previous*
LR by a ratio rather than computing an absolute value from the epoch number
- so restoring an optimizer whose LR was already 0 pinned it at 0 forever;
every subsequent step multiplied zero by zero. Caught this because two
different model states produced bit-identical validation loss to 15
significant figures, which is essentially impossible by chance. Deliberately
chose to fix this with a fresh optimizer/scheduler (only the model weights
carry over from v2) rather than patch the LR floor post-load and keep
Adam's momentum buffers: simpler, closes off that whole bug class rather
than one instance of it, and the cost - a few steps of cold-started
momentum - is negligible at this scale (a few dozen epochs total). The
patch-the-floor version would be worth the extra complexity in a much
longer training run where warm momentum actually matters.

**What a v4 would target, now that the mechanism is known precisely**: not
just "more epochs" (v3 already shows that's actively harmful past a point) -
curriculum unrolling (start at k=1, grow k over training, so the loss sees
longer horizons before the model can overfit to a short one), or an explicit
penalty on the local Jacobian's spectral radius, or simply stopping
earlier/more conservatively and checking full-rollout error every couple of
epochs instead of trusting the training loss - which is exactly the
discipline that caught this before it went out as a reported result.

### v4: curriculum unrolling (in progress)

The mechanism behind v3's failure is now precisely known: a fixed unroll
horizon k cannot penalize a per-step amplification factor that is invisible
at that k but catastrophic at k=100. The correct fix is to not fix k at all.

**Curriculum unrolling** (`train_v4.py`): k grows linearly from 1 to
`K_MAX=15` over the full training run. At any epoch e, the current k(e) is
large enough to see failure modes that slipped past at k(e-1). A model that
starts to develop a per-step factor of 1.1x gets caught when k grows to 8
(1.1^8 ≈ 2.1x, clearly penalized in the loss), then 12, then 15. By K_MAX
the horizon is long enough that only genuinely bounded dynamics are rewarded.

This directly addresses v3's structural problem -- training on a fixed k=5
horizon gave the model no signal about behavior at k>5 -- while v1's problem
(single-step training, no rollout signal at all) is handled by starting k at
1 and immediately growing it.

Two further additions from v3's lessons:
- **Gradient clipping** (norm 1.0): long unrolled gradients can spike as k
  grows; clipping prevents the optimizer from making large destabilizing
  updates on rare high-gradient batches.
- **In-chunk rollout spot check**: at the end of every training chunk,
  `train_v4.py` runs one full 100-step rollout and prints the final relative
  L2 and max prediction magnitude. This is the check that would have caught
  v3's blowup immediately rather than only when `evaluate_rollout.py` was
  run explicitly. The spot-check results are also appended to `history_v4.json`.

Warm-starts from v2 (the best stable model). Validation loss is computed at
fixed `k=K_MAX=15` throughout, so it is comparable across epochs regardless
of the current curriculum k.

### Results

Trained 20 epochs (hidden dim 16, 2 layers, CPU-only, ~3 min total). Val
loss decreased from 0.018273 (epoch 1, k=1) to 0.010821 (best, epoch 17,
k=13), and k grew from 1 to 15 exactly as scheduled.

| | v1 (naive) | v2 (unrolled k=5) | v4 (curriculum k=1→15) | **Cole-Hopf v2 (log-phi)** |
|---|---|---|---|---|
| rel L2 @ t=0.125 (25%) | 0.674 | 0.582 | 0.488 | **0.105** |
| rel L2 @ t=0.25 (50%) | 0.778 | 0.648 | 0.594 | **0.122** |
| rel L2 @ t=0.5 (final) | 0.796 | 0.648 | 0.626 | **0.237** |
| diverged (any of 15 test traj) | 0 | 0 | 0 | **0** |
| max prediction magnitude | 2.97 | 2.97 | 2.97 | **2.97** |

v4 is the best result from training directly in u-space. Cole-Hopf v2 beats
v4 by 4.6x at the 25% mark and 2.6x at final time -- the single biggest
accuracy jump in the project -- with plain one-step training (no curriculum,
no unrolling), 30 epochs, same architecture as v1.

**Why Cole-Hopf v2 works**: the transformation maps u to psi = log(phi),
where phi = exp(-1/(2*nu) * integral(u)). Two things change in psi-space:

First, the inverse is `u = -2*nu * psi_x` (spectral derivative only, no
division), eliminating the positivity constraint that caused v1 Cole-Hopf to
blow up on 10/15 test trajectories. Psi is unconstrained real-valued, so
GNN output fits without any salvage step.

Second, psi is the antiderivative of u (scaled). Shock discontinuities in u
appear only as kinks in psi -- the field the GNN actually regresses is
smoother than u, making the target easier to learn. The model doesn't have
to directly represent a propagating discontinuity; it models a diffusing
primitive function.

**Why v4 (curriculum unrolling) still mattered**: it diagnosed the fixed-horizon
spectral-radius failure precisely, which motivated looking for a
representation change. The log-phi idea wouldn't have been obvious without
first understanding exactly why v3 blew up.

`results/rollout_error_cole_hopf_v2.png` shows the error curve;
`results/rollout_spacetime_cole_hopf_v2.png` shows space-time comparisons.

### Running it

```
cd hyperbolic/scripts
python generate_dataset.py
python train.py --fresh --epochs <n> --total-epochs <target>                   # v1, naive one-step
python evaluate_rollout.py --tag v1
python train_v2.py --fresh --epochs <n> --total-epochs <target> --unroll-k 5   # v2, warm-started + unrolled
python evaluate_rollout.py --tag v2
python train_v3.py --fresh --epochs <n> --total-epochs <target> --unroll-k 5   # v3, cautionary tale - do not ship
python evaluate_rollout.py --tag v3
python train_v4.py --fresh --epochs <n> --total-epochs 20                       # v4, curriculum unrolling
python evaluate_rollout.py --tag v4
python compare_rollouts.py v1 v2 v4        # overlay error curves

# Cole-Hopf v2: best model
python generate_log_phi_dataset.py          # transforms Burgers' traj -> psi = log(phi)
python train_log_phi.py --fresh --epochs 30 --total-epochs 30
python cole_hopf_v2_rollout.py
```

## Parabolic stage: 1D heat equation

**Problem**: periodic 1D domain `x in [0,1)`, diffusion coefficient `nu=0.01`,
same smooth random initial conditions as the hyperbolic stage (truncated
Fourier series with amplitude decaying by mode number). 80 train / 15 val /
15 test trajectories, each rolled out to `t=0.5`, 101 snapshots, `dt_model=0.005`.

**Reference solver** (`parabolic/scripts/heat_solver.py`): exact spectral
method. The heat equation is linear, so each Fourier mode evolves
independently: `u_hat(k, t) = u_hat(k, 0) * exp(-nu * (2*pi*k/L)^2 * t)`.
This is applied as a per-mode multiplicative decay at each timestep -- there
is no stability constraint on dt (diffusion is unconditionally stable), no
nonlinear term, and no dealiasing needed. The reference solution is exact up
to floating-point rounding and the initial Fourier truncation, making it a
cleaner reference than the Burgers' solver's RK4+dealiasing.

`nu=0.01` was chosen so that high Fourier modes (k=4 and above) decay by 96%
within the first fifth of the trajectory, while the lowest modes (k=1) retain
~82% of their amplitude at `t=0.5` -- giving an interesting multi-scale
transient rather than immediate collapse to near-zero or unremarkably slow
diffusion.

**Model** (`parabolic/scripts/model.py`): same encode-process-decode
message-passing family as both prior stages -- `HeatStepGNN`, predicts the
increment `u_{t+dt} - u_t`, periodic chain graph with stride 1/2/4/8 shortcut
edges. The architecture is deliberately identical to the hyperbolic stage so
any performance difference is attributable to the PDE, not the model.

**The parabolic hypothesis**: diffusion is dissipative -- prediction errors are
smoothed away by the dynamics rather than advected and amplified as they are
in the hyperbolic case. If this holds for the GNN, one-step supervised
training (`train.py`) should already produce a rollout-stable model, and no
v2 unrolled fine-tuning pass should be necessary.

### Results

**The hypothesis is confirmed.** Trained 30 epochs, best one-step val MSE =
5e-6 (cosine annealing drove a sharp improvement starting at epoch 19).
Autoregressive rollout over 100 steps on all 15 test trajectories:

| | parabolic (one-step, 30 ep) | hyperbolic v4 (curriculum k=1→15, 20 ep) |
|---|---|---|
| rel L2 @ t=0.125 (25%) | **0.053** | 0.488 |
| rel L2 @ t=0.25 (50%) | **0.057** | 0.594 |
| rel L2 @ t=0.5 (final) | **0.070** | 0.626 |
| diverged (any of 15 traj) | **0** | 0 |
| max prediction magnitude | **2.97** | 2.97 |

The error curve is essentially flat: 5.3% at the 25% mark, 7.0% at the end
of the trajectory, no growth trend at all. This is the dissipation effect in
action -- small per-step prediction errors are smoothed out by the dynamics
rather than advected. The same model architecture, the same graph structure,
the same one-step training recipe that required a v2 unrolled fix and a v4
curriculum fix on Burgers' needs nothing extra here, and achieves 9x lower
final error than hyperbolic v4.

This makes the parabolic result a clean control case for the hyperbolic
instability story: the v1 failure (stalled shock, error climbing to 0.80) was
specific to the hyperbolic nature of Burgers' equation, not a fundamental
flaw in one-step GNN training. `results/rollout_error.png` and
`results/rollout_spacetime.png` show the flat error curve and per-trajectory
comparisons; `results/rollout_summary.json` has the numbers.

### Running it

```
cd parabolic/scripts
python generate_dataset.py
python train.py --fresh --epochs <n> --total-epochs 30
python evaluate_rollout.py
```

## Validator: a configurable check suite across stages

Every stage above already had its own `evaluate.py` / `evaluate_rollout.py`
doing real validation (rollout divergence, relative-L2 accuracy, the v3
blow-up catch) -- but each was a bespoke script hardcoded to one model class
and one data path. `validator/` generalizes that pattern into one
config-driven tool, instead of rewriting a new evaluate script per
checkpoint: point it at a model class + checkpoint + dataset via YAML, get
back the same kind of checks (stability, accuracy, conservation,
generalization) as a JSON report.

This is deliberately scoped to what this repo actually has, not to every
possible GNN-PDE setup: two problem types (`rollout` for the time-dependent
stages, `steady_state` for elliptic), and a small conservation-check
registry (`validator/checks/conservation.py`) with a real implementation
only for the Poisson residual that elliptic already trains against --
future stages (a real CFD stage: incompressible Navier-Stokes, compressible
flow, RANS) get a new registry entry each, not a redesign.

```
python -m validator.cli --config validator/configs/parabolic.yaml
python -m validator.cli --config validator/configs/hyperbolic_cole_hopf_v2.yaml --out validator/results/cole_hopf.json
python validator/tests/test_checks.py    # check logic, synthetic data, no checkpoints needed
```

(Run from the repo root, not from `validator/`, since it's invoked as a
module.) Default output is a plain-English pass/fail summary with an
explanation of what each failing check probably means (pass `--json` for
the raw report a tool would want instead). Every config in
`validator/configs/` reproduces the exact numbers already reported above --
e.g. `hyperbolic_v3_cautionary.yaml` re-runs the known-blown-up v3 checkpoint
and correctly comes back `status: "fail"` with `max |pred| = 8.28e+06`,
matching the blow-up trace recorded in the hyperbolic section. See
`validator/config.py` for the full config schema and `validator/checks/` for
each check's implementation.

**This is not tied to the shipped checkpoints.** Every config just points at
a model class + `init_kwargs` + checkpoint path + dataset -- if you change
hidden_dim, add layers, swap the architecture, or retrain on different data
(different viscosity, domain, mesh resolution), write a new YAML pointing at
the new checkpoint and rerun. The only real constraint is the calling
convention: either the model exposes `.rollout(u0, pos, edge_index,
edge_attr, n_steps)` (every stage above does) or, if you change that
interface, you supply an external adapter function via `rollout_fn` in the
config (see `hyperbolic_cole_hopf_v2.yaml`, which rolls out in a transformed
coordinate rather than calling the model's own `.rollout()`). A genuinely
new PDE class only needs one new entry in `validator/checks/conservation.py`'s
registry, not a redesign.

### Training a new checkpoint and validating it

Every stage's training script already uses a fixed seed (`SEED` near the top
of `train.py` / `train_v4.py`), so re-running any of the commands under each
stage's "Running it" section above reproduces that stage's shipped result
byte-for-byte. To validate a *new* checkpoint -- one you trained yourself,
whether reproducing an existing run or with changed hyperparameters/data --
copy the closest existing config in `validator/configs/`, point `checkpoint`
(and `init_kwargs`, if the architecture changed) at your new `.pt` file, and
run it:

```
cd hyperbolic/scripts
python train_v4.py --fresh --epochs 20 --total-epochs 20    # produces ../models/best_model_v4.pt
cd ../..
python -m validator.cli --config validator/configs/hyperbolic_v4.yaml
```

The printed summary tells you directly whether anything's wrong -- divergence,
NaN, an accuracy threshold miss, a conservation-residual violation -- and,
for each, a short explanation of what that failure mode usually means (see
`validator/report.py: format_summary`). This is the intended way for someone
new to GNN-PDE surrogates to get a real answer to "did my training run work"
without needing to already know what to look for in a loss curve.

### Dashboard: viewing computed results

`GNNPDEValidationUI.jsx` is a self-contained React dashboard (no other
project files import it) that now loads real computed results instead of
hand-typed numbers. Regenerate the data after training or re-validating
anything, then view it:

```
python -m validator.build_dashboard_data   # writes validator/results/dashboard_data.json

# one-time: bundle the component (needs Node; only re-run this if you edit
# GNNPDEValidationUI.jsx itself, not after every validator run)
npx esbuild GNNPDEValidationUI.jsx --loader:.jsx=jsx --jsx=transform --bundle \
  --format=iife --global-name=GNNPDEBundle --alias:react=./react-global-shim.js \
  --outfile=dashboard.bundle.js

python -m http.server 8000   # fetch() needs http://, not a file:// URL
# open http://localhost:8000/index.html
```

`dashboard.bundle.js` is generated (gitignored) -- only `index.html`,
`react-global-shim.js`, and `GNNPDEValidationUI.jsx` itself are checked in.
The dashboard fetches `validator/results/dashboard_data.json` at load time
and falls back to a frozen offline sample (`FALLBACK_CHECKPOINTS` in the
JSX) if that fetch fails, so it still renders something useful if you drop
the component into a different project. The narrative text per checkpoint
(tagline/note/which checkpoint to compare against) lives in
`validator/configs/dashboard_meta.yaml`, separate from the computed numbers
-- add an entry there for any new config you write, or its tagline/note will
be blank.
