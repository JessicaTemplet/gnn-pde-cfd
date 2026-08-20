# When your training loss is a liar and your rollout is the one telling the truth

I've been training graph neural networks to solve PDEs, one equation class at a time: elliptic, then hyperbolic, then parabolic. The hyperbolic stage (Burgers' equation, `u_t + u u_x = nu u_xx`) turned into a proper debugging story, so here it is.

## The setup

The model is a standard encode-process-decode message-passing GNN, periodic chain graph with multiscale shortcut edges at strides 1/2/4/8 so a 2-layer network gets full-domain receptive field in a couple of hops. Predicts the increment `du = u_{t+dt} - u_t` rather than the absolute next state, so an untrained network defaults to "nothing changes" instead of "random garbage" when you roll it out. 80 training trajectories, each a smooth initial condition that forms a shock by t=0.5.

The evaluation that matters is autoregressive rollout: start from a test initial condition, feed the model's own output back as input for 100 steps, compare against a reference spectral solver. This is the thing we actually care about. The training loss is a one-step MSE against independent (u_t, u_{t+dt}) pairs, which is much cheaper to compute but is only a proxy for rollout quality.

These two metrics do not always agree. Figuring out when they disagree and why is basically the whole story.

## v1: looks fine, isn't fine

First version trains as a plain one-step regressor. Training and validation loss converge together, no overfitting, val MSE of 4.8e-4 against a "predict no change" baseline of 5.7e-4. By training metrics, this is a working model.

The rollout says otherwise. Feed it a test initial condition and let it run 100 steps: the shock steepens correctly for the first ~20 steps, then just... stops. The reference solution's shock front is supposed to advect across the domain. The v1 rollout's shock freezes in place and slowly decays in amplitude. Mean relative L2 error against the reference climbs to 0.80 within the first fifth of the trajectory and sits there.

One-step training has no incentive to get propagation speed right. A model that predicts "small change, basically no advection" is already near-optimal for the MSE on independently sampled pairs, because each individual step is small. So nothing in training ever pushed the model to actually move the shock.

## v2: real improvement, not a full fix

Warm-start from v1 weights (the local one-step dynamics are fine, no need to relearn those) and fine-tune with an unrolled multi-step loss: 5 consecutive steps, model output fed back as input, loss averaged over all 5 steps against the true trajectory. This directly penalizes the stall, because a stalled shock diverges from the reference over those 5 steps in a way a single-step loss can never see.

Before running a full training pass I verified the unrolled loss itself wasn't broken by overfitting a single fixed batch. Loss went from 0.087 to 0.0024 over 80 iterations, gradients flowing correctly through the backprop-through-time chain. Worth doing.

Result: final-time rollout error dropped from 0.80 to 0.65. Consistent improvement across the whole trajectory, not just at one time point. The shock still stalls visibly, 15 epochs of fine-tuning on a small model wasn't enough to fully relearn the correct propagation speed, but it measurably dented the bias toward zero velocity. v2 shipped as the stable model. Then I asked the obvious question.

## v3: the other shoe drops

If 15 epochs of unrolled fine-tuning improved things, does more help further? v3 continues from v2 weights with the same unroll-k=5 for 13 more epochs. Training and validation loss kept going down the whole time, 0.0038 to 0.0025, smooth, no noise. By that metric: clean continued win.

Full-rollout evaluation at epoch 13: max field magnitude of 8.3 million against a true field scale of about 3.

I ran it back and looked at the raw prediction magnitude at every step:

```
t=0.000  max|pred|=2.97      t=0.200  max|pred|=642
t=0.025  max|pred|=3.42      t=0.300  max|pred|=14,833
t=0.050  max|pred|=5.85      t=0.400  max|pred|=348,316
t=0.100  max|pred|=29.3      t=0.500  max|pred|=8,283,525
```

That's not a delayed-onset transient. It's not the shock stalling in a new shape. It's clean exponential growth from step one, roughly 2.2x every 5 model steps. Which is 1.17x per single step. Which is invisible at k=5 (1.17^5 is 2.2x, consistent with normal loss noise) but is 1.17^100 over a full 100-step rollout, which is roughly 10^7. We had a come-to-Jessica meeting and I put v3 on notice. v2 remains the shipped model.

Worth flagging: the spacetime plots were initially misleading here. Matplotlib silently clamped its color scale to the reference field's range, so the spacetime visualization looked like an ordinary stall until I pulled the raw numbers. The raw magnitude trace is the check that actually catches this failure mode.

## The mechanism

A fixed unroll horizon k cannot penalize a per-step amplification factor that is invisible at that k but catastrophic at 100 steps. Training exclusively at k=5 gives the model no signal about behavior at k=6, 7, or 100. The model is perfectly free to develop a per-step spectral radius slightly above 1 as long as the growth is small enough to be lost in the loss at k=5. And it did exactly that.

This is also why v3's training loss kept going down smoothly: within the 5-step window it was genuinely improving. The instability only shows up when you actually run it for 100 steps, which the training loop never does.

## v4: curriculum unrolling

The fix is to not fix k. Curriculum unrolling: k grows linearly from 1 to 15 over the full training run. At epoch 1, k=1, we're basically back to one-step training (enough to start rebuilding the shock dynamics from scratch). By epoch 10, k=8. At 1.17x per step, 1.17^8 is 3.7x over the window, clearly penalized. By k=15, anything with a per-step radius meaningfully above 1 is costing real loss.

Two additions from v3's lessons: gradient clipping at norm 1.0 (long unrolled gradients can spike as k grows, clipping keeps the optimizer from making destabilizing updates on rare bad batches) and a per-chunk rollout spot check (run one full 100-step rollout at the end of each training chunk, print the final relative L2 and max prediction magnitude, append to history). The spot check is the thing that would have caught v3's blowup immediately. It's also how I know v4 is actually working: all 4 spot checks across 20 epochs came back with max|pred|=2.97, exactly matching the true field scale.

Training results: 20 epochs, val loss improved from 0.018 to 0.011 (best at epoch 17, k=13). Final rollout error 0.626 vs v2's 0.648, most of the gain in the early trajectory where the shock is steepening (25% mark: 0.488 vs 0.582). Zero diverged trajectories across all 15 test runs. The curriculum did what the theory said it would.

## The control case

After all of that I ran the parabolic stage: 1D heat equation, `u_t = nu u_xx`, same architecture, same graph, same one-step training recipe as v1.

Final rollout error: 0.07. Flat across the entire trajectory. 5.3% error at the 25% mark, 7.0% at the end, no growth trend.

That's 9x lower than hyperbolic v4 with curriculum unrolling, achieved with the exact training approach that failed catastrophically on Burgers'.

The heat equation is dissipative: small prediction errors get smoothed out by the dynamics rather than advected and amplified. The v1 training failure on Burgers' wasn't a GNN training pathology. It was specific to the hyperbolic nature of Burgers' equation, where the dynamics amplify rather than damp perturbations. The parabolic result is the control that proves the diagnosis.

## One more thing: the Cole-Hopf detour

After all of that, the obvious question was whether there was a smarter representation than u itself. The Cole-Hopf transformation maps Burgers' to the linear heat equation: if phi satisfies phi_t = nu*phi_xx, then u = -2*nu * phi_x / phi solves Burgers'. We have a trained heat GNN. Can we just plug it in?

First attempt (Cole-Hopf v1): transform u_0 to phi, step phi with the heat GNN, invert back to u. Failed badly: 10 of 15 test trajectories diverged, max prediction magnitude 1160 against a true scale of 3. The problem wasn't the nu mismatch (heat model trained at nu=0.01, Burgers' at nu=0.02). The problem was that the inverse u = -2*nu * phi_x / phi requires phi > 0, but the GNN outputs unconstrained real values. Enforcing positivity by shifting phi destroys the spatial gradient structure that the inverse depends on, and the error compounds at every step.

The fix was to notice that if you work in psi = log(phi) instead of phi directly, the inverse simplifies to u = -2*nu * psi_x (just a spectral derivative, no division, no positivity constraint). Train a GNN to advance psi one step -- using psi data generated from Cole-Hopf applied to the actual Burgers' trajectories -- and you have a model that lives entirely in unconstrained real-valued space, with a clean inverse.

Results: 0 diverged trajectories, max prediction magnitude exactly matching the true field scale, relative L2 error 0.105 at the 25% mark and 0.237 at the end. Compared to v4's 0.488 and 0.626, that's 4.6x and 2.6x better with plain one-step training, 30 epochs, same architecture as v1.

Why the representation helps so much: psi is the antiderivative of u (scaled). A shock in u is a kink in psi -- the field the GNN actually regresses on is smoother than u by one derivative. The model doesn't have to directly represent a propagating discontinuity; it models a diffusing primitive function. The learning problem was genuinely easier.

## What this actually means if you're training neural PDE surrogates

When your training metric and your rollout metric disagree, the rollout is telling the truth. One-step validation loss is a useful sanity check but it cannot measure propagation speed, spectral radius, or any property that only shows up over many steps.

Unrolled training helps, but fixed-horizon unrolling has a structural blindspot: it constrains rollout behavior at k steps but says nothing about k+1 or k+100. Curriculum unrolling keeps the model honest at every step of training.

Representation matters more than training recipe. The Cole-Hopf v2 result with plain one-step training beats the best unrolled approach by 4.6x at the 25% mark. If the right coordinate system makes the dynamics smoother and the inverse transform cleaner, that's worth more than any amount of curriculum engineering in the wrong space.

And if you're trying to figure out whether a training instability is a fundamental problem or a PDE-specific problem: run the same setup on a dissipative equation and see what happens. It took about 30 minutes of training time and told me more than reading papers about it would have.
