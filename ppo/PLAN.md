# PPO baselines for the uncertain cart-pole (Reviewer 4, Comment 2)

## Goal

Add black-box LSTM policies trained with PPO to Fig. 13
(`lcp_vanilla_combined_training.pdf`), with the MLP removed.

## Rules

- The experiment setup does not change. Only the optimiser changes (APG → PPO).
- Nothing outside `ppo/` is modified. Nothing is added/committed/pushed to git.
- Python is used for training only; Julia is used to export the test set and to plot.

## Approach

PPO from [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib==6.0.0`,
ETH Zurich RSL; the PPO used in legged_gym and Isaac Lab), unmodified. The
environment is re-written in torch and runs on the GPU.

| Existing APG setup | PPO equivalent |
|---|---|
| 64 envs, masses `LinRange(0.14, 0.35, 64)`, fixed | Same |
| Random x0 ∈ ±`x0_max`, Gaussian noise (same variances) | Same |
| 200-step training window, reset every 800 steps | 200 steps per env per update; episode times out at 800 |
| No early termination | Training only: episode terminated if any \|state\| > 100, penalty = state cost × steps remaining |
| Minimise mean of x'Qx + u'Ru | Reward = −(x'Qx + u'Ru), scaled by running std of discounted return |
| Policy input is y only, no base controller | Same |
| Test every 5 epochs: seed 1, 64 envs, 1600 steps | Same, on the identical test set exported from Julia; deterministic actions, true cost, no termination |
| 6400 epochs, lr cut 10× at 3/4 and 7/8 | Same |

One PPO update uses the same 64 × 200 = 12,800 simulated steps as one APG
epoch, so the x-axis ("training epochs") means equal simulated data.

### Policies and critic

- LSTM policy: y → LSTM(154) → linear → u. (An MLP policy, y → 4 × 177 ReLU → u,
  was dropped after the pilot.)
- Critic: MLP 4 × 177 ReLU, input (y, true state x). Asymmetric
  actor-critic, standard in sim-to-real RL. The policy never sees x.
- Observation normalisation (RSL-RL `obs_normalization`, running mean/std, no
  clipping) in front of both networks; frozen at test time.
- Gaussian exploration with learned state-independent std (init 1.0).
- Unbounded actions (no squashing), as in the APG setup.

### Hyperparameters

RSL-RL PPO defaults (no learning-rate sweep, default entropy), except γ and the lr cut:

| Setting | Value |
|---|---|
| Envs × steps per update | 64 × 200 (matches APG) |
| Minibatches × passes per update | 4 × 5 |
| Learning rate | adaptive (KL target 0.01), initial 1e-3, range [1e-5, 1e-2]; fixed at 10× below its current value at epoch 4800, 10× lower again at 5600 |
| Clip ratio, GAE λ, entropy coef, value coef | 0.2, 0.95, 0.01, 1.0 |
| Gradient clipping | norm 1.0, actor and critic separately |
| `gamma` | 0.999 (default 0.99 did not settle, see history) |
| Seeds | 6 (matches APG: 6 seeds per learning rate) |

## History: why these choices

1. **SB3, raw observations:** PPO-MLP collapsed within a few updates at every
   learning rate tried (3e-5 to 3e-4). The plant is open-loop unstable, so y
   grows to ~1e7 by the end of an episode; tiny weight steps then move the
   action by ~1e3, PPO's ratios and critic blow up, and the ReLUs die.
   Gradient clipping and `target_kl` don't help (they limit the weight step,
   not the input scale).
2. **Observation normalisation alone:** the running std of y was set by the
   diverging trajectories (5.8e6), so normal-range y mapped to a constant and
   the policy could not distinguish states.
3. **Early termination on divergence** (standard in RL cart-pole benchmarks)
   keeps training data bounded; then normalisation works (y = 5 m → ~0.19) and
   training is numerically healthy. The penalty means failing early is never
   better than surviving. The base LQG never exceeds |state| = 38 on the test
   set, so stabilising policies are never terminated.
4. **Switch to RSL-RL:** SB3 RecurrentPPO was slow (~5 s/update) and SB3 has
   no recurrent policy with a true-state critic. RSL-RL supports this out of
   the box and runs fully on the GPU (~0.24 s/update for MLP and LSTM).

5. **Pilot (RSL-RL, seed 0, stopped at ~epoch 600):** LSTM with γ = 0.99 learned
   to survive most training episodes (length 250 → 773 of 800; test cost
   5e28 → 3e19). γ = 0.999 was much slower (length ~190). PPO-MLP only delayed
   divergence (~250 → ~310), as expected from observability. Decision: drop the
   MLP from the PPO experiments, keep the RSL-RL defaults (γ = 0.99).

6. **Defaults runs (6 seeds, stopped at epoch 4300):**
   all seeds survive the 800-step training episodes from ~epoch 1000, but the
   1600-step test cost bounces between ~1e3 and ~1e30 with no settling or trend.
   Likely cause: with γ = 0.99 PPO only looks ~100 steps ahead, so slow
   long-horizon instability is barely penalised.
7. **γ = 0.999, entropy off, lr cut (failed):**
   without the entropy bonus the exploration std collapsed (1 → ~5e-5 by epoch
   800), the adaptive lr hit its floor, and learning stopped at untrained-level
   episode lengths (~220–250). Noise only adds cost on an unstable plant, so
   PPO removes it before learning anything.
8. **Final config:** γ = 0.999 and an APG-style lr cut (10× at 3/4 and again
   at 7/8 of training); everything else RSL-RL default (entropy 0.01). 6 seeds,
   6400 epochs, `results/`. Seeds 1, 2 and 5 reached costs comparable to the
   Youla-γREN (best 30.5, 10.8, 9.8) but none kept them to the end: seed 2
   collapsed as its exploration std grew (3 → 22), seed 5 held ~21–27 through
   the lr cuts and then drifted up from epoch ~6000 (to ~1e9 at 6400). Seeds
   0, 3 and 4 never stabilised the system. See the README for the table.

## Plotting

- `lcp_vanilla_ppo_combined_training.pdf` (paper): same size (412 × 240 pt) and
  colours as the original Fig. 13, MLP removed. Two panels, "Analytic Gradients"
  (left) and "PPO" (right), shared log y-axis, shared x-label and a two-row
  boxed legend along the bottom. Black-box LSTMs in the same blue in both panels, best
  APG seed highlighted as before. Youla-γREN band and base/LQG references in both.

## Folder layout

```
ppo/
  PLAN.md               this file
  README.md             install and run instructions
  pyproject.toml        uv project
  export_test_set.jl    dumps test set + reference costs from Julia
  cartpole.py           torch cart-pole env (RSL-RL) + test-set evaluation
  train_ppo.py          trains one seed
  run_sweep.sh          runs all 6 seeds in parallel on the GPU
  plot_ppo.jl           the figure
  data/                 exported test set
  results/              PPO cost curves (CSV), logs, models, figure
```

## Checklist

- [x] Agree on plan
- [x] Install Julia deps (`Pkg.instantiate()` with Julia 1.10.14; 1.10.4 won't start on this kernel). Project/Manifest unchanged
- [x] Export test set from Julia
- [x] Python env + validation: base LQG test cost 404.9976151746354 in both Python and Julia (exact)
- [x] SB3 pilots: collapse with raw observations; fixed with obs normalisation + early termination; too slow for LSTM
- [x] Switch to RSL-RL with true-state MLP critic; smoke test healthy, ~0.24 s/epoch
- [x] Pilot: γ = 0.99 clearly better; LSTM learns; MLP dropped
- [x] Full runs with RSL-RL defaults: LSTM × 6 seeds (CSV and model saved every 100 epochs)
- [x] Defaults runs stopped at epoch 4300: not settling
- [x] γ = 0.999 with entropy off: exploration collapsed, no learning (stopped)
- [x] γ = 0.999 + lr cut, default entropy, seeds 0–2: seed 2 settled at a good cost (epochs ~1300–2800) then collapsed as the exploration std grew (3 → 22); seed 0 failed; seed 1 slow
- [x] Decision: stop tuning; final config = γ = 0.999 + lr cut, otherwise RSL-RL defaults
- [x] Seeds 3–5 of the final config: seed 5 good from ~980 to ~6000 then drifted up; seeds 3, 4 never learned
- [x] Plot scripts written and tested on early checkpoints (APG panel matches published Fig. 13)
- [x] Long test rollouts: not needed (PPO not better than APG). Code and results moved to `to_delete/`
- [x] Final figures: paper version split into two panels with a bottom legend (same footprint)
- [x] Unneeded files moved to `to_delete/` (rollout code/results, γ = 0.99 defaults runs, entropy-off runs, long test set)
- [x] Clean up at the very end (after seeds 3–5 finish, since they use the current code): cut the code and saved results down to only the files needed to generate the data and figures for the final plots (e.g. drop unused MLP / rollout / sweep options and exploratory results), then update `ppo/README.md` to match
  - Done: LSTM-only code with the final config hard-coded; long test set and LQG validation removed from the export (validation kept as `to_delete/validate_lqg.py`, still matches Julia); results flattened into `results/ppo_lstm_s*.{csv,log}`; figures regenerated and unchanged; seed 0 retrained for 8 epochs and matched the saved run exactly
- [x] User to review, then delete `to_delete/`
