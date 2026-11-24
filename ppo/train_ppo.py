"""
Train a black-box MLP or LSTM policy on the uncertain cart-pole with PPO (RSL-RL).

The policy sees only the measurement y. The critic (always an MLP) also sees
the true state x (asymmetric actor-critic).

Example:
    uv run python train_ppo.py --policy lstm --gamma 0.99 --seed 0
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO

from cartpole import CartPoleEnv, load_test_set, rsl_policy, simulate

RESULTS_DIR = Path(__file__).parent / "results"

# Match the APG setup (train_vanilla_batch.jl)
train_batches = 64
train_horizon = 200
max_steps = 800
nv_lstm = 154
nv_mlp = 177


def make_cfg(policy, gamma, entropy_coef=0.01):
    """RSL-RL configuration. Other PPO hyperparameters are the RSL-RL defaults."""
    gaussian = {"class_name": "GaussianDistribution", "init_std": 1.0}
    mlp = {"class_name": "MLPModel", "hidden_dims": [nv_mlp] * 4,
           "activation": "relu", "obs_normalization": True}
    if policy == "mlp":
        actor = {**mlp, "distribution_cfg": gaussian}
    else:
        actor = {"class_name": "RNNModel", "rnn_type": "lstm", "rnn_hidden_dim": nv_lstm,
                 "network_cfg": {"class_name": "torch.nn:Linear"},  # linear readout
                 "obs_normalization": True, "distribution_cfg": gaussian}
    return {
        "num_steps_per_env": train_horizon,
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "algorithm": {"class_name": "PPO", "gamma": gamma, "entropy_coef": entropy_coef},
        "actor": actor,
        "critic": mlp,
        "multi_gpu": None,
    }


def test_cost(alg, test, device):
    """
    Test cost of the deterministic policy. The actor's hidden state is saved
    and restored so that training rollouts are unaffected, and eval mode stops
    the observation normaliser from updating on test data.
    """
    hidden = alg.actor.get_hidden_state()
    if hidden is not None:
        hidden = tuple(h.clone() for h in hidden)
    alg.eval_mode()
    J = simulate(rsl_policy(alg.actor, device), test)
    alg.train_mode()
    alg.actor.reset(hidden_state=hidden)
    return J


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["mlp", "lstm"], required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=6400)
    parser.add_argument("--lr-decay", action="store_true",
                        help="Cut the lr by 10x at 3/4 and again at 7/8 of training (as in APG)")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="", help="Optional sub-folder in results/")
    args = parser.parse_args()

    savedir = RESULTS_DIR / args.tag
    savedir.mkdir(parents=True, exist_ok=True)
    save_name = f"ppo_{args.policy}_g{args.gamma:g}_ent{args.entropy_coef:g}_s{args.seed}"
    print("Starting experiment", save_name, flush=True)

    torch.manual_seed(args.seed)
    env = CartPoleEnv(train_batches, max_steps, args.gamma, seed=args.seed, device=args.device)
    obs = env.get_observations()
    cfg = make_cfg(args.policy, args.gamma, args.entropy_coef)
    alg = PPO.construct_algorithm(obs, env, cfg, args.device)
    decay_epochs = [int(args.epochs * f) for f in (3/4, 7/8)] if args.lr_decay else []
    test = load_test_set()

    # Save test costs as we go, and checkpoint the model (includes obs normalisers)
    def save(costs):
        np.savetxt(savedir / f"{save_name}.csv", np.array(costs),
                   delimiter=",", header="epoch,test_cost,ep_len_mean", comments="")
        torch.save(alg.save(), savedir / f"{save_name}_model.pt")

    # Training loop (same as rsl_rl's OnPolicyRunner.learn, minus its logging),
    # logging the test cost every few epochs as in train_model!
    costs = []
    t0 = time.time()
    alg.train_mode()
    for epoch in range(args.epochs + 1):

        if epoch % args.log_every == 0:
            J = test_cost(alg, test, args.device)
            ep_len = np.mean(env.finished_lengths[-100:]) if env.finished_lengths else np.nan
            costs.append((epoch, J, ep_len))
            if epoch % 100 == 0:
                std = alg.actor.distribution.std_param.mean().item()
                print(f"Epoch {epoch} test cost: {J:.4g}, episode length: {ep_len:.0f}, "
                      f"action std: {std:.3g}, lr: {alg.learning_rate:.2g} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            if epoch % args.save_every == 0:
                save(costs)
        if epoch == args.epochs:
            break

        # Fix the lr at 10x below its current (adaptive) value, as in the APG runs
        if epoch in decay_epochs:
            alg.schedule = "fixed"
            alg.learning_rate *= 0.1
            for group in alg.optimizer.param_groups:
                group["lr"] = alg.learning_rate
            print(f"Epoch {epoch}: learning rate fixed at {alg.learning_rate:.2g}", flush=True)

        with torch.inference_mode():
            for _ in range(train_horizon):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            alg.compute_returns(obs)
        alg.update()

    save(costs)
    print("Done", save_name, flush=True)


if __name__ == "__main__":
    main()
