"""
Uncertain linearised cart-pole, re-created from src/model-uncertainty/setup_mass.jl
and src/model-uncertainty/models.jl. Batch dimension is first (Julia has it last).
"""
from pathlib import Path

import numpy as np
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules.normalization import EmpiricalDiscountedVariationNormalization
from tensordict import TensorDict

DATA_DIR = Path(__file__).parent / "data"

# Linearised (uncertain) cartpole system
mc = 1.0
l = 10.0
g = 9.81
mp_range = (0.14, 0.35)
x0_max = np.array([5, 0.5, 1.0, 0.5])
dt = 0.02


def A(mp):
    Ac = np.array([[0, 1, 0, 0],
                   [0, 0, -mp * g / mc, 0],
                   [0, 0, 0, 1],
                   [0, 0, (mc + mp) * g / (mc * l), 0]])
    return np.eye(4) + dt * Ac


B = dt * np.array([0, 1 / mc, 0, -1 / mc])

# Noise variances (not stdev)
σw = 1e-3 * np.ones(4) * dt
σv = 1e-4 / dt

# Cost weights Q = diag(1, 0, 1, 0), R = 1
q = np.array([1.0, 0.0, 1.0, 0.0])


def cost(x, u):
    """Per-environment stage cost x'Qx + u'Ru. Shapes (n, 4), (n, 1) -> (n,)"""
    return (x**2) @ q + u[:, 0]**2


def dynamics(As, x, u, w):
    """x+ = A(mp) x + B u + w for each environment."""
    return np.einsum("nij,nj->ni", As, x) + u * B + w


def measurement(x, v):
    """y = C x + v with C = [1 0 0 0]. Returns shape (n, 1)."""
    return x[:, :1] + v


########### Training environment (torch, for RSL-RL) ###########

class CartPoleEnv(VecEnv):
    """
    Batched cart-pole matching the APG training setup: fixed, evenly-spaced
    pole masses, random initial states, Gaussian process and measurement noise.
    Episodes are truncated (time out) after `max_steps`.

    Observation groups:
    - "policy": the measurement y (cart position + noise).
    - "privileged": the true state x, for the critic only.

    For PPO only:
    - An episode is terminated early if any |state| exceeds `x_term`, with a
      penalty of (current state cost) x (steps remaining in the episode), so
      that failing early is never better than surviving.
    - Rewards are scaled by the running std of the discounted return
      (RSL-RL's normaliser, equivalent to SB3's VecNormalize).
    """

    def __init__(self, num_envs=64, max_steps=800, gamma=0.99, x_term=100.0, seed=0, device="cuda"):
        self.num_envs = num_envs
        self.num_actions = 1
        self.max_episode_length = max_steps
        self.x_term = x_term
        self.device = device
        self.cfg = {}

        tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        mps = np.linspace(*mp_range, num_envs)
        self.As = tensor(np.stack([A(mp) for mp in mps]))
        self.B = tensor(B)
        self.q = tensor(q)
        self.x0_max = tensor(x0_max)
        self.σw = tensor(np.sqrt(σw))
        self.σv = float(np.sqrt(σv))

        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.reward_normalizer = EmpiricalDiscountedVariationNormalization(1, gamma=gamma).to(device)
        self.finished_lengths = []

        self.x = torch.zeros(num_envs, 4, device=device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        self.obs = self._observe()

    def _randn(self, *shape):
        return torch.randn(shape, generator=self.rng, device=self.device)

    def _reset(self, idx):
        n = int(idx.sum())
        u = torch.rand((n, 4), generator=self.rng, device=self.device)
        self.x[idx] = (2 * u - 1) * self.x0_max
        self.episode_length_buf[idx] = 0

    def _observe(self):
        y = self.x[:, :1] + self.σv * self._randn(self.num_envs, 1)
        return TensorDict({"policy": y, "privileged": self.x.clone()},
                          batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self.obs

    def step(self, actions):
        u = actions.reshape(self.num_envs, 1)
        rewards = -((self.x**2) @ self.q + u[:, 0]**2)
        w = self.σw * self._randn(self.num_envs, 4)
        self.x = torch.einsum("nij,nj->ni", self.As, self.x) + u * self.B + w
        self.episode_length_buf += 1

        # Early termination with cost-to-go penalty, or time out at max_steps
        diverged = self.x.abs().amax(dim=1) > self.x_term
        time_outs = (self.episode_length_buf >= self.max_episode_length) & ~diverged
        remaining = self.max_episode_length - self.episode_length_buf
        rewards = rewards - diverged * ((self.x**2) @ self.q) * remaining
        rewards = self.reward_normalizer(rewards.unsqueeze(1)).squeeze(1)

        dones = diverged | time_outs
        if dones.any():
            self.finished_lengths += self.episode_length_buf[dones].tolist()
            self._reset(dones)

        self.obs = self._observe()
        return self.obs, rewards, dones.long(), {"time_outs": time_outs}


########### Test set (exported from Julia by export_test_set.jl) ###########

def readcsv(name):
    return np.loadtxt(DATA_DIR / name, delimiter=",", ndmin=2)


def load_test_set():
    """Initial states, noise, and A-matrices identical to the Julia test set."""
    x0 = readcsv("x0.csv")                                  # (n, 4)
    n = x0.shape[0]
    w = readcsv("w.csv").reshape(-1, n, 4)                  # (T, n, 4)
    v = readcsv("v.csv").reshape(-1, n, 1)                  # (T, n, 1)
    As = readcsv("A.csv").reshape(n, 4, 4)                  # (n, 4, 4)
    return dict(x0=x0, w=w, v=v, As=As)


def load_long_test_set():
    """As `load_test_set`, but 12 x 800 steps long (raw Float64, column-major)."""
    test = load_test_set()
    n = test["x0"].shape[0]
    read = lambda name, m: np.fromfile(DATA_DIR / name).reshape(-1, n, m)
    test["w"], test["v"] = read("w_long.bin", 4), read("v_long.bin", 1)
    return test


def rollout_costs(policy, test):
    """
    Stage cost at each time step (mean over envs) of a policy
    `policy(y, state, first) -> (u, state)` on the test set.
    """
    x, w, v, As = test["x0"], test["w"], test["v"], test["As"]
    T = w.shape[0]
    costs, state = np.zeros(T), None
    for t in range(T):
        y = measurement(x, v[t])
        u, state = policy(y, state, t == 0)
        u = np.asarray(u, dtype=np.float64)
        costs[t] = np.mean(cost(x, u))
        x = dynamics(As, x, u, w[t])
    return costs


def simulate(policy, test):
    """Time-averaged test cost (mean over envs and time)."""
    return np.mean(rollout_costs(policy, test))


def rsl_policy(actor, device):
    """Wrap an RSL-RL actor (in eval mode) as a deterministic policy for `simulate`."""
    @torch.inference_mode()
    def policy(y, state, first):
        if first:
            actor.reset()
        y = torch.tensor(y, dtype=torch.float32, device=device)
        obs = TensorDict({"policy": y}, batch_size=[y.shape[0]], device=device)
        return actor(obs).cpu().numpy(), None
    return policy


def lqg_policy(K, L, Anom):
    """Base LQG controller (static Kalman filter + LQR gain), as in setup_mass.jl."""
    def policy(y, xh, first):
        xh = np.zeros((y.shape[0], 4)) if first else xh
        u = -xh @ K.T
        xh_next = xh @ Anom.T + u * B + (y - xh[:, :1]) @ L.T
        return u, xh_next
    return policy
