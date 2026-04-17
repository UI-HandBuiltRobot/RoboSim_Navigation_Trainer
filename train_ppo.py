"""PPO fine-tuning for IL-trained MLP navigation policies.

Loads an IL model JSON (weights, biases, x_scale, y_scale, activation, layer
sizes, mode, memory_steps), warm-starts a torch actor with identical topology,
runs PPO against parallel HeadlessSimulator workers, evaluates against the
maps in a benchmark folder, and saves the improved policy back into the same
IL JSON schema (so benchmark_gui.py and the simulator load it identically).

Reward shaping: terminal-dominant. Success pays a large flat bonus plus a
time-inverse bonus (capped, success-only). Collisions and stuck both pay a
strong flat penalty. Per-step shaping is intentionally weak so the policy
optimises for ultimate outcome, not local greedy behaviour. A near-instant
collision can never earn the time bonus because the bonus is only paid on
success.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

from headless_sim import HeadlessSimulator
from sim_core import Robot


# --- Constants -------------------------------------------------------------

# Env action layout (order accepted by HeadlessSimulator.step) per control mode.
ENV_ACTION_LAYOUTS: Dict[str, List[str]] = {
    "heading_drive":  ["rotation_rate", "drive_speed"],
    "heading_strafe": ["rotation_rate", "vx", "vy"],
    "xy_strafe":      ["vx", "vy"],
}

# Physical bounds; env actions are normalized by these.
ACTION_PHYSICAL_BOUND = {
    "rotation_rate": Robot.GAMEPAD_MAX_ROTATE_RATE_DPS,  # 40.0 dps
    "drive_speed":   Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,  # 0.40 mps
    "vx":            Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
    "vy":            Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
}

# Reward shaping defaults. Designed so terminal events dominate. The success
# time bonus is capped at SUCCESS_TIME_BONUS_AMP, achieved only as t -> 0,
# but it is never paid on collision/stuck so it cannot offset a failure.
REWARD_PARAMS = {
    "distance_reduction_scale": 0.05,   # weak per-step shaping
    "timestep_penalty":         -0.002, # mild time pressure
    "collision_penalty":         0.0,   # we use the fail penalty instead
    "collision_fail_penalty":   -50.0,  # collision (terminal) -- strong
    "timeout_penalty":          -50.0,  # stuck (terminal)     -- strong, == collision
    "target_reached_bonus":     100.0,  # success base; > any time bonus
    "terminate_on_collision":   True,
}
SUCCESS_TIME_BONUS_AMP = 30.0  # max extra reward for instant success
SUCCESS_TIME_BONUS_TAU = 30.0  # seconds; bonus = AMP * TAU / (TAU + t)

# Stuck/safety: hard episode cap (steps). Sim's own watchdog kicks in earlier.
EPISODE_STEP_HARD_CAP = 12000

# PPO defaults (conservative -- chosen to avoid destroying the warm-start).
DEFAULT_HPARAMS = {
    "n_envs":          4,
    "n_steps":         2048,    # per env per iteration
    "total_iters":     50,
    "gamma":           0.995,
    "gae_lambda":      0.95,
    "clip_range":      0.10,
    "epochs":          10,
    "minibatch_size":  256,
    "lr_actor":        3e-5,
    "lr_critic":       3e-4,
    "vf_coef":         0.5,
    "ent_coef":        5e-4,
    "max_grad_norm":   0.5,
    "kl_target":       0.01,
    "kl_stop_factor":  1.5,
    "log_std_init":    -1.5,            # std ~= 0.22
    "log_std_min":     math.log(0.03),  # allow exploration to shrink as policy improves
    "log_std_max":     math.log(0.7),
    "eval_every":      2,       # iterations
    "device":          "cpu",
}


# --- IL JSON loader --------------------------------------------------------

def load_il_blob(path: Path) -> Dict[str, Any]:
    """Read an IL model JSON and return a dict with parsed numpy arrays."""
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    required = ["weights", "biases", "x_scale", "y_scale", "mode",
                "input_dim", "output_dim", "hidden_layers", "hidden_width",
                "activation", "memory_steps"]
    missing = [k for k in required if k not in blob]
    if missing:
        raise ValueError(f"IL model missing keys: {missing}")

    weights = [np.asarray(w, dtype=np.float32) for w in blob["weights"]]
    biases  = [np.asarray(b, dtype=np.float32).reshape(-1) for b in blob["biases"]]

    return {
        "raw":           blob,
        "weights":       weights,
        "biases":        biases,
        "x_scale":       np.asarray(blob["x_scale"], dtype=np.float32),
        "y_scale":       np.asarray(blob["y_scale"], dtype=np.float32),
        "mode":          str(blob["mode"]),
        "input_dim":     int(blob["input_dim"]),
        "output_dim":    int(blob["output_dim"]),
        "hidden_layers": int(blob["hidden_layers"]),
        "hidden_width":  int(blob["hidden_width"]),
        "activation":    str(blob["activation"]).lower(),
        "memory_steps":  int(blob["memory_steps"]),
        "output_layout": list(blob.get("output_layout", [])),
        "input_layout":  list(blob.get("input_layout", [])),
    }


def model_output_to_env_action_indices(mode: str, output_layout: List[str]) -> np.ndarray:
    """Permutation that reorders model output -> env action layout.

    env_action[i] = model_output[idx[i]] (with appropriate scaling already
    handled because y_scale and GAMEPAD_MAX bounds match per channel).
    """
    env_layout = ENV_ACTION_LAYOUTS[mode]
    if not output_layout:
        # Older IL files might omit it; assume identity (matches strafe modes).
        if len(env_layout) != 2 and len(env_layout) != 3:
            raise ValueError(f"unknown env layout for mode={mode}")
        return np.arange(len(env_layout), dtype=np.int64)
    name_to_model_idx = {name: i for i, name in enumerate(output_layout)}
    idx = []
    for env_name in env_layout:
        if env_name not in name_to_model_idx:
            raise ValueError(
                f"env action '{env_name}' not present in model output_layout {output_layout}"
            )
        idx.append(name_to_model_idx[env_name])
    return np.asarray(idx, dtype=np.int64)


# --- Actor / Critic networks ----------------------------------------------

class MLP(nn.Module):
    """Simple feed-forward MLP, configurable depth/width/activation."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: int, hidden_width: int, activation: str) -> None:
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh,
                   "leaky_relu": lambda: nn.LeakyReLU(0.01)}[activation]
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_width))
            layers.append(act_cls())
            prev = hidden_width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_layers(self) -> List[nn.Linear]:
        return [m for m in self.net.modules() if isinstance(m, nn.Linear)]


def load_il_weights_into_mlp(mlp: MLP, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
    """Copy IL (numpy) parameters into a torch MLP with matching topology."""
    linears = mlp.linear_layers()
    if len(linears) != len(weights):
        raise ValueError(f"layer count mismatch: torch={len(linears)} il={len(weights)}")
    with torch.no_grad():
        for layer, w, b in zip(linears, weights, biases):
            if layer.weight.shape != (w.shape[1], w.shape[0]):
                raise ValueError(
                    f"weight shape mismatch: torch={tuple(layer.weight.shape)} "
                    f"il={tuple(w.shape)} (expected transpose)"
                )
            # IL stores W as (in, out); torch expects (out, in).
            layer.weight.copy_(torch.from_numpy(w.T.astype(np.float32)))
            layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))


class ActorCritic(nn.Module):
    """Warm-started actor + fresh critic. Actor input/output are RAW physical;
    normalisation by x_scale and rescaling by y_scale happen inside forward()
    so callers can pass env observations / receive env actions directly."""

    def __init__(self, il: Dict[str, Any], log_std_init: float,
                 log_std_min: float, log_std_max: float) -> None:
        super().__init__()
        self.input_dim    = il["input_dim"]
        self.output_dim   = il["output_dim"]
        self.activation   = il["activation"]
        self.hidden_layers = il["hidden_layers"]
        self.hidden_width  = il["hidden_width"]

        # Buffers: x_scale (input divisor) / y_scale (output multiplier).
        # Both are saturation bounds; never use to scale grads of policy params.
        x_scale = torch.from_numpy(il["x_scale"].astype(np.float32))
        y_scale = torch.from_numpy(il["y_scale"].astype(np.float32))
        self.register_buffer("x_scale", torch.where(x_scale.abs() < 1e-6,
                                                    torch.ones_like(x_scale), x_scale))
        self.register_buffer("y_scale", torch.where(y_scale.abs() < 1e-6,
                                                    torch.ones_like(y_scale), y_scale))

        # Permutation that maps actor output (model layout) -> env action layout.
        perm = model_output_to_env_action_indices(il["mode"], il["output_layout"])
        self.register_buffer("env_action_perm", torch.from_numpy(perm))

        # Actor warm-started from IL weights.
        self.actor = MLP(self.input_dim, self.output_dim,
                         self.hidden_layers, self.hidden_width, self.activation)
        load_il_weights_into_mlp(self.actor, il["weights"], il["biases"])

        # Critic: fresh, same topology, scalar output. Final layer zero-bias init.
        self.critic = MLP(self.input_dim, 1,
                          self.hidden_layers, self.hidden_width, self.activation)
        with torch.no_grad():
            self.critic.linear_layers()[-1].bias.zero_()

        # Action std (state-independent, learnable).
        self.log_std = nn.Parameter(torch.full((self.output_dim,), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def _normalize_obs(self, raw_obs: torch.Tensor) -> torch.Tensor:
        return raw_obs / self.x_scale

    def policy_mean_normalized(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Mean of the action distribution in MODEL output layout, normalised
        to [-1, 1] saturation space (same units as env action, before perm).
        """
        return self.actor(self._normalize_obs(raw_obs))

    def value(self, raw_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._normalize_obs(raw_obs)).squeeze(-1)

    def clamped_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(self.log_std_min, self.log_std_max)

    def to_env_action(self, model_action: torch.Tensor) -> torch.Tensor:
        """Reorder model-layout action -> env-layout action and clip to [-1, 1]."""
        env_action = model_action.index_select(-1, self.env_action_perm)
        return env_action.clamp(-1.0, 1.0)

    def sample_action(self, raw_obs: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (env_action, model_action_unclipped, log_prob).

        We sample in MODEL layout (so log_std lines up with the policy net's
        output channels). log_prob is computed on the unclipped sample and
        used directly by PPO -- the clip-on-step bias is small at our scales
        and is the standard treatment in clipped-Gaussian PPO.
        """
        mean = self.policy_mean_normalized(raw_obs)
        if deterministic:
            sample = mean
            log_prob = torch.zeros(mean.shape[:-1], device=mean.device)
        else:
            std = self.clamped_log_std().exp().expand_as(mean)
            eps = torch.randn_like(mean)
            sample = mean + std * eps
            # Diagonal Gaussian log prob.
            var = std * std
            log_prob = (-0.5 * ((sample - mean) ** 2) / var
                        - self.clamped_log_std().expand_as(mean)
                        - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        env_action = self.to_env_action(sample)
        return env_action, sample, log_prob

    def evaluate_actions(self, raw_obs: torch.Tensor, model_action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update: log_prob & entropy of given (model-layout) actions."""
        mean = self.policy_mean_normalized(raw_obs)
        log_std = self.clamped_log_std()
        std = log_std.exp().expand_as(mean)
        var = std * std
        log_prob = (-0.5 * ((model_action - mean) ** 2) / var
                    - log_std.expand_as(mean)
                    - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        entropy = (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)
        entropy = entropy.expand(raw_obs.shape[0])
        value = self.value(raw_obs)
        return log_prob, entropy, value


# --- Worker process: one HeadlessSimulator per worker ----------------------

def _worker_main(worker_id: int, control_mode: str, history_window: int,
                 seed: int, n_obstacles_min: int, n_obstacles_max: int,
                 cmd_q: mp.Queue, res_q: mp.Queue) -> None:
    """Long-running worker process. Each worker holds one HeadlessSimulator.

    Each episode samples a uniformly random obstacle count in
    [n_obstacles_min, n_obstacles_max] (inclusive). If both are 0, defers
    to the simulator's built-in default (random 3-10).
    """
    try:
        # RNG for obstacle-count sampling, independent of the sim's own RNG.
        obs_rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15))

        def _sample_obs_count() -> int:
            lo = max(0, int(n_obstacles_min))
            hi = max(lo, int(n_obstacles_max))
            if lo == 0 and hi == 0:
                return 0  # sim default (3-10 random)
            return int(obs_rng.integers(lo, hi + 1))

        env = HeadlessSimulator(
            seed=seed,
            control_mode=control_mode,
            history_window=history_window,
            n_obstacles=_sample_obs_count(),
            distance_reduction_scale=REWARD_PARAMS["distance_reduction_scale"],
            collision_penalty=REWARD_PARAMS["collision_penalty"],
            collision_fail_penalty=REWARD_PARAMS["collision_fail_penalty"],
            timestep_penalty=REWARD_PARAMS["timestep_penalty"],
            target_reached_bonus=REWARD_PARAMS["target_reached_bonus"],
            timeout_penalty=REWARD_PARAMS["timeout_penalty"],
            terminate_on_collision=REWARD_PARAMS["terminate_on_collision"],
        )
        obs = env.reset()
        ep_steps = 0

        while True:
            cmd, payload = cmd_q.get()
            if cmd == "step":
                action = np.asarray(payload, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_steps += 1
                # Hard safety cap.
                if not (terminated or truncated) and ep_steps >= EPISODE_STEP_HARD_CAP:
                    truncated = True
                    info = dict(info)
                    info["stuck_failed"] = True
                # Success-only time bonus.
                shaped_reward = float(reward)
                if terminated and bool(info.get("reached_target", False)):
                    t = float(info.get("simulated_time", 0.0))
                    shaped_reward += SUCCESS_TIME_BONUS_AMP * SUCCESS_TIME_BONUS_TAU \
                                     / (SUCCESS_TIME_BONUS_TAU + t)
                done = bool(terminated or truncated)
                next_obs_for_buffer = obs.copy()
                if done:
                    env.n_obstacles_count = _sample_obs_count()
                    obs = env.reset()
                    ep_steps_reported = ep_steps
                    ep_steps = 0
                else:
                    ep_steps_reported = 0
                res_q.put((
                    "step_ok",
                    next_obs_for_buffer,
                    shaped_reward,
                    done,
                    bool(info.get("reached_target", False)),
                    bool(info.get("collision_failed", False)),
                    bool(info.get("stuck_failed", False)),
                    float(info.get("simulated_time", 0.0)),
                    obs,            # next observation to feed policy (post-reset if done)
                    ep_steps_reported,
                ))
            elif cmd == "reset":
                env.n_obstacles_count = _sample_obs_count()
                obs = env.reset()
                ep_steps = 0
                res_q.put(("reset_ok", obs))
            elif cmd == "shutdown":
                res_q.put(("bye", None))
                return
            else:
                res_q.put(("error", f"unknown cmd {cmd}"))
    except Exception as exc:
        res_q.put(("error", f"worker {worker_id}: {exc}\n{traceback.format_exc()}"))


class VectorEnv:
    """Manages N worker processes; sends actions in lockstep, collects results."""

    def __init__(self, n_envs: int, control_mode: str, history_window: int,
                 base_seed: int, n_obstacles_min: int, n_obstacles_max: int) -> None:
        ctx = mp.get_context("spawn")
        self.n_envs = int(n_envs)
        self.cmd_qs: List[mp.Queue] = []
        self.res_qs: List[mp.Queue] = []
        self.procs: List[mp.Process] = []
        for i in range(self.n_envs):
            cq = ctx.Queue()
            rq = ctx.Queue()
            p = ctx.Process(
                target=_worker_main,
                args=(i, control_mode, history_window,
                      int(base_seed) + i * 10007,
                      int(n_obstacles_min), int(n_obstacles_max), cq, rq),
                daemon=True,
            )
            p.start()
            self.cmd_qs.append(cq)
            self.res_qs.append(rq)
            self.procs.append(p)

        # Get initial observations.
        for cq in self.cmd_qs:
            cq.put(("reset", None))
        self.last_obs = []
        for rq in self.res_qs:
            tag, payload = rq.get()
            if tag != "reset_ok":
                raise RuntimeError(f"worker reset failed: {tag} {payload}")
            self.last_obs.append(np.asarray(payload, dtype=np.float32))

    def current_obs(self) -> np.ndarray:
        return np.stack(self.last_obs, axis=0)

    def step(self, actions: np.ndarray):
        """actions: (n_envs, action_dim). Returns dict of arrays."""
        for cq, a in zip(self.cmd_qs, actions):
            cq.put(("step", a))
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)
        reached = np.zeros(self.n_envs, dtype=np.bool_)
        coll = np.zeros(self.n_envs, dtype=np.bool_)
        stuck = np.zeros(self.n_envs, dtype=np.bool_)
        sim_time = np.zeros(self.n_envs, dtype=np.float32)
        ep_lens = np.zeros(self.n_envs, dtype=np.int64)
        next_obs_buf = []
        next_obs_policy = []
        for i, rq in enumerate(self.res_qs):
            tag, *rest = rq.get()
            if tag != "step_ok":
                raise RuntimeError(f"worker step failed: {tag} {rest}")
            (nob_buf, rew, done, rch, cf, sf, st, nob_pol, ep_len) = rest
            rewards[i] = rew
            dones[i] = done
            reached[i] = rch
            coll[i] = cf
            stuck[i] = sf
            sim_time[i] = st
            ep_lens[i] = ep_len
            next_obs_buf.append(nob_buf)
            next_obs_policy.append(nob_pol)
        self.last_obs = next_obs_policy
        return {
            "rewards":          rewards,
            "dones":            dones,
            "reached":          reached,
            "collision_failed": coll,
            "stuck_failed":     stuck,
            "sim_time":         sim_time,
            "ep_lens":          ep_lens,
            "next_obs_buffer":  np.stack(next_obs_buf, axis=0),  # for bootstrap value
        }

    def close(self) -> None:
        for cq in self.cmd_qs:
            try:
                cq.put(("shutdown", None))
            except Exception:
                pass
        deadline = time.time() + 5.0
        for p in self.procs:
            remaining = max(0.05, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                p.terminate()


# --- Rollout buffer & GAE --------------------------------------------------

@dataclass
class Rollout:
    obs:          np.ndarray   # (T, N, obs_dim)
    actions:      np.ndarray   # (T, N, act_dim)  -- model layout, unclipped
    log_probs:    np.ndarray   # (T, N)
    values:       np.ndarray   # (T, N)
    rewards:      np.ndarray   # (T, N)
    dones:        np.ndarray   # (T, N) bool
    next_value:   np.ndarray   # (N,)
    advantages:   np.ndarray = field(default=None)  # filled by compute_gae
    returns:      np.ndarray = field(default=None)


def compute_gae(roll: Rollout, gamma: float, lam: float) -> None:
    T, N = roll.rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    next_v = roll.next_value.astype(np.float32)
    next_nonterminal = 1.0 - roll.dones[-1].astype(np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            v_next = next_v
            mask = next_nonterminal
        else:
            v_next = roll.values[t + 1]
            mask = 1.0 - roll.dones[t].astype(np.float32)
        delta = roll.rewards[t] + gamma * v_next * mask - roll.values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    roll.advantages = adv
    roll.returns = adv + roll.values


# --- Benchmark evaluation -------------------------------------------------

def load_benchmark_maps(folder: Path) -> List[Dict[str, Any]]:
    map_files = sorted(folder.glob("map_*.json"))
    if not map_files:
        raise FileNotFoundError(f"no map_*.json files found in {folder}")
    out = []
    for p in map_files:
        with open(p, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def evaluate_policy_on_maps(model: ActorCritic, scenarios: List[Dict[str, Any]],
                            il_meta: Dict[str, Any], device: torch.device
                            ) -> Dict[str, float]:
    model.eval()
    success = 0
    success_times: List[float] = []
    all_times: List[float] = []
    coll_fails = 0
    stuck_fails = 0
    n = len(scenarios)
    history_window = il_meta["memory_steps"]
    mode = il_meta["mode"]

    for sc in scenarios:
        env = HeadlessSimulator(
            seed=int(sc.get("seed", 0)),
            control_mode=mode,
            history_window=history_window,
            n_obstacles=0,
            distance_reduction_scale=0.0,
            collision_penalty=0.0,
            collision_fail_penalty=0.0,
            timestep_penalty=0.0,
            target_reached_bonus=0.0,
            timeout_penalty=0.0,
            terminate_on_collision=True,
        )
        obs = env.reset(scenario=sc)
        info: Dict[str, Any] = {}
        steps = 0
        while True:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                env_action, _, _ = model.sample_action(obs_t, deterministic=True)
                a = env_action.squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(a)
            steps += 1
            if terminated or truncated:
                break
            if steps > EPISODE_STEP_HARD_CAP:
                info = dict(info)
                info["stuck_failed"] = True
                break
        reached = bool(info.get("reached_target", False))
        sim_time = float(info.get("simulated_time", 0.0))
        all_times.append(sim_time)
        if reached:
            success += 1
            success_times.append(sim_time)
        else:
            if bool(info.get("collision_failed", False)):
                coll_fails += 1
            elif bool(info.get("stuck_failed", False)):
                stuck_fails += 1

    model.train()
    return {
        "episodes":            n,
        "success_rate":        (success / n) if n else 0.0,
        "avg_time_success":    float(np.mean(success_times)) if success_times else float("nan"),
        "min_time_success":    float(min(success_times)) if success_times else float("nan"),
        "avg_time_all":        float(np.mean(all_times)) if all_times else 0.0,
        "collision_fail_rate": coll_fails / n if n else 0.0,
        "stuck_fail_rate":     stuck_fails / n if n else 0.0,
    }


def benchmark_score(metrics: Dict[str, float]) -> float:
    """Composite eval score: success dominates, time is a tiebreaker."""
    sr = metrics["success_rate"]
    avg_t = metrics["avg_time_success"]
    if not math.isfinite(avg_t):
        return sr
    # 1.0 success rate is worth 1.0; subtract a tiny time penalty.
    return sr - 0.001 * avg_t


# --- Save trained policy back into IL JSON schema -------------------------

def save_actor_as_il(model: ActorCritic, il_meta: Dict[str, Any],
                     output_path: Path, ppo_meta: Dict[str, Any]) -> None:
    base = il_meta["raw"]
    linears = model.actor.linear_layers()
    weights_out = []
    biases_out = []
    for layer in linears:
        # IL uses (in, out); torch (out, in).
        w = layer.weight.detach().cpu().numpy().T.astype(np.float64)
        b = layer.bias.detach().cpu().numpy().reshape(1, -1).astype(np.float64)
        weights_out.append(w.tolist())
        biases_out.append(b.tolist())

    blob = dict(base)  # preserve all original keys (input_layout, deployment meta, etc.)
    blob["saved_at"] = datetime.now().isoformat(timespec="seconds")
    blob["weights"] = weights_out
    blob["biases"]  = biases_out
    blob["ppo_meta"] = ppo_meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


# --- PPO trainer (runs in a worker thread; posts updates via queue) -------

@dataclass
class TrainerConfig:
    warm_start_path: Path
    benchmark_dir:   Path
    output_dir:      Path
    output_stem:     str
    hparams:         Dict[str, Any]
    n_obstacles_min: int          # inclusive; (0,0) defers to sim default 3-10
    n_obstacles_max: int          # inclusive
    base_seed:       int


@dataclass
class StatusUpdate:
    iteration:        int = 0
    total_iters:      int = 0
    mean_ep_reward:   float = float("nan")
    mean_ep_len:      int = 0
    success_rate_train: float = float("nan")
    policy_loss:      float = float("nan")
    value_loss:       float = float("nan")
    entropy:          float = float("nan")
    approx_kl:        float = float("nan")
    eval_score:       float = float("nan")
    eval_metrics:     Dict[str, float] = field(default_factory=dict)
    baseline_score:   float = float("nan")
    best_score:       float = float("nan")
    log_line:         str = ""
    done:             bool = False
    saved_path:       Optional[str] = None


class Trainer:
    def __init__(self, cfg: TrainerConfig, status_q: "queue.Queue[StatusUpdate]",
                 stop_evt: threading.Event) -> None:
        self.cfg = cfg
        self.status_q = status_q
        self.stop_evt = stop_evt

    def _emit(self, **kwargs) -> None:
        self.status_q.put(StatusUpdate(**kwargs))

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:
            tb = traceback.format_exc()
            self._emit(log_line=f"FATAL: {exc}\n{tb}", done=True)

    def _run_inner(self) -> None:
        cfg = self.cfg
        hp = cfg.hparams
        requested = str(hp["device"]).lower()
        cuda_ok = bool(torch.cuda.is_available())
        if requested.startswith("cuda"):
            if not cuda_ok:
                self._emit(log_line=(
                    "ERROR: device='cuda' requested but torch.cuda.is_available() is False. "
                    f"torch={torch.__version__} built with cuda={torch.version.cuda}. "
                    "Reinstall a CUDA-enabled PyTorch wheel (e.g. "
                    "pip install --index-url https://download.pytorch.org/whl/cu121 torch)."
                ), done=True)
                return
            device = torch.device("cuda")
            gpu_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            cap = torch.cuda.get_device_capability(gpu_idx)
            self._emit(log_line=(
                f"Device: cuda:{gpu_idx}  ({gpu_name}, compute cap {cap[0]}.{cap[1]})  "
                f"torch={torch.__version__} cuda_build={torch.version.cuda}"
            ))
            self._emit(log_line=(
                "  Note: env workers always run on CPU -- they are the bottleneck. "
                "GPU only handles the small actor/critic forward+backward."
            ))
        else:
            device = torch.device("cpu")
            self._emit(log_line=(
                f"Device: cpu  (cuda_available={cuda_ok})  torch={torch.__version__}"
            ))

        self._emit(log_line=f"Loading IL model: {cfg.warm_start_path}")
        il = load_il_blob(cfg.warm_start_path)
        self._emit(log_line=(
            f"  mode={il['mode']} input_dim={il['input_dim']} "
            f"output_dim={il['output_dim']} hidden={il['hidden_layers']}x{il['hidden_width']} "
            f"act={il['activation']} memory_steps={il['memory_steps']}"
        ))

        scenarios = load_benchmark_maps(cfg.benchmark_dir)
        self._emit(log_line=f"Loaded {len(scenarios)} benchmark maps from {cfg.benchmark_dir}")

        model = ActorCritic(
            il, log_std_init=hp["log_std_init"],
            log_std_min=hp["log_std_min"], log_std_max=hp["log_std_max"],
        ).to(device)

        # Baseline benchmark eval.
        self._emit(log_line="Running baseline benchmark eval ...")
        baseline = evaluate_policy_on_maps(model, scenarios, il, device)
        baseline_score = benchmark_score(baseline)
        self._emit(log_line=(
            f"BASELINE: success={baseline['success_rate']*100:.1f}% "
            f"avg_time_succ={baseline['avg_time_success']:.2f}s "
            f"score={baseline_score:.4f}"
        ), baseline_score=baseline_score, best_score=baseline_score,
            eval_score=baseline_score, eval_metrics=baseline)

        if self.stop_evt.is_set():
            self._emit(log_line="Stop requested before training started.", done=True)
            return

        # Spin up workers.
        self._emit(log_line=f"Spawning {hp['n_envs']} env workers ...")
        venv = VectorEnv(
            n_envs=hp["n_envs"],
            control_mode=il["mode"],
            history_window=il["memory_steps"],
            base_seed=cfg.base_seed,
            n_obstacles_min=cfg.n_obstacles_min,
            n_obstacles_max=cfg.n_obstacles_max,
        )

        try:
            self._train_loop(model, venv, il, scenarios, device,
                             baseline_score, baseline)
        finally:
            venv.close()

    def _train_loop(self, model: ActorCritic, venv: VectorEnv,
                    il: Dict[str, Any], scenarios: List[Dict[str, Any]],
                    device: torch.device,
                    baseline_score: float, baseline_metrics: Dict[str, float]) -> None:
        cfg = self.cfg
        hp = cfg.hparams
        T = int(hp["n_steps"])
        N = int(hp["n_envs"])
        obs_dim = il["input_dim"]
        act_dim = il["output_dim"]

        actor_params = list(model.actor.parameters()) + [model.log_std]
        critic_params = list(model.critic.parameters())
        opt_actor = optim.Adam(actor_params, lr=hp["lr_actor"])
        opt_critic = optim.Adam(critic_params, lr=hp["lr_critic"])

        # Episode reward tracking for the live graph.
        ep_returns = np.zeros(N, dtype=np.float64)
        ep_steps_running = np.zeros(N, dtype=np.int64)
        recent_ep_rewards: List[float] = []
        recent_ep_lens: List[int] = []
        recent_successes: List[int] = []
        EP_HISTORY = 200

        best_score = baseline_score
        best_save_path: Optional[Path] = None

        for it in range(1, int(hp["total_iters"]) + 1):
            if self.stop_evt.is_set():
                self._emit(log_line=f"Stop requested at iter {it}.", done=False)
                break

            # --- Collect rollout ---
            obs_buf = np.zeros((T, N, obs_dim), dtype=np.float32)
            act_buf = np.zeros((T, N, act_dim), dtype=np.float32)
            logp_buf = np.zeros((T, N), dtype=np.float32)
            val_buf = np.zeros((T, N), dtype=np.float32)
            rew_buf = np.zeros((T, N), dtype=np.float32)
            done_buf = np.zeros((T, N), dtype=np.bool_)

            obs_np = venv.current_obs()
            for t in range(T):
                obs_t = torch.from_numpy(obs_np).to(device)
                with torch.no_grad():
                    env_action_t, model_action_t, log_prob_t = model.sample_action(obs_t)
                    value_t = model.value(obs_t)
                env_action = env_action_t.cpu().numpy()
                model_action = model_action_t.cpu().numpy()

                obs_buf[t] = obs_np
                act_buf[t] = model_action
                logp_buf[t] = log_prob_t.cpu().numpy()
                val_buf[t] = value_t.cpu().numpy()

                step_out = venv.step(env_action)
                rew_buf[t] = step_out["rewards"]
                done_buf[t] = step_out["dones"]

                ep_returns += step_out["rewards"]
                ep_steps_running += 1
                for i in range(N):
                    if step_out["dones"][i]:
                        recent_ep_rewards.append(float(ep_returns[i]))
                        recent_ep_lens.append(int(ep_steps_running[i]))
                        recent_successes.append(1 if step_out["reached"][i] else 0)
                        ep_returns[i] = 0.0
                        ep_steps_running[i] = 0
                        if len(recent_ep_rewards) > EP_HISTORY:
                            recent_ep_rewards = recent_ep_rewards[-EP_HISTORY:]
                            recent_ep_lens = recent_ep_lens[-EP_HISTORY:]
                            recent_successes = recent_successes[-EP_HISTORY:]
                obs_np = venv.current_obs()

                if self.stop_evt.is_set():
                    break

            # Bootstrap value at the end of the rollout.
            with torch.no_grad():
                last_val = model.value(torch.from_numpy(obs_np).to(device)).cpu().numpy()

            roll = Rollout(
                obs=obs_buf, actions=act_buf, log_probs=logp_buf,
                values=val_buf, rewards=rew_buf, dones=done_buf,
                next_value=last_val,
            )
            compute_gae(roll, gamma=hp["gamma"], lam=hp["gae_lambda"])

            # --- PPO update ---
            B = T * N
            obs_flat = roll.obs.reshape(B, obs_dim)
            act_flat = roll.actions.reshape(B, act_dim)
            old_logp_flat = roll.log_probs.reshape(B)
            old_val_flat = roll.values.reshape(B)
            adv_flat = roll.advantages.reshape(B)
            ret_flat = roll.returns.reshape(B)

            adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            obs_T = torch.from_numpy(obs_flat).to(device)
            act_T = torch.from_numpy(act_flat).to(device)
            old_logp_T = torch.from_numpy(old_logp_flat).to(device)
            old_val_T = torch.from_numpy(old_val_flat).to(device)
            adv_T = torch.from_numpy(adv_norm).to(device)
            ret_T = torch.from_numpy(ret_flat).to(device)

            mb = int(hp["minibatch_size"])
            clip_range = float(hp["clip_range"])
            ent_coef = float(hp["ent_coef"])
            vf_coef = float(hp["vf_coef"])
            max_grad_norm = float(hp["max_grad_norm"])
            kl_target = float(hp["kl_target"])
            kl_stop_factor = float(hp["kl_stop_factor"])

            policy_losses: List[float] = []
            value_losses: List[float] = []
            entropies: List[float] = []
            kls: List[float] = []
            stopped_early = False

            for _epoch in range(int(hp["epochs"])):
                idx = np.arange(B)
                np.random.shuffle(idx)
                epoch_kls: List[float] = []
                for start in range(0, B, mb):
                    end = min(start + mb, B)
                    mbi = idx[start:end]
                    if mbi.size == 0:
                        continue
                    mbi_T = torch.from_numpy(mbi.astype(np.int64)).to(device)
                    obs_mb = obs_T.index_select(0, mbi_T)
                    act_mb = act_T.index_select(0, mbi_T)
                    old_logp_mb = old_logp_T.index_select(0, mbi_T)
                    old_val_mb = old_val_T.index_select(0, mbi_T)
                    adv_mb = adv_T.index_select(0, mbi_T)
                    ret_mb = ret_T.index_select(0, mbi_T)

                    new_logp, entropy, value_pred = model.evaluate_actions(obs_mb, act_mb)
                    ratio = (new_logp - old_logp_mb).exp()
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss with clipping.
                    v_clipped = old_val_mb + (value_pred - old_val_mb).clamp(
                        -clip_range, clip_range)
                    vl1 = (value_pred - ret_mb) ** 2
                    vl2 = (v_clipped - ret_mb) ** 2
                    value_loss = 0.5 * torch.max(vl1, vl2).mean()

                    entropy_mean = entropy.mean()
                    loss = (policy_loss
                            + vf_coef * value_loss
                            - ent_coef * entropy_mean)

                    opt_actor.zero_grad(set_to_none=True)
                    opt_critic.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
                    opt_actor.step()
                    opt_critic.step()

                    with torch.no_grad():
                        approx_kl = (old_logp_mb - new_logp).mean().item()
                    epoch_kls.append(approx_kl)
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_mean.item()))

                if epoch_kls:
                    mean_kl = float(np.mean(epoch_kls))
                    kls.append(mean_kl)
                    if mean_kl > kl_target * kl_stop_factor:
                        stopped_early = True
                        break

            mean_ep_r = float(np.mean(recent_ep_rewards)) if recent_ep_rewards else float("nan")
            mean_ep_l = int(np.mean(recent_ep_lens)) if recent_ep_lens else 0
            train_succ = float(np.mean(recent_successes)) if recent_successes else float("nan")

            iter_log = (
                f"iter {it}/{hp['total_iters']}  "
                f"reward={mean_ep_r:8.2f}  ep_len={mean_ep_l:5d}  "
                f"train_succ={train_succ*100:5.1f}%  "
                f"pi_loss={np.mean(policy_losses) if policy_losses else float('nan'):+.4f}  "
                f"v_loss={np.mean(value_losses) if value_losses else float('nan'):.4f}  "
                f"entropy={np.mean(entropies) if entropies else float('nan'):+.4f}  "
                f"kl={np.mean(kls) if kls else float('nan'):.5f}"
                f"{'  [kl-stop]' if stopped_early else ''}"
            )
            self._emit(
                iteration=it, total_iters=int(hp["total_iters"]),
                mean_ep_reward=mean_ep_r, mean_ep_len=mean_ep_l,
                success_rate_train=train_succ,
                policy_loss=float(np.mean(policy_losses)) if policy_losses else float("nan"),
                value_loss=float(np.mean(value_losses)) if value_losses else float("nan"),
                entropy=float(np.mean(entropies)) if entropies else float("nan"),
                approx_kl=float(np.mean(kls)) if kls else float("nan"),
                baseline_score=baseline_score, best_score=best_score,
                log_line=iter_log,
            )

            # --- Periodic benchmark eval & gated save ---
            if it % int(hp["eval_every"]) == 0 or it == int(hp["total_iters"]):
                eval_metrics = evaluate_policy_on_maps(model, scenarios, il, device)
                score = benchmark_score(eval_metrics)
                eval_log = (
                    f"  eval@{it}: success={eval_metrics['success_rate']*100:.1f}%  "
                    f"avg_time_succ={eval_metrics['avg_time_success']:.2f}s  "
                    f"score={score:.4f}  (baseline={baseline_score:.4f}  best={best_score:.4f})"
                )
                saved_path: Optional[str] = None
                if score > best_score:
                    best_score = score
                    save_path = self._next_output_path()
                    save_actor_as_il(model, il, save_path, ppo_meta={
                        "trained_at":           datetime.now().isoformat(timespec="seconds"),
                        "warm_start_path":      str(cfg.warm_start_path),
                        "benchmark_dir":        str(cfg.benchmark_dir),
                        "n_benchmark_maps":     len(scenarios),
                        "iteration":            it,
                        "baseline_score":       baseline_score,
                        "score":                score,
                        "baseline_metrics":     baseline_metrics,
                        "metrics":              eval_metrics,
                        "hparams":              {k: v for k, v in hp.items() if k != "device"},
                        "reward_params":        REWARD_PARAMS,
                        "success_time_bonus":   {"amp": SUCCESS_TIME_BONUS_AMP,
                                                 "tau": SUCCESS_TIME_BONUS_TAU},
                    })
                    best_save_path = save_path
                    saved_path = str(save_path)
                    eval_log += f"  -> saved {save_path.name}"
                self._emit(
                    iteration=it, total_iters=int(hp["total_iters"]),
                    mean_ep_reward=mean_ep_r, mean_ep_len=mean_ep_l,
                    success_rate_train=train_succ,
                    eval_score=score, eval_metrics=eval_metrics,
                    baseline_score=baseline_score, best_score=best_score,
                    log_line=eval_log, saved_path=saved_path,
                )

        # End of run.
        if best_save_path is None:
            self._emit(log_line=(
                f"DONE: no improvement over baseline ({baseline_score:.4f}). "
                "IL model retained; nothing saved."
            ), done=True)
        else:
            self._emit(log_line=(
                f"DONE: best score {best_score:.4f} (baseline {baseline_score:.4f}). "
                f"Saved: {best_save_path.name}"
            ), done=True, saved_path=str(best_save_path))

    def _next_output_path(self) -> Path:
        out_dir = self.cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.cfg.output_stem
        i = 1
        while True:
            p = out_dir / f"{stem}_{i:03d}.json"
            if not p.exists():
                return p
            i += 1


# --- Tk GUI ---------------------------------------------------------------

class TrainingGui:
    POLL_MS = 200

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PPO Fine-tuning")
        self.root.geometry("1100x780")

        repo_root = Path(__file__).parent
        self.warm_start_var = tk.StringVar(value=str(repo_root / "trained_models"))
        self.benchmark_dir_var = tk.StringVar(value=str(repo_root / "benchmark"))
        self.output_dir_var = tk.StringVar(value=str(repo_root / "trained_models"))
        self.output_stem_var = tk.StringVar(value="model_ppo")

        self.n_envs_var      = tk.IntVar(value=DEFAULT_HPARAMS["n_envs"])
        self.n_steps_var     = tk.IntVar(value=DEFAULT_HPARAMS["n_steps"])
        self.total_iters_var = tk.IntVar(value=DEFAULT_HPARAMS["total_iters"])
        self.lr_actor_var    = tk.DoubleVar(value=DEFAULT_HPARAMS["lr_actor"])
        self.lr_critic_var   = tk.DoubleVar(value=DEFAULT_HPARAMS["lr_critic"])
        self.clip_range_var  = tk.DoubleVar(value=DEFAULT_HPARAMS["clip_range"])
        self.epochs_var      = tk.IntVar(value=DEFAULT_HPARAMS["epochs"])
        self.minibatch_var   = tk.IntVar(value=DEFAULT_HPARAMS["minibatch_size"])
        self.eval_every_var  = tk.IntVar(value=DEFAULT_HPARAMS["eval_every"])
        self.n_obstacles_min_var = tk.IntVar(value=3)
        self.n_obstacles_max_var = tk.IntVar(value=12)
        self.base_seed_var   = tk.IntVar(value=20260416)
        self.device_var      = tk.StringVar(value="cpu")

        self.status_var = tk.StringVar(value="Ready")
        self.iter_var = tk.StringVar(value="-")
        self.reward_var = tk.StringVar(value="-")
        self.success_var = tk.StringVar(value="-")
        self.baseline_var = tk.StringVar(value="-")
        self.best_var = tk.StringVar(value="-")

        self.status_q: "queue.Queue[StatusUpdate]" = queue.Queue()
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # Plot data.
        self.iters: List[int] = []
        self.rewards: List[float] = []
        self.rewards_ema: List[float] = []
        self.train_succ: List[float] = []
        self.eval_iters: List[int] = []
        self.eval_scores: List[float] = []
        self.eval_success_rates: List[float] = []
        self.baseline_score: Optional[float] = None
        self.baseline_success_rate: Optional[float] = None
        self.eval_time_mean: List[float] = []
        self.eval_time_best: List[float] = []
        self.baseline_time_mean: Optional[float] = None
        self.baseline_time_best: Optional[float] = None
        self.ema_alpha = 2.0 / (10.0 + 1.0)  # span ~= 10

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.POLL_MS, self._poll_status)

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # --- Top: file paths ---
        top = ttk.LabelFrame(self.root, text="Inputs / Output")
        top.pack(fill="x", padx=10, pady=(10, 4))

        def _row(parent, r, label, var, browse_cmd):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(parent, textvariable=var, width=70).grid(row=r, column=1, sticky="ew", **pad)
            ttk.Button(parent, text="Browse", command=browse_cmd).grid(row=r, column=2, **pad)

        _row(top, 0, "Warm-start IL model (.json)", self.warm_start_var, self._pick_warm_start)
        _row(top, 1, "Benchmark folder",            self.benchmark_dir_var, self._pick_benchmark)
        _row(top, 2, "Output folder",               self.output_dir_var, self._pick_output_dir)

        ttk.Label(top, text="Output filename stem").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.output_stem_var, width=30).grid(
            row=3, column=1, sticky="w", **pad)
        top.columnconfigure(1, weight=1)

        # --- Middle: hyperparameters ---
        hp = ttk.LabelFrame(self.root, text="Hyperparameters")
        hp.pack(fill="x", padx=10, pady=4)

        def _spin(parent, r, c, label, var, frm, to, inc=1):
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="e", **pad)
            sb = ttk.Spinbox(parent, from_=frm, to=to, increment=inc,
                             textvariable=var, width=10)
            sb.grid(row=r, column=c + 1, sticky="w", **pad)

        _spin(hp, 0, 0, "Parallel envs",     self.n_envs_var,      1, 64)
        _spin(hp, 0, 2, "Steps per env",     self.n_steps_var,     64, 16384, 64)
        _spin(hp, 0, 4, "Total iterations",  self.total_iters_var, 1, 10000)
        _spin(hp, 0, 6, "Eval every (iters)", self.eval_every_var, 1, 100)

        ttk.Label(hp, text="LR actor").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(hp, textvariable=self.lr_actor_var, width=10).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(hp, text="LR critic").grid(row=1, column=2, sticky="e", **pad)
        ttk.Entry(hp, textvariable=self.lr_critic_var, width=10).grid(row=1, column=3, sticky="w", **pad)
        ttk.Label(hp, text="Clip range").grid(row=1, column=4, sticky="e", **pad)
        ttk.Entry(hp, textvariable=self.clip_range_var, width=10).grid(row=1, column=5, sticky="w", **pad)
        _spin(hp, 1, 6, "Epochs",            self.epochs_var,      1, 50)

        _spin(hp, 2, 0, "Minibatch size",    self.minibatch_var,   16, 8192, 16)
        _spin(hp, 2, 2, "Obstacles min",     self.n_obstacles_min_var, 0, 200)
        _spin(hp, 2, 4, "Obstacles max",     self.n_obstacles_max_var, 0, 200)
        ttk.Label(hp, text="Base seed").grid(row=2, column=6, sticky="e", **pad)
        ttk.Entry(hp, textvariable=self.base_seed_var, width=10).grid(row=2, column=7, sticky="w", **pad)
        ttk.Label(hp, text="Device").grid(row=3, column=0, sticky="e", **pad)
        ttk.Combobox(hp, textvariable=self.device_var,
                     values=["cpu", "cuda"], width=8, state="readonly").grid(
            row=3, column=1, sticky="w", **pad)
        ttk.Button(hp, text="Check CUDA...",
                   command=self._show_cuda_info).grid(
            row=3, column=2, sticky="w", **pad)

        # --- Controls + status row ---
        ctl = ttk.Frame(self.root)
        ctl.pack(fill="x", padx=10, pady=(4, 6))
        self.start_btn = ttk.Button(ctl, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left")
        ttk.Label(ctl, textvariable=self.status_var,
                  foreground="#444").pack(side="left", padx=12)

        stats = ttk.Frame(self.root)
        stats.pack(fill="x", padx=10)
        for col, (label, var) in enumerate([
            ("iter:", self.iter_var),
            ("mean reward:", self.reward_var),
            ("train succ:", self.success_var),
            ("baseline:", self.baseline_var),
            ("best score:", self.best_var),
        ]):
            ttk.Label(stats, text=label).grid(row=0, column=2 * col, sticky="e", padx=(8, 2))
            ttk.Label(stats, textvariable=var, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=2 * col + 1, sticky="w", padx=(0, 8))
        ttk.Button(stats, text="Explain metrics...",
                   command=self._show_metrics_help).grid(
            row=0, column=2 * 5, sticky="e", padx=(16, 0))

        # --- Plot ---
        plot_frame = ttk.LabelFrame(self.root, text="Training progress")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(6, 4))
        self.fig = Figure(figsize=(8, 6.5), dpi=100)
        gs = GridSpec(3, 1, figure=self.fig, hspace=0.55)
        self.ax = self.fig.add_subplot(gs[0:2, 0])
        self.ax.tick_params(labelbottom=False)   # x labels live on the bottom plot
        self.ax.set_ylabel("mean episode reward (last 200 episodes)", color="#1f77b4")
        self.ax.tick_params(axis="y", labelcolor="#1f77b4")
        self.ax.grid(True, alpha=0.3)
        self.line_reward, = self.ax.plot(
            [], [], color="#1f77b4", alpha=0.35, linewidth=1,
            label="mean episode reward")
        self.line_reward_ema, = self.ax.plot(
            [], [], color="#1f77b4", linewidth=2,
            label="mean episode reward (EMA)")

        # Right axis: rates in [0, 1].
        self.ax_rate = self.ax.twinx()
        self.ax_rate.set_ylabel("success rate / benchmark score", color="#2ca02c")
        self.ax_rate.tick_params(axis="y", labelcolor="#2ca02c")
        self.ax_rate.set_ylim(0.0, 1.0)
        self.line_train_succ, = self.ax_rate.plot(
            [], [], color="#2ca02c", linewidth=1.5,
            label="train success rate")
        self.line_eval, = self.ax_rate.plot(
            [], [], color="#d62728", marker="o", linestyle="--",
            label="benchmark score")
        self.line_eval_succ, = self.ax_rate.plot(
            [], [], color="#ff7f0e", marker="s", linestyle=":",
            label="benchmark success rate")
        self.baseline_artist = None
        self.baseline_succ_artist = None

        # Combined legend across both axes.
        self._refresh_legend()

        # --- Time subplot (bottom row) ---
        self.ax_time = self.fig.add_subplot(gs[2, 0], sharex=self.ax)
        self.ax_time.set_xlabel("PPO iteration")
        self.ax_time.set_ylabel("nav time — successful runs (s)", color="#9467bd")
        self.ax_time.tick_params(axis="y", labelcolor="#9467bd")
        self.ax_time.grid(True, alpha=0.3)
        self.line_time_mean, = self.ax_time.plot(
            [], [], color="#9467bd", marker="o", linestyle="-", linewidth=1.5,
            label="mean nav time")
        self.line_time_best, = self.ax_time.plot(
            [], [], color="#e377c2", marker="v", linestyle="--", linewidth=1.2,
            label="best nav time")
        self.baseline_time_mean_artist = None
        self.baseline_time_best_artist = None
        self.ax_time.legend(loc="upper right", fontsize=7)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    def _refresh_legend(self) -> None:
        h1, l1 = self.ax.get_legend_handles_labels()
        h2, l2 = self.ax_rate.get_legend_handles_labels()
        self.ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    def _refresh_time_legend(self) -> None:
        self.ax_time.legend(loc="upper right", fontsize=7)

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=False, padx=10, pady=(4, 10))
        self.log_text = tk.Text(log_frame, height=10, wrap="none", font=("Consolas", 9))
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set)

    # --- File pickers ---
    def _pick_warm_start(self) -> None:
        start = self.warm_start_var.get()
        path = filedialog.askopenfilename(
            title="Pick warm-start IL model JSON",
            initialdir=start if Path(start).is_dir() else str(Path(start).parent),
            filetypes=[("IL model", "*.json"), ("All", "*.*")],
        )
        if path:
            self.warm_start_var.set(path)

    def _pick_benchmark(self) -> None:
        path = filedialog.askdirectory(
            title="Pick benchmark folder (containing map_*.json)",
            initialdir=self.benchmark_dir_var.get(),
        )
        if path:
            self.benchmark_dir_var.set(path)

    def _pick_output_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Pick output folder for trained models",
            initialdir=self.output_dir_var.get(),
        )
        if path:
            self.output_dir_var.set(path)

    # --- Start/Stop ---
    def _on_start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        warm = Path(self.warm_start_var.get())
        if not warm.is_file():
            messagebox.showerror("Bad input", f"Warm-start model not found:\n{warm}")
            return
        bench = Path(self.benchmark_dir_var.get())
        if not bench.is_dir() or not list(bench.glob("map_*.json")):
            messagebox.showerror(
                "Bad input",
                f"Benchmark folder must contain map_*.json files:\n{bench}\n\n"
                "Generate maps via benchmark_gui.py first.")
            return

        try:
            hp = dict(DEFAULT_HPARAMS)
            hp.update({
                "n_envs":         int(self.n_envs_var.get()),
                "n_steps":        int(self.n_steps_var.get()),
                "total_iters":    int(self.total_iters_var.get()),
                "lr_actor":       float(self.lr_actor_var.get()),
                "lr_critic":      float(self.lr_critic_var.get()),
                "clip_range":     float(self.clip_range_var.get()),
                "epochs":         int(self.epochs_var.get()),
                "minibatch_size": int(self.minibatch_var.get()),
                "eval_every":     int(self.eval_every_var.get()),
                "device":         str(self.device_var.get()),
            })
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Bad input", f"Invalid hyperparameter: {exc}")
            return

        # Reset plot state.
        self.iters.clear()
        self.rewards.clear()
        self.rewards_ema.clear()
        self.train_succ.clear()
        self.eval_iters.clear()
        self.eval_scores.clear()
        self.eval_success_rates.clear()
        self.baseline_score = None
        self.baseline_success_rate = None
        self.eval_time_mean.clear()
        self.eval_time_best.clear()
        self.baseline_time_mean = None
        self.baseline_time_best = None
        self.line_reward.set_data([], [])
        self.line_reward_ema.set_data([], [])
        self.line_train_succ.set_data([], [])
        self.line_eval.set_data([], [])
        self.line_eval_succ.set_data([], [])
        self.line_time_mean.set_data([], [])
        self.line_time_best.set_data([], [])
        for attr in ("baseline_artist", "baseline_succ_artist",
                     "baseline_time_mean_artist", "baseline_time_best_artist"):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax_rate.set_ylim(0.0, 1.0)
        self.ax_time.relim()
        self.ax_time.autoscale_view()
        self._refresh_legend()
        self._refresh_time_legend()
        self.canvas.draw_idle()
        self.log_text.delete("1.0", tk.END)

        obs_min = int(self.n_obstacles_min_var.get())
        obs_max = int(self.n_obstacles_max_var.get())
        if obs_min < 0 or obs_max < obs_min:
            messagebox.showerror("Bad input",
                                 f"Obstacles min/max invalid: min={obs_min} max={obs_max}")
            return

        cfg = TrainerConfig(
            warm_start_path=warm,
            benchmark_dir=bench,
            output_dir=Path(self.output_dir_var.get()),
            output_stem=self.output_stem_var.get().strip() or "model_ppo",
            hparams=hp,
            n_obstacles_min=obs_min,
            n_obstacles_max=obs_max,
            base_seed=int(self.base_seed_var.get()),
        )

        self.stop_evt.clear()
        trainer = Trainer(cfg, self.status_q, self.stop_evt)
        self.thread = threading.Thread(target=trainer.run, daemon=True)
        self.thread.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Training ...")

    def _on_stop(self) -> None:
        self.stop_evt.set()
        self.status_var.set("Stop requested ...")

    def _on_close(self) -> None:
        self.stop_evt.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.root.destroy()

    # --- CUDA diagnostics ---
    def _show_cuda_info(self) -> None:
        lines: List[str] = []
        lines.append(f"torch version:    {torch.__version__}")
        lines.append(f"built with CUDA:  {torch.version.cuda}")
        avail = torch.cuda.is_available()
        lines.append(f"cuda available:   {avail}")
        if avail:
            n = torch.cuda.device_count()
            lines.append(f"device count:     {n}")
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024 ** 3)
                lines.append(f"  [{i}] {name}  cap={cap[0]}.{cap[1]}  mem={mem_gb:.1f} GiB")

            # Round-trip a tensor to confirm the device actually executes ops.
            try:
                t0 = time.perf_counter()
                a = torch.randn(512, 512, device="cuda")
                b = (a @ a).sum().item()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                lines.append(f"matmul smoke-test: OK ({b:+.2f}, {dt:.1f} ms)")
            except Exception as exc:
                lines.append(f"matmul smoke-test: FAILED -- {exc}")
        else:
            lines.append("")
            lines.append("PyTorch cannot see a CUDA device. Common causes:")
            lines.append("  * PyTorch installed CPU-only. Reinstall a CUDA wheel:")
            lines.append("    pip install --index-url "
                         "https://download.pytorch.org/whl/cu121 torch")
            lines.append("  * NVIDIA driver too old for the PyTorch CUDA build.")
            lines.append("  * Laptop GPU disabled (check Windows graphics settings).")

        lines.append("")
        lines.append("Important: env workers always run on CPU. Even with CUDA")
        lines.append("fully enabled, most Task Manager CPU load comes from the")
        lines.append("parallel simulators. The actor/critic net is small, so")
        lines.append("GPU utilisation may be < 10% and that is expected.")

        messagebox.showinfo("CUDA diagnostics", "\n".join(lines))

    # --- Help popup ---
    METRICS_HELP_TEXT = (
        "STATUS BAR\n"
        "----------\n"
        "iter            Current PPO iteration / total iterations.\n"
        "                Each iteration collects (parallel envs) x (steps per env)\n"
        "                transitions, then runs (epochs) passes of mini-batch SGD.\n"
        "\n"
        "mean reward     Mean total episode reward over the last 200 completed\n"
        "                episodes (rolling window). Reward shape:\n"
        "                  +100  base for reaching the target\n"
        "                  +up to +30  inverse-time bonus, success only\n"
        "                  -50   collision (terminal)\n"
        "                  -50   stuck/timeout (terminal)\n"
        "                  +0.05 * (distance closed this step)   weak shaping\n"
        "                  -0.002 per step                       mild time pressure\n"
        "                A near-instant collision NEVER earns the time bonus, so\n"
        "                a fast crash always nets ~ -50, never anything positive.\n"
        "                Expected ceiling for a near-perfect policy: ~115-130.\n"
        "\n"
        "train succ      Fraction of those rolling-window episodes that reached the\n"
        "                target (training scenarios = freshly random, NOT benchmark).\n"
        "\n"
        "baseline        Benchmark score of the warm-start IL model, measured ONCE\n"
        "                before training. Anchors all comparisons.\n"
        "\n"
        "best score      Highest benchmark score observed so far during this run.\n"
        "                A new model JSON is written ONLY when this exceeds the\n"
        "                previous best. If 'best' never beats 'baseline', no file\n"
        "                is saved -- the IL model is preserved.\n"
        "\n"
        "BENCHMARK SCORE\n"
        "---------------\n"
        "Composite score on the fixed maps in the benchmark folder, evaluated\n"
        "with deterministic actions (mean of the policy, no noise):\n"
        "    score = success_rate - 0.001 * avg_time_success_seconds\n"
        "Success rate dominates; avg time is only a tiebreaker between\n"
        "policies of equal success rate. With typical runs of 30-90 s the\n"
        "time penalty is 0.03-0.09, so score is usually ~0.05-0.08 below\n"
        "success_rate -- they will look close on the plot but are distinct.\n"
        "Range typically [0, ~1].\n"
        "\n"
        "PLOT SERIES\n"
        "-----------\n"
        "TOP PLOT\n"
        "  blue (faint)       Per-iteration mean episode reward (raw, noisy).\n"
        "  blue (bold)        EMA-smoothed mean episode reward (span~10).\n"
        "  green solid        Train success rate, rolling window.\n"
        "  red dashed         Benchmark score (composite, right axis).\n"
        "  orange dotted      Benchmark success rate (right axis).\n"
        "  grey dotted        Baseline benchmark score (right axis, once).\n"
        "  green dashed (dim) Baseline benchmark success rate (right axis, once).\n"
        "Left axis: reward.   Right axis: rates / score in [0, 1].\n"
        "\n"
        "BOTTOM PLOT — Navigation time (successful benchmark runs only)\n"
        "  purple solid       Mean navigation time across successful benchmark runs.\n"
        "  pink dashed        Best (fastest) successful run time.\n"
        "  purple dotted      Baseline mean time (horizontal reference).\n"
        "  pink dotted        Baseline best time (horizontal reference).\n"
        "Only eval iterations appear on this plot (same x-axis as the top plot).\n"
        "NaN / missing points mean zero successful runs at that eval.\n"
        "\n"
        "PPO TELEMETRY (in the log pane)\n"
        "-------------------------------\n"
        "pi_loss         PPO clipped policy-gradient loss. Negative = the policy\n"
        "                update is increasing expected return on this batch.\n"
        "                Magnitudes near zero are normal once the policy converges.\n"
        "v_loss          Value-function regression loss (predicted return vs. GAE\n"
        "                target). Should trend downward and stabilise; spikes mean\n"
        "                the critic is chasing a moving target.\n"
        "entropy         Differential entropy of the action distribution. High =\n"
        "                exploratory; low = deterministic. Falls naturally as the\n"
        "                policy sharpens. Floor is set by log_std_min.\n"
        "kl              Approximate mean KL divergence between the old policy\n"
        "                (used to collect the rollout) and the updated policy.\n"
        "                If kl > 1.5 x kl_target (=0.01), the inner loop\n"
        "                early-stops to avoid destroying the warm-start. The\n"
        "                tag '[kl-stop]' indicates this happened.\n"
        "\n"
        "OBSTACLES MIN / MAX\n"
        "-------------------\n"
        "Each training episode samples its obstacle count uniformly in\n"
        "[min, max] (inclusive). Setting both to 0 defers to the simulator's\n"
        "internal default of 3-10. Increasing the range biases training toward\n"
        "harder scenarios -- useful once the policy plateaus on easy maps.\n"
        "\n"
        "DEVICE / CUDA\n"
        "-------------\n"
        "'cpu' runs the actor/critic on the CPU. 'cuda' runs them on the GPU\n"
        "when a CUDA-enabled PyTorch wheel and a supported NVIDIA GPU are\n"
        "present. Use the 'Check CUDA...' button to verify availability and\n"
        "run a matmul smoke-test on the device.\n"
        "\n"
        "Important: env workers ALWAYS run on CPU -- they are pure-Python\n"
        "simulators and are the dominant cost of a PPO iteration. The actor/\n"
        "critic net is small (a few dozen Kflops per forward), so even with\n"
        "CUDA fully enabled, Task Manager will show high CPU usage (the\n"
        "workers) and low GPU usage (tiny net). That is NOT a bug; it means\n"
        "the bottleneck is simulation, not neural-net compute. To speed things\n"
        "up further, increase 'Parallel envs' -- not GPU clock.\n"
        "\n"
        "At Start, the log prints the resolved device (including GPU name for\n"
        "cuda), so you can confirm the policy is actually on the GPU.\n"
    )

    def _show_metrics_help(self) -> None:
        # Reuse window if already open.
        existing = getattr(self, "_help_win", None)
        if existing is not None and bool(existing.winfo_exists()):
            existing.lift()
            existing.focus_force()
            return
        win = tk.Toplevel(self.root)
        win.title("Metrics & training telemetry")
        win.geometry("760x680")
        win.transient(self.root)
        self._help_win = win

        frame = ttk.Frame(win, padding=8)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", font=("Consolas", 9), padx=8, pady=8)
        text.insert("1.0", self.METRICS_HELP_TEXT)
        text.configure(state="disabled")
        text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=text.yview)
        sb.pack(side="right", fill="y")
        text.configure(yscrollcommand=sb.set)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    # --- Status polling ---
    def _poll_status(self) -> None:
        try:
            while True:
                upd = self.status_q.get_nowait()
                self._apply_update(upd)
        except queue.Empty:
            pass
        self.root.after(self.POLL_MS, self._poll_status)

    def _apply_update(self, upd: StatusUpdate) -> None:
        if upd.log_line:
            self.log_text.insert(tk.END, upd.log_line + "\n")
            self.log_text.see(tk.END)
        if upd.iteration:
            self.iter_var.set(f"{upd.iteration}/{upd.total_iters}")

        # Per-iter telemetry (skip on eval-only updates that re-emit the same iter).
        is_new_iter = bool(upd.iteration) and (not self.iters or self.iters[-1] != upd.iteration)
        if math.isfinite(upd.mean_ep_reward):
            self.reward_var.set(f"{upd.mean_ep_reward:.2f}")
            if is_new_iter:
                self.iters.append(upd.iteration)
                self.rewards.append(upd.mean_ep_reward)
                if self.rewards_ema:
                    prev = self.rewards_ema[-1]
                    self.rewards_ema.append(prev + self.ema_alpha * (upd.mean_ep_reward - prev))
                else:
                    self.rewards_ema.append(upd.mean_ep_reward)
                self.line_reward.set_data(self.iters, self.rewards)
                self.line_reward_ema.set_data(self.iters, self.rewards_ema)

        if math.isfinite(upd.success_rate_train):
            self.success_var.set(f"{upd.success_rate_train*100:.1f}%")
            # Mirror the per-iter rule: only append when iters grew this update.
            if is_new_iter:
                self.train_succ.append(upd.success_rate_train)
                self.line_train_succ.set_data(self.iters, self.train_succ)

        if math.isfinite(upd.baseline_score):
            self.baseline_var.set(f"{upd.baseline_score:.4f}")
            if self.baseline_score is None:
                self.baseline_score = upd.baseline_score
                # Baseline lives on the score axis (right), where it's meaningful.
                self.baseline_artist = self.ax_rate.axhline(
                    upd.baseline_score, color="#7f7f7f", linestyle=":",
                    linewidth=1, label=f"baseline score ({upd.baseline_score:.3f})")
                bsr = float(upd.eval_metrics.get("success_rate", float("nan"))) \
                    if upd.eval_metrics else float("nan")
                if math.isfinite(bsr):
                    self.baseline_success_rate = bsr
                    self.baseline_succ_artist = self.ax_rate.axhline(
                        bsr, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.5,
                        label=f"baseline success rate ({bsr:.1%})")
                # Baseline time lines on the time subplot.
                btm = float(upd.eval_metrics.get("avg_time_success", float("nan"))) \
                    if upd.eval_metrics else float("nan")
                btb = float(upd.eval_metrics.get("min_time_success", float("nan"))) \
                    if upd.eval_metrics else float("nan")
                if math.isfinite(btm):
                    self.baseline_time_mean = btm
                    self.baseline_time_mean_artist = self.ax_time.axhline(
                        btm, color="#9467bd", linestyle=":", linewidth=1, alpha=0.6,
                        label=f"baseline mean ({btm:.1f} s)")
                if math.isfinite(btb):
                    self.baseline_time_best = btb
                    self.baseline_time_best_artist = self.ax_time.axhline(
                        btb, color="#e377c2", linestyle=":", linewidth=1, alpha=0.6,
                        label=f"baseline best ({btb:.1f} s)")
                self._refresh_legend()
                self._refresh_time_legend()
        if math.isfinite(upd.best_score):
            self.best_var.set(f"{upd.best_score:.4f}")
        if math.isfinite(upd.eval_score) and upd.iteration:
            self.eval_iters.append(upd.iteration)
            self.eval_scores.append(upd.eval_score)
            self.line_eval.set_data(self.eval_iters, self.eval_scores)
            sr = float(upd.eval_metrics.get("success_rate", float("nan"))) \
                if upd.eval_metrics else float("nan")
            if math.isfinite(sr):
                self.eval_success_rates.append(sr)
                self.line_eval_succ.set_data(self.eval_iters, self.eval_success_rates)
            tm = float(upd.eval_metrics.get("avg_time_success", float("nan"))) \
                if upd.eval_metrics else float("nan")
            tb = float(upd.eval_metrics.get("min_time_success", float("nan"))) \
                if upd.eval_metrics else float("nan")
            if math.isfinite(tm):
                self.eval_time_mean.append(tm)
                self.line_time_mean.set_data(self.eval_iters[:len(self.eval_time_mean)],
                                             self.eval_time_mean)
            if math.isfinite(tb):
                self.eval_time_best.append(tb)
                self.line_time_best.set_data(self.eval_iters[:len(self.eval_time_best)],
                                             self.eval_time_best)

        if self.iters or self.eval_iters:
            self.ax.relim()
            self.ax.autoscale_view()
            # Right axis stays pinned to [0, 1].
            self.ax_rate.set_ylim(0.0, 1.0)
            self.ax_time.relim()
            self.ax_time.autoscale_view()
            self.canvas.draw_idle()

        if upd.done:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Done.")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = TrainingGui()
    gui.run()


if __name__ == "__main__":
    # Required for Windows multiprocessing 'spawn' start method.
    mp.freeze_support()
    main()
