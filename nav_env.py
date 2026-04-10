"""Gymnasium wrapper for the headless navigation simulator."""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from headless_sim import HeadlessSimulator


class NavigationEnv(gym.Env):
    """Gymnasium-compatible navigation environment backed by HeadlessSimulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        self.config = dict(config)
        self.base_seed = int(seed)
        self._seed_counter = 0

        self.history_window = int(self.config.get("history_window", 5))
        control_mode = str(self.config.get("control_mode", "heading_drive"))
        _n_action = {"heading_drive": 2, "xy_strafe": 2, "heading_strafe": 3}.get(control_mode, 2)
        obs_dim = 12 * self.history_window + _n_action * (self.history_window - 1)
        # Bounds are wide because observations are raw (un-normalised) physical values.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_n_action,), dtype=np.float32)

        self.sim = self._make_sim(self.base_seed)
        self._episode_reward = 0.0

    def _make_sim(self, seed: int) -> HeadlessSimulator:
        return HeadlessSimulator(
            seed=seed,
            control_mode=str(self.config.get("control_mode", "heading_drive")),
            history_window=int(self.config.get("history_window", 5)),
            n_obstacles=int(self.config.get("n_obstacles", 0)),
            distance_reduction_scale=float(self.config.get("distance_reduction_scale", 1.0)),
            collision_penalty=float(self.config.get("collision_penalty", -0.1)),
            collision_fail_penalty=float(self.config.get("collision_fail_penalty", -5.0)),
            timestep_penalty=float(self.config.get("timestep_penalty", -0.01)),
            target_reached_bonus=float(self.config.get("target_reached_bonus", 10.0)),
            timeout_penalty=float(self.config.get("timeout_penalty", -1.0)),
            terminate_on_collision=bool(self.config.get("terminate_on_collision", True)),
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        if seed is not None:
            use_seed = int(seed)
        else:
            self._seed_counter += 1
            use_seed = self.base_seed + self._seed_counter

        self.sim = self._make_sim(use_seed)
        obs = self.sim.reset()
        self._episode_reward = 0.0

        info = {
            "episode_steps": 0,
            "collision_count": 0,
            "reached_target": False,
            "simulated_time": 0.0,
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.sim.step(np.asarray(action, dtype=np.float32))
        self._episode_reward += float(reward)
        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = float(self._episode_reward)
            info["render_state"] = self.sim.get_render_state()
        return obs, float(reward), bool(terminated), bool(truncated), info
