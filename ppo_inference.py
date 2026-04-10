"""PPO policy inference adapter with IL-compatible interface."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO


class PPOPolicy:
    """Load and run a Stable-Baselines3 PPO policy."""

    def __init__(self, model_path: str):
        """Load a saved SB3 PPO model."""
        self.model = PPO.load(model_path)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return deterministic 2D action in [-1, 1] for the provided observation."""
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(2)
