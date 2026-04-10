"""Stable-Baselines3 PPO training workflow for navigation."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from nav_env import NavigationEnv

ProgressCallbackType = Callable[[int, float], None]
EpisodeCallbackType = Callable[[dict], None]


def _make_env(config: Dict[str, Any], seed: int):
    def _init():
        return NavigationEnv(config=config, seed=seed)

    return _init


class _TrainerCallback(BaseCallback):
    def __init__(
        self,
        trainer: "PPOTrainer",
        progress_callback: Optional[ProgressCallbackType],
        episode_callback: Optional[EpisodeCallbackType],
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.progress_callback = progress_callback
        self.episode_callback = episode_callback
        self.last_progress_time = time.time()

    def _on_step(self) -> bool:
        if self.trainer._stop_event.is_set():
            return False

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            if i >= len(infos):
                continue
            info = infos[i] or {}
            ep_reward = float(info.get("episode_reward", 0.0))
            self.trainer._episode_rewards.append(ep_reward)
            self.trainer.episodes_total += 1
            reached = bool(info.get("reached_target", False))
            if reached:
                self.trainer.success_total += 1

            rolling = float(np.mean(self.trainer._episode_rewards)) if self.trainer._episode_rewards else 0.0
            if rolling > self.trainer.best_mean_reward:
                self.trainer.best_mean_reward = rolling
                self.trainer.model.save(str(self.trainer.output_dir / "best_model"))

            if self.trainer.episodes_total >= self.trainer._next_checkpoint_episode:
                self.trainer._save_checkpoint(rolling)
                self.trainer._next_checkpoint_episode += self.trainer.checkpoint_interval

            if self.trainer.render_enabled and self.episode_callback:
                render_state = dict(info.get("render_state", {}))
                render_state["episode"] = self.trainer.episodes_total
                render_state["total_reward"] = ep_reward
                render_state["reached_target"] = reached
                self.episode_callback(render_state)

        now = time.time()
        if now - self.last_progress_time >= 2.0 and self.progress_callback:
            mean_reward = float(np.mean(self.trainer._episode_rewards)) if self.trainer._episode_rewards else 0.0
            success_rate = (
                self.trainer.success_total / self.trainer.episodes_total
                if self.trainer.episodes_total > 0 else 0.0
            )
            self.progress_callback(self.trainer.episodes_total, mean_reward, success_rate)
            self.last_progress_time = now

        return True


class PPOTrainer:
    """Threaded PPO trainer using SubprocVecEnv parallel environments."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.output_dir = Path(self.config.get("output_dir", "trained_models/ppo_runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = float(self.config.get("learning_rate", 3e-4))
        self.n_steps = int(self.config.get("n_steps", 2048))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.n_epochs = int(self.config.get("n_epochs", 10))
        self.gamma = float(self.config.get("gamma", 0.99))
        self.gae_lambda = float(self.config.get("gae_lambda", 0.95))
        self.clip_range = float(self.config.get("clip_range", 0.2))
        self.total_timesteps = int(self.config.get("total_timesteps", 1_000_000))
        self.n_envs = int(self.config.get("parallel_envs", 8))
        self.base_seed = int(self.config.get("seed", 12345))

        self.checkpoint_interval = max(1, int(self.config.get("checkpoint_interval", 100)))
        self.render_enabled = bool(self.config.get("render", False))
        self.warm_start_model_path = self.config.get("warm_start_model_path")

        self.model: Optional[PPO] = None
        self.vec_env: Optional[SubprocVecEnv] = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        self.start_wall_time = 0.0
        self.episodes_total = 0
        self.success_total = 0
        self.best_mean_reward = float("-inf")
        self._episode_rewards: deque[float] = deque(maxlen=50)
        self._next_checkpoint_episode = self.checkpoint_interval

    def is_running(self) -> bool:
        return self._running

    def _save_checkpoint(self, mean_reward: float) -> None:
        if self.model is None:
            return

        stem = f"checkpoint_{self.episodes_total:05d}"
        model_path = self.output_dir / f"{stem}.zip"
        metadata_path = self.output_dir / f"{stem}_metadata.json"

        self.model.save(str(model_path.with_suffix("")))
        metadata = {
            "episode": int(self.episodes_total),
            "mean_reward": float(mean_reward),
            "wall_time_seconds": float(time.time() - self.start_wall_time),
            "hyperparameters": self.config,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _build_model(self, env: SubprocVecEnv) -> PPO:
        policy_kwargs = dict(net_arch=[128, 128], activation_fn=torch.nn.ReLU)
        warm_path = self.warm_start_model_path
        if warm_path:
            # Auto-convert an IL JSON warm-start to a PPO ZIP before loading.
            if str(warm_path).lower().endswith(".json"):
                from il_to_ppo import convert_il_to_ppo
                tmp_zip = str(self.output_dir / "_il_warmstart_converted")
                convert_il_to_ppo(warm_path, tmp_zip + ".zip")
                warm_path = tmp_zip + ".zip"

            # Do NOT pass policy_kwargs here — let SB3 read the saved architecture
            # from the ZIP.  Overriding policy_kwargs forces a new net_arch which
            # conflicts with the saved weights, causing silent random reinitialization.
            return PPO.load(
                warm_path,
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
            )

        return PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

    def start(
        self,
        progress_callback: Optional[ProgressCallbackType],
        episode_callback: Optional[EpisodeCallbackType],
    ) -> None:
        if self._running:
            return

        self._stop_event.clear()
        self.thread = threading.Thread(
            target=self._run_training,
            args=(progress_callback, episode_callback),
            daemon=True,
        )
        self.thread.start()

    def _run_training(
        self,
        progress_callback: Optional[ProgressCallbackType],
        episode_callback: Optional[EpisodeCallbackType],
    ) -> None:
        self._running = True
        self.start_wall_time = time.time()

        try:
            env_fns = [_make_env(self.config, self.base_seed + i) for i in range(self.n_envs)]
            self.vec_env = SubprocVecEnv(env_fns)
            self.model = self._build_model(self.vec_env)

            callback = _TrainerCallback(self, progress_callback, episode_callback)
            self.model.learn(total_timesteps=self.total_timesteps, callback=callback, progress_bar=False)
        finally:
            if self.model is not None:
                self.model.save(str(self.output_dir / "final_model"))
            if self.vec_env is not None:
                self.vec_env.close()
            self._running = False

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30.0)
            self.thread = None
