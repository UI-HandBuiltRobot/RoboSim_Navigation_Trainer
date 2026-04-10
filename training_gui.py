"""Tkinter GUI for PPO training workflow."""

from __future__ import annotations

import os
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import pygame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ppo_trainer import PPOTrainer
from sim_core import ROOM_HEIGHT, ROOM_WIDTH


class TrainingGUI:
    """Desktop GUI that configures and runs PPO training in the background."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("RoboSim PPO Trainer")
        self.root.geometry("1320x780")

        self.msg_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.trainer: Optional[PPOTrainer] = None

        self.start_wall_time = 0.0
        self.status_text = tk.StringVar(value="Idle")
        self.total_episodes_var = tk.StringVar(value="0")
        self.best_mean_var = tk.StringVar(value="-")
        self.eta_var = tk.StringVar(value="-")
        self.success_rate_var = tk.StringVar(value="-")

        self.reward_x: List[int] = []
        self.reward_y: List[float] = []
        self.best_mean_reward = float("-inf")

        self._render_window_open = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._poll_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        cfg_frame = ttk.LabelFrame(left, text="Training Parameters", padding=10)
        cfg_frame.pack(fill=tk.X, pady=6)

        self.learning_rate_var = tk.StringVar(value="3e-4")
        self.parallel_envs_var = tk.IntVar(value=8)
        self.total_timesteps_var = tk.StringVar(value="1000000")
        self.checkpoint_interval_var = tk.IntVar(value=100)
        self.control_mode_var = tk.StringVar(value="heading_drive")
        self.history_window_var = tk.IntVar(value=5)
        self.n_obstacles_var = tk.IntVar(value=0)
        self.output_dir_var = tk.StringVar(value=str((Path.cwd() / "trained_models" / "ppo_runs").resolve()))
        self.warm_start_var = tk.StringVar(value="")
        self.render_success_var = tk.BooleanVar(value=False)

        self.n_steps_var = tk.IntVar(value=2048)
        self.batch_size_var = tk.IntVar(value=64)
        self.n_epochs_var = tk.IntVar(value=10)
        self.gamma_var = tk.StringVar(value="0.99")
        self.gae_lambda_var = tk.StringVar(value="0.95")
        self.clip_range_var = tk.StringVar(value="0.2")

        row = 0
        self._labeled_entry(cfg_frame, "Learning rate", self.learning_rate_var, row)
        row += 1

        ttk.Label(cfg_frame, text="Parallel environments").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg_frame, from_=1, to=32, textvariable=self.parallel_envs_var, width=12).grid(row=row, column=1, sticky="w")
        row += 1

        self._labeled_entry(cfg_frame, "Total training timesteps", self.total_timesteps_var, row)
        row += 1

        ttk.Label(cfg_frame, text="Checkpoint interval (episodes)").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg_frame, from_=1, to=100000, textvariable=self.checkpoint_interval_var, width=12).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="Control mode").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        mode_frame = ttk.Frame(cfg_frame)
        mode_frame.grid(row=row, column=1, sticky="w")
        ttk.Radiobutton(mode_frame, text="Heading+Drive", variable=self.control_mode_var, value="heading_drive").pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="XY Strafe", variable=self.control_mode_var, value="xy_strafe").pack(anchor="w")
        row += 1

        ttk.Label(cfg_frame, text="History window N").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg_frame, from_=1, to=20, textvariable=self.history_window_var, width=12).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="Obstacles per episode (0=random 3-10)").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg_frame, from_=0, to=30, textvariable=self.n_obstacles_var, width=12).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="Output directory").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        out_frame = ttk.Frame(cfg_frame)
        out_frame.grid(row=row, column=1, sticky="ew")
        ttk.Entry(out_frame, textvariable=self.output_dir_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="Browse", command=self._browse_output_dir).pack(side=tk.LEFT, padx=4)
        row += 1

        ttk.Label(cfg_frame, text="Warm-start model (.zip or IL .json)").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        warm_frame = ttk.Frame(cfg_frame)
        warm_frame.grid(row=row, column=1, sticky="ew")
        ttk.Entry(warm_frame, textvariable=self.warm_start_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(warm_frame, text="Browse", command=self._browse_warm_start).pack(side=tk.LEFT, padx=4)
        row += 1

        ttk.Label(cfg_frame, text="IL .json auto-converted to PPO on Start").grid(row=row, column=1, sticky="w", padx=4)
        row += 1

        ttk.Checkbutton(
            cfg_frame,
            text="Show episode trajectories (all outcomes)",
            variable=self.render_success_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4)

        reward_frame = ttk.LabelFrame(left, text="Reward Parameters", padding=10)
        reward_frame.pack(fill=tk.X, pady=6)

        self.distance_scale_var = tk.StringVar(value="1.0")
        self.collision_penalty_var = tk.StringVar(value="-0.1")
        self.timestep_penalty_var = tk.StringVar(value="-0.01")
        self.target_bonus_var = tk.StringVar(value="10.0")
        self.timeout_penalty_var = tk.StringVar(value="-1.0")

        self._labeled_entry(reward_frame, "Distance reduction scale", self.distance_scale_var, 0)
        self._labeled_entry(reward_frame, "Collision penalty", self.collision_penalty_var, 1)
        self._labeled_entry(reward_frame, "Timestep penalty", self.timestep_penalty_var, 2)
        self._labeled_entry(reward_frame, "Target reached bonus", self.target_bonus_var, 3)
        self._labeled_entry(reward_frame, "Stuck fail penalty", self.timeout_penalty_var, 4)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=8)
        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self._start_training)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Training", command=self._stop_training, state=tk.DISABLED)
        self.open_btn = ttk.Button(btn_frame, text="Open Output Folder", command=self._open_output_folder)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn.pack(fill=tk.X, pady=2)
        self.open_btn.pack(fill=tk.X, pady=2)

        metrics_frame = ttk.LabelFrame(right, text="Live Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True)

        labels = ttk.Frame(metrics_frame)
        labels.pack(fill=tk.X)
        ttk.Label(labels, text="Total episodes:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(labels, textvariable=self.total_episodes_var).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(labels, text="Best mean reward:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(labels, textvariable=self.best_mean_var).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(labels, text="Estimated time remaining:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(labels, textvariable=self.eta_var).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(labels, text="Success rate:").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(labels, textvariable=self.success_rate_var).grid(row=3, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(labels, text="Status:").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(labels, textvariable=self.status_text).grid(row=4, column=1, sticky="w", padx=4, pady=2)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Rolling Mean Reward (window=50)")
        self.ax.set_xlabel("Total Episodes")
        self.ax.set_ylabel("Mean Reward")
        self.line, = self.ax.plot([], [], color="tab:blue")

        self.canvas = FigureCanvasTkAgg(self.fig, master=metrics_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _labeled_entry(parent: ttk.Widget, label: str, var: tk.Variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(parent, textvariable=var, width=24).grid(row=row, column=1, sticky="w", padx=4, pady=4)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(Path.cwd()))
        if path:
            self.output_dir_var.set(path)

    def _browse_warm_start(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Model files", "*.zip;*.json"),
                ("PPO ZIP", "*.zip"),
                ("IL JSON", "*.json"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.warm_start_var.set(path)

    def _build_config(self) -> Dict[str, Any]:
        output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg: Dict[str, Any] = {
            "learning_rate": float(self.learning_rate_var.get()),
            "parallel_envs": int(self.parallel_envs_var.get()),
            "total_timesteps": int(self.total_timesteps_var.get()),
            "checkpoint_interval": int(self.checkpoint_interval_var.get()),
            "control_mode": str(self.control_mode_var.get()),
            "history_window": int(self.history_window_var.get()),
            "n_obstacles": int(self.n_obstacles_var.get()),
            "output_dir": str(output_dir),
            "distance_reduction_scale": float(self.distance_scale_var.get()),
            "collision_penalty": float(self.collision_penalty_var.get()),
            "timestep_penalty": float(self.timestep_penalty_var.get()),
            "target_reached_bonus": float(self.target_bonus_var.get()),
            "timeout_penalty": float(self.timeout_penalty_var.get()),
            "n_steps": int(self.n_steps_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "n_epochs": int(self.n_epochs_var.get()),
            "gamma": float(self.gamma_var.get()),
            "gae_lambda": float(self.gae_lambda_var.get()),
            "clip_range": float(self.clip_range_var.get()),
            "render": bool(self.render_success_var.get()),
        }

        warm = self.warm_start_var.get().strip()
        if warm:
            cfg["warm_start_model_path"] = warm
        return cfg

    def _start_training(self) -> None:
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.trainer = PPOTrainer(config)
        self.reward_x.clear()
        self.reward_y.clear()
        self.best_mean_reward = float("-inf")
        self.total_episodes_var.set("0")
        self.best_mean_var.set("-")
        self.eta_var.set("-")

        self.start_wall_time = time.time()
        self.status_text.set("Training")

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        self.trainer.start(
            progress_callback=lambda episodes, mean_reward, success_rate: self.msg_queue.put(("progress", (episodes, mean_reward, success_rate))),
            episode_callback=lambda render_state: self.msg_queue.put(("episode", render_state)),
        )

    def _stop_training(self) -> None:
        if self.trainer is None:
            return
        self.status_text.set("Stopping")
        self.trainer.stop()
        self.status_text.set("Complete")
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def _open_output_folder(self) -> None:
        path = Path(self.output_dir_var.get()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def _format_eta(self, elapsed_s: float, trainer: PPOTrainer) -> str:
        if trainer.model is None or trainer.total_timesteps <= 0:
            return "-"
        done = float(getattr(trainer.model, "num_timesteps", 0))
        frac = max(1e-9, min(1.0, done / float(trainer.total_timesteps)))
        if frac >= 1.0:
            return "0s"
        remaining = elapsed_s * (1.0 - frac) / frac
        return f"{int(remaining)}s"

    def _draw_episode_render(self, render_state: Dict[str, Any]) -> None:
        pygame.init()
        pygame.font.init()
        if self._render_window_open:
            try:
                pygame.display.quit()
            except Exception:
                pass
        pygame.display.init()
        reached = bool(render_state.get("reached_target", False))
        outcome_label = "SUCCESS" if reached else "FAILED"
        ep_num = int(render_state.get("episode", 0))
        pygame.display.set_caption(f"Episode {ep_num} — {outcome_label}")
        screen = pygame.display.set_mode((900, 620))
        self._render_window_open = True

        screen.fill((20, 20, 24))
        room_rect = pygame.Rect(50, 50, 800, 520)
        pygame.draw.rect(screen, (245, 245, 245), room_rect)
        pygame.draw.rect(screen, (40, 40, 40), room_rect, 2)

        ppm = room_rect.width / ROOM_WIDTH

        def world_to_screen(x: float, y: float) -> Tuple[int, int]:
            sx = int(room_rect.left + x * ppm)
            sy = int(room_rect.bottom - y * ppm)
            return sx, sy

        for ob in render_state.get("obstacles", []):
            ox = float(ob["x"])
            oy = float(ob["y"])
            r = float(ob["radius"])
            pygame.draw.circle(screen, (120, 120, 130), world_to_screen(ox, oy), max(2, int(r * ppm)))

        tgt = render_state.get("target", {})
        if tgt:
            pygame.draw.circle(
                screen,
                (50, 180, 70),
                world_to_screen(float(tgt["x"]), float(tgt["y"])),
                max(2, int(float(tgt["radius"]) * ppm)),
            )

        traj = render_state.get("trajectory_history", [])
        if len(traj) >= 2:
            pts = [world_to_screen(float(x), float(y)) for x, y in traj]
            traj_color = (50, 200, 90) if reached else (220, 80, 60)
            pygame.draw.lines(screen, traj_color, False, pts, 2)

        font = pygame.font.SysFont("consolas", 20)
        outcome_color = (80, 230, 110) if reached else (230, 90, 70)
        ep_txt = font.render(f"Ep {ep_num}  {outcome_label}", True, outcome_color)
        rew_txt = font.render(f"Reward {float(render_state.get('total_reward', 0.0)):.3f}", True, (200, 200, 200))
        screen.blit(ep_txt, (20, 10))
        screen.blit(rew_txt, (260, 10))

        pygame.display.flip()
        pygame.event.pump()

    def _poll_queue(self) -> None:
        while True:
            try:
                tag, payload = self.msg_queue.get_nowait()
            except queue.Empty:
                break

            if tag == "progress":
                episodes, mean_reward, success_rate = payload
                self.total_episodes_var.set(str(int(episodes)))
                self.reward_x.append(int(episodes))
                self.reward_y.append(float(mean_reward))
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = float(mean_reward)
                self.best_mean_var.set(f"{self.best_mean_reward:.4f}")
                self.success_rate_var.set(f"{success_rate:.1%}")

                self.line.set_data(self.reward_x, self.reward_y)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw_idle()

                if self.trainer is not None:
                    elapsed = max(0.0, time.time() - self.start_wall_time)
                    self.eta_var.set(self._format_eta(elapsed, self.trainer))
                    if not self.trainer.is_running():
                        self.status_text.set("Complete")
                        self.start_btn.configure(state=tk.NORMAL)
                        self.stop_btn.configure(state=tk.DISABLED)

            elif tag == "episode":
                last_episode = payload  # keep only the latest; older ones are dropped

        if "last_episode" in locals() and self.render_success_var.get():
            self._draw_episode_render(last_episode)

        self.root.after(500, self._poll_queue)

    def _on_close(self) -> None:
        if self.trainer is not None and self.trainer.is_running():
            self.status_text.set("Stopping")
            self.trainer.stop()
        if self._render_window_open:
            pygame.display.quit()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = TrainingGUI()
    app.run()


if __name__ == "__main__":
    main()
