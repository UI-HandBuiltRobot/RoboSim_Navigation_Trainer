"""Benchmark map generation and headless model evaluation GUI.

This tool creates reproducible benchmark scenarios and evaluates IL JSON
models on the exact same maps for apples-to-apples comparisons.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from headless_sim import HeadlessSimulator
from sim_core import Robot

_STATE_DIM = 23


@dataclass
class LoadedModel:
    model_type: str  # "il"
    mode: str
    history_window: int
    il_blob: Optional[Dict[str, Any]] = None


def _n_action_for_mode(mode: str) -> int:
    return {"heading_drive": 2, "xy_strafe": 2, "heading_strafe": 3}.get(mode, 2)


def _history_from_dim(obs_or_input_dim: int, n_action: int) -> int:
    denom = _STATE_DIM + n_action
    numerator = int(obs_or_input_dim) + int(n_action)
    if denom <= 0 or numerator <= 0 or numerator % denom != 0:
        raise ValueError(
            f"Cannot infer history_window from dim={obs_or_input_dim} with n_action={n_action}."
        )
    return max(1, numerator // denom)


def _load_il_blob(model_path: Path) -> Dict[str, Any]:
    with open(model_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    required = ["weights", "biases", "x_scale", "y_scale"]
    missing = [k for k in required if k not in blob]
    if missing:
        raise ValueError(
            f"IL model missing keys: {missing}. Expected saturation-normalised model; "
            f"retrain with current train_mlp."
        )

    weights = [np.asarray(w, dtype=np.float64) for w in blob["weights"]]
    biases = [np.asarray(b, dtype=np.float64) for b in blob["biases"]]

    return {
        "weights": weights,
        "biases": biases,
        "x_scale": np.asarray(blob["x_scale"], dtype=np.float64),
        "y_scale": np.asarray(blob["y_scale"], dtype=np.float64),
        "input_dim": int(blob.get("input_dim", weights[0].shape[0])),
        "output_dim": int(blob.get("output_dim", weights[-1].shape[1])),
        "mode": str(blob.get("mode", "heading_drive")),
        "activation": str(blob.get("activation", "relu")),
    }


def load_model_with_inferred_history(model_path: Path, fallback_mode: str) -> LoadedModel:
    suffix = model_path.suffix.lower()
    if suffix == ".json":
        il = _load_il_blob(model_path)
        mode = il["mode"] or fallback_mode
        n_action = _n_action_for_mode(mode)
        history = _history_from_dim(il["input_dim"], n_action)
        return LoadedModel(model_type="il", mode=mode, history_window=history, il_blob=il)

    raise ValueError("Unsupported model type. Pick an IL .json model.")


def _predict_il_physical(il_blob: Dict[str, Any], obs_row: np.ndarray) -> np.ndarray:
    x_scale = np.where(np.abs(il_blob["x_scale"]) < 1e-6, 1.0, il_blob["x_scale"])
    y_scale = np.where(np.abs(il_blob["y_scale"]) < 1e-6, 1.0, il_blob["y_scale"])
    weights = il_blob["weights"]
    biases = il_blob["biases"]
    activation = str(il_blob.get("activation", "relu")).lower()

    a = obs_row / x_scale
    for i in range(len(weights) - 1):
        z = a @ weights[i] + biases[i]
        if activation == "tanh":
            a = np.tanh(z)
        elif activation == "leaky_relu":
            a = np.where(z > 0.0, z, 0.01 * z)
        else:
            a = np.maximum(0.0, z)
    out_norm = a @ weights[-1] + biases[-1]
    return (out_norm * y_scale).reshape(-1)


def _to_env_normalized_action(mode: str, model: LoadedModel, output: np.ndarray) -> np.ndarray:
    out = np.asarray(output, dtype=np.float64).reshape(-1)

    # IL output is in physical units.
    if mode == "heading_drive":
        # IL order: [drive_speed_mps, rotation_rate_dps]
        drive = float(out[0]) if out.shape[0] > 0 else 0.0
        rot = float(out[1]) if out.shape[0] > 1 else 0.0
        env = np.array([
            rot / Robot.GAMEPAD_MAX_ROTATE_RATE_DPS,
            drive / Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
        ], dtype=np.float32)
        return np.clip(env, -1.0, 1.0)

    if mode == "heading_strafe":
        rot = float(out[0]) if out.shape[0] > 0 else 0.0
        vx = float(out[1]) if out.shape[0] > 1 else 0.0
        vy = float(out[2]) if out.shape[0] > 2 else 0.0
        env = np.array([
            rot / Robot.GAMEPAD_MAX_ROTATE_RATE_DPS,
            vx / Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
            vy / Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
        ], dtype=np.float32)
        return np.clip(env, -1.0, 1.0)

    # xy_strafe
    vx = float(out[0]) if out.shape[0] > 0 else 0.0
    vy = float(out[1]) if out.shape[0] > 1 else 0.0
    env = np.array([
        vx / Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
        vy / Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS,
    ], dtype=np.float32)
    return np.clip(env, -1.0, 1.0)


def _load_benchmark_maps(folder: Path) -> List[Dict[str, Any]]:
    map_files = sorted(folder.glob("map_*.json"))
    if not map_files:
        raise FileNotFoundError(f"No map_*.json files found in {folder}")

    maps: List[Dict[str, Any]] = []
    for p in map_files:
        with open(p, "r", encoding="utf-8") as f:
            maps.append(json.load(f))
    return maps


def generate_benchmark_maps(
    folder: Path,
    num_maps: int,
    min_obstacles: int,
    max_obstacles: int,
    base_seed: int,
) -> Dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)

    if min_obstacles < 0 or max_obstacles < min_obstacles:
        raise ValueError("Invalid obstacle range.")

    generated_paths: List[str] = []
    for i in range(int(num_maps)):
        seed_i = int(base_seed) + i
        rng = np.random.default_rng(seed_i)
        obs_count = int(rng.integers(min_obstacles, max_obstacles + 1))

        sim = HeadlessSimulator(
            seed=seed_i,
            # Geometry generation is independent of control mode.
            control_mode="heading_drive",
            history_window=1,
            n_obstacles=obs_count,
        )
        sim.reset()
        state = sim.get_render_state()

        payload = {
            "map_index": i,
            "seed": seed_i,
            "n_obstacles": int(len(state["obstacles"])),
            "robot_pose": state["robot_pose"],
            "target": state["target"],
            "obstacles": state["obstacles"],
        }

        map_path = folder / f"map_{i:03d}.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        generated_paths.append(map_path.name)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "num_maps": int(num_maps),
        "min_obstacles": int(min_obstacles),
        "max_obstacles": int(max_obstacles),
        "base_seed": int(base_seed),
        "files": generated_paths,
    }
    with open(folder / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def evaluate_model_on_maps(
    model: LoadedModel,
    scenarios: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    per_map: List[Dict[str, Any]] = []
    total = len(scenarios)

    for idx, scenario in enumerate(scenarios, start=1):
        mode = model.mode
        sim = HeadlessSimulator(
            seed=int(scenario.get("seed", 0)),
            control_mode=mode,
            history_window=model.history_window,
            n_obstacles=0,
        )
        obs = sim.reset(scenario=scenario)

        done = False
        info: Dict[str, Any] = {}
        steps = 0

        while not done:
            il_out = _predict_il_physical(model.il_blob, obs.astype(np.float64).reshape(1, -1))
            env_action = _to_env_normalized_action(mode, model, il_out)

            obs, _reward, terminated, truncated, info = sim.step(env_action)
            done = bool(terminated or truncated)
            steps += 1

            # Safety break in case termination logic is changed/disabled.
            if steps > 12000:
                info = dict(info)
                info["stuck_failed"] = True
                break

        reached = bool(info.get("reached_target", False))
        record = {
            "map_index": int(scenario.get("map_index", len(per_map))),
            "seed": int(scenario.get("seed", 0)),
            "success": reached,
            "simulated_time_s": float(info.get("simulated_time", 0.0)),
            "episode_steps": int(info.get("episode_steps", steps)),
            "collision_count": int(info.get("collision_count", 0)),
            "collision_failed": bool(info.get("collision_failed", False)),
            "stuck_failed": bool(info.get("stuck_failed", False)),
            "n_obstacles": int(scenario.get("n_obstacles", len(scenario.get("obstacles", [])))),
        }
        per_map.append(record)
        if progress_callback is not None:
            progress_callback(idx, total)

    n = len(per_map)
    success = sum(1 for r in per_map if r["success"])
    success_times = [r["simulated_time_s"] for r in per_map if r["success"]]

    summary = {
        "episodes": n,
        "success_count": int(success),
        "success_rate_percent": (100.0 * success / n) if n else 0.0,
        "avg_navigation_time_s_all": float(np.mean([r["simulated_time_s"] for r in per_map])) if n else 0.0,
        "avg_navigation_time_s_success_only": float(np.mean(success_times)) if success_times else 0.0,
        "avg_collisions": float(np.mean([r["collision_count"] for r in per_map])) if n else 0.0,
        "total_collisions": int(sum(r["collision_count"] for r in per_map)),
        "collision_fail_count": int(sum(1 for r in per_map if r["collision_failed"])),
        "stuck_fail_count": int(sum(1 for r in per_map if r["stuck_failed"])),
    }

    return {"summary": summary, "per_map": per_map}


class BenchmarkGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Benchmark Evaluator")
        self.root.geometry("900x700")

        self.model_path_var = tk.StringVar(value="trained_models/model_heading_002.json")
        self.benchmark_dir_var = tk.StringVar(value="benchmark")

        self.maps_count_var = tk.IntVar(value=100)
        self.min_obs_var = tk.IntVar(value=15)
        self.max_obs_var = tk.IntVar(value=35)
        self.seed_var = tk.IntVar(value=20260410)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Progress: 0/0")

        self.status_var = tk.StringVar(value="Ready")
        self._running = False

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        top = ttk.LabelFrame(self.root, text="Benchmark Setup")
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Benchmark folder").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.benchmark_dir_var, width=55).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(top, text="Browse", command=self._pick_benchmark_folder).grid(row=0, column=2, **pad)

        ttk.Label(top, text="# maps").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(top, from_=1, to=5000, textvariable=self.maps_count_var, width=8).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(top, text="Obstacles min/max").grid(row=3, column=0, sticky="w", **pad)
        row3 = ttk.Frame(top)
        row3.grid(row=3, column=1, sticky="w", **pad)
        ttk.Spinbox(row3, from_=0, to=200, textvariable=self.min_obs_var, width=8).pack(side="left")
        ttk.Label(row3, text="to").pack(side="left", padx=6)
        ttk.Spinbox(row3, from_=0, to=200, textvariable=self.max_obs_var, width=8).pack(side="left")

        ttk.Label(top, text="Base seed").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.seed_var, width=14).grid(row=4, column=1, sticky="w", **pad)

        ttk.Button(top, text="Generate Benchmark Maps", command=self._on_generate_maps).grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=10
        )

        top.columnconfigure(1, weight=1)

        eval_frame = ttk.LabelFrame(self.root, text="Model Evaluation")
        eval_frame.pack(fill="x", padx=10, pady=6)

        ttk.Label(eval_frame, text="Model file (.json)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(eval_frame, textvariable=self.model_path_var, width=55).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(eval_frame, text="Browse", command=self._pick_model).grid(row=0, column=2, **pad)

        ttk.Button(eval_frame, text="Run Evaluation", command=self._on_run_eval).grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=10
        )

        self.progress_bar = ttk.Progressbar(
            eval_frame,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 6))
        ttk.Label(eval_frame, textvariable=self.progress_text_var, anchor="w").grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8)
        )

        eval_frame.columnconfigure(1, weight=1)

        results_frame = ttk.LabelFrame(self.root, text="Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=6)

        self.results_text = tk.Text(results_frame, wrap="word", height=18)
        self.results_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.results_text.insert("1.0", "Results will appear here.\n")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=10, pady=6)

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _append_results(self, text: str) -> None:
        self.results_text.insert("end", text + "\n")
        self.results_text.see("end")
        self.root.update_idletasks()

    def _update_progress(self, current: int, total: int) -> None:
        total_safe = max(1, int(total))
        current_safe = max(0, min(int(current), total_safe))
        pct = 100.0 * current_safe / total_safe
        self.progress_var.set(pct)
        self.progress_text_var.set(f"Progress: {current_safe}/{total_safe} ({pct:.1f}%)")
        self.root.update_idletasks()

    def _pick_model(self) -> None:
        p = filedialog.askopenfilename(
            title="Select model",
            filetypes=[("IL JSON", "*.json"), ("All files", "*.*")],
        )
        if p:
            self.model_path_var.set(p)

    def _pick_benchmark_folder(self) -> None:
        p = filedialog.askdirectory(title="Select benchmark folder")
        if p:
            self.benchmark_dir_var.set(p)

    def _on_generate_maps(self) -> None:
        if self._running:
            return

        try:
            folder = Path(self.benchmark_dir_var.get()).expanduser().resolve()
            manifest = generate_benchmark_maps(
                folder=folder,
                num_maps=int(self.maps_count_var.get()),
                min_obstacles=int(self.min_obs_var.get()),
                max_obstacles=int(self.max_obs_var.get()),
                base_seed=int(self.seed_var.get()),
            )
            self._append_results("Generated benchmark maps")
            self._append_results(json.dumps(manifest, indent=2))
            self._set_status(f"Generated {manifest['num_maps']} maps in {folder}")
        except Exception as e:
            messagebox.showerror("Benchmark generation failed", str(e))

    def _on_run_eval(self) -> None:
        if self._running:
            return
        self._running = True
        self._update_progress(0, 1)
        self.progress_text_var.set("Progress: preparing benchmark...")
        self._set_status("Running evaluation...")
        thread = threading.Thread(target=self._run_eval_worker, daemon=True)
        thread.start()

    def _run_eval_worker(self) -> None:
        try:
            model_path = Path(self.model_path_var.get()).expanduser().resolve()
            bench_dir = Path(self.benchmark_dir_var.get()).expanduser().resolve()

            scenarios = _load_benchmark_maps(bench_dir)
            fallback_mode = "heading_drive"
            loaded = load_model_with_inferred_history(model_path, fallback_mode=fallback_mode)

            t0 = time.time()
            self.root.after(0, self._update_progress, 0, len(scenarios))
            report = evaluate_model_on_maps(
                loaded,
                scenarios,
                progress_callback=lambda done, total: self.root.after(0, self._update_progress, done, total),
            )
            elapsed = time.time() - t0

            summary = report["summary"]
            output = {
                "generated_at": datetime.now().isoformat(),
                "model_path": str(model_path),
                "model_type": loaded.model_type,
                "model_mode": loaded.mode,
                "inferred_history_window": loaded.history_window,
                "benchmark_dir": str(bench_dir),
                "summary": summary,
                "per_map": report["per_map"],
            }

            out_name = f"eval_{model_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path = bench_dir / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            self.root.after(0, self._append_results, "\nEvaluation complete")
            self.root.after(0, self._append_results, f"Model: {model_path.name}")
            self.root.after(0, self._append_results, f"Type: {loaded.model_type} | Mode: {loaded.mode}")
            self.root.after(0, self._append_results, f"Inferred memory length: {loaded.history_window}")
            self.root.after(0, self._append_results, f"Episodes: {summary['episodes']}")
            self.root.after(0, self._append_results, f"Success rate: {summary['success_rate_percent']:.2f}%")
            self.root.after(0, self._append_results, f"Avg nav time (all): {summary['avg_navigation_time_s_all']:.3f} s")
            self.root.after(0, self._append_results, f"Avg nav time (success): {summary['avg_navigation_time_s_success_only']:.3f} s")
            self.root.after(0, self._append_results, f"Avg collisions: {summary['avg_collisions']:.3f}")
            self.root.after(0, self._append_results, f"Total collisions: {summary['total_collisions']}")
            self.root.after(0, self._append_results, f"Collision fails: {summary['collision_fail_count']}")
            self.root.after(0, self._append_results, f"Stuck failures: {summary['stuck_fail_count']}")
            self.root.after(0, self._append_results, f"Saved report: {out_path}")
            self.root.after(0, self._update_progress, len(scenarios), len(scenarios))
            self.root.after(0, self._set_status, f"Evaluation done in {elapsed:.2f}s")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Evaluation failed", str(e)))
            self.root.after(0, self._set_status, "Evaluation failed")
        finally:
            self._running = False

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    BenchmarkGui().run()
