import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


STATE_INPUT_DIM = 12  # 11 whiskers + heading_to_target
MODE_HEADING = "heading_drive"
MODE_STRAFE = "xy_strafe"
MODE_HEADING_STRAFE = "heading_strafe"
EPS = 1e-6


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray


LEAKY_ALPHA = 0.01
ACTIVATIONS = ("relu", "tanh", "leaky_relu")


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        if self.tip_window is not None or not self.text:
            return

        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#fff8cc",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=3,
        )
        label.pack()

    def _hide(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class NumpyMLP:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_width: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if hidden_layers < 1 or hidden_width < 1:
            raise ValueError("hidden_layers and hidden_width must be >= 1")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")

        self.activation = activation
        rng = np.random.default_rng(seed)
        layer_sizes = [input_dim] + [hidden_width] * hidden_layers + [output_dim]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            is_hidden = i < len(layer_sizes) - 2
            if is_hidden and activation == "tanh":
                scale = np.sqrt(1.0 / fan_in)   # Xavier
            elif is_hidden:
                scale = np.sqrt(2.0 / fan_in)   # He (relu / leaky_relu)
            else:
                scale = np.sqrt(1.0 / fan_in)
            w = rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "leaky_relu":
            return np.where(x > 0.0, x, LEAKY_ALPHA * x)
        return np.maximum(0.0, x)  # relu

    def _act_grad(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            t = np.tanh(x)
            return 1.0 - t * t
        if self.activation == "leaky_relu":
            return np.where(x > 0.0, 1.0, LEAKY_ALPHA).astype(np.float64)
        return (x > 0.0).astype(np.float64)  # relu

    def _forward_train(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        preacts: List[np.ndarray] = []

        a = x
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            preacts.append(z)
            a = self._act(z)
            activations.append(a)

        z_out = a @ self.weights[-1] + self.biases[-1]
        preacts.append(z_out)
        activations.append(z_out)
        return activations, preacts

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = x
        for i in range(len(self.weights) - 1):
            a = self._act(a @ self.weights[i] + self.biases[i])
        return a @ self.weights[-1] + self.biases[-1]

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 123,
    ) -> Dict[str, List[float]]:
        rng = np.random.default_rng(seed)
        n = x_train.shape[0]
        if n == 0:
            raise ValueError("x_train is empty")

        history = {"train_loss": [], "val_loss": []}

        for _ in range(epochs):
            idx = rng.permutation(n)
            x_shuf = x_train[idx]
            y_shuf = y_train[idx]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = x_shuf[start:end]
                yb = y_shuf[start:end]
                m = xb.shape[0]

                activations, preacts = self._forward_train(xb)
                y_pred = activations[-1]

                d = (2.0 / m) * (y_pred - yb)

                grad_w = [np.zeros_like(w) for w in self.weights]
                grad_b = [np.zeros_like(b) for b in self.biases]

                for layer in reversed(range(len(self.weights))):
                    a_prev = activations[layer]
                    grad_w[layer] = a_prev.T @ d
                    grad_b[layer] = np.sum(d, axis=0, keepdims=True)

                    if layer > 0:
                        d = (d @ self.weights[layer].T) * self._act_grad(preacts[layer - 1])

                for layer in range(len(self.weights)):
                    self.weights[layer] -= learning_rate * grad_w[layer]
                    self.biases[layer] -= learning_rate * grad_b[layer]

            train_pred = self.predict(x_train)
            val_pred = self.predict(x_val)
            train_loss = float(np.mean((train_pred - y_train) ** 2))
            val_loss = float(np.mean((val_pred - y_val) ** 2))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        return history


def _action_keys_for_mode(mode: str) -> List[str]:
    if mode == MODE_HEADING:
        return ["drive_speed", "rotation_rate"]
    if mode == MODE_HEADING_STRAFE:
        return ["rotation_rate", "vx", "vy"]
    return ["vx", "vy"]


def _model_mode_name(mode: str) -> str:
    mode_map = {
        MODE_HEADING: "heading",
        MODE_STRAFE: "strafe",
        MODE_HEADING_STRAFE: "heading_strafe",
    }
    return mode_map.get(mode, str(mode))


def _infer_memory_steps(input_dim: int, mode: str) -> int:
    action_dim = len(_action_keys_for_mode(mode))
    denom = STATE_INPUT_DIM + action_dim
    numer = input_dim + action_dim
    if denom <= 0 or numer <= 0 or numer % denom != 0:
        raise ValueError(
            f"cannot infer memory_steps for mode={mode}: input_dim={input_dim}, action_dim={action_dim}"
        )
    return numer // denom


def _build_input_layout(mode: str, memory_steps: int) -> List[str]:
    layout: List[str] = []
    action_keys = _action_keys_for_mode(mode)
    oldest_offset = -(memory_steps - 1)
    for i in range(memory_steps):
        offset = oldest_offset + i
        prefix = f"t{offset:+d}"
        layout.append(f"{prefix}.whiskers")
        layout.append(f"{prefix}.heading_to_target")
        if i < memory_steps - 1:
            for action_key in action_keys:
                layout.append(f"{prefix}.action.{action_key}")
    return layout


def _zero_history_step(mode: str) -> Dict[str, object]:
    return {
        "whisker_lengths": [0.0] * 11,
        "heading_to_target": 0.0,
        "action": {k: 0.0 for k in _action_keys_for_mode(mode)},
    }


def _legacy_row_to_history(row: Dict[str, object], mode: str) -> List[Dict[str, object]]:
    action = {k: float(row.get(k, 0.0)) for k in _action_keys_for_mode(mode)}
    return [{
        "whisker_lengths": row["whisker_lengths"],
        "heading_to_target": row["heading_to_target"],
        "action": action,
    }]


def _flatten_history_features(mode: str, history: List[Dict[str, object]]) -> List[float]:
    features: List[float] = []
    for i, step in enumerate(history):
        whiskers = step["whisker_lengths"]
        heading = float(step["heading_to_target"])
        if not isinstance(whiskers, list) or len(whiskers) != 11:
            raise ValueError("whisker_lengths must be a list of length 11")

        state = [float(v) for v in whiskers] + [heading]
        features.extend(state)

        # Current timestep contributes state only; prior timesteps include action.
        if i < len(history) - 1:
            action_map = step.get("action", {})
            if not isinstance(action_map, dict):
                action_map = {}
            for k in _action_keys_for_mode(mode):
                features.append(float(action_map.get(k, 0.0)))
    return features


def parse_logs(file_paths: List[str]):
    x_by_mode: Dict[str, List[List[float]]] = {
        MODE_HEADING: [],
        MODE_STRAFE: [],
        MODE_HEADING_STRAFE: [],
    }
    y_by_mode: Dict[str, List[List[float]]] = {
        MODE_HEADING: [],
        MODE_STRAFE: [],
        MODE_HEADING_STRAFE: [],
    }
    inferred_input_dims: Dict[str, int] = {}
    # Per-mode aggregation of deployment-relevant metadata across samples.
    # Each value is a Counter[float] mapping observed value -> sample count.
    from collections import Counter
    metadata_counters: Dict[str, Dict[str, "Counter"]] = {
        m: {
            "min_turn_rate_dps": Counter(),
            "inner_deadband_dps": Counter(),
            "active_log_rate_hz": Counter(),
            "history_len": Counter(),
        }
        for m in (MODE_HEADING, MODE_STRAFE, MODE_HEADING_STRAFE)
    }

    stats = {
        "files": len(file_paths),
        "lines": 0,
        "valid": 0,
        "invalid": 0,
        "heading_samples": 0,
        "strafe_samples": 0,
        "heading_strafe_samples": 0,
    }

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stats["lines"] += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                    mode = row["mode"]
                    if mode not in (MODE_HEADING, MODE_STRAFE, MODE_HEADING_STRAFE):
                        raise ValueError("unknown mode")

                    if "history" in row and isinstance(row["history"], list):
                        history_len = int(row.get("history_len", len(row["history"])))
                        raw_history = [h for h in row["history"] if isinstance(h, dict)]
                    else:
                        history_len = int(row.get("history_len", 1))
                        raw_history = _legacy_row_to_history(row, mode)

                    history_len = max(1, history_len)
                    history = raw_history[-history_len:]
                    while len(history) < history_len:
                        history.insert(0, _zero_history_step(mode))

                    x_row = _flatten_history_features(mode, history)

                    action_map = history[-1].get("action", {}) if history else {}
                    if not isinstance(action_map, dict):
                        action_map = {}

                    if mode == MODE_HEADING:
                        y_row = [float(action_map.get("drive_speed", 0.0)), float(action_map.get("rotation_rate", 0.0))]
                        stats["heading_samples"] += 1
                    elif mode == MODE_STRAFE:
                        y_row = [float(action_map.get("vx", 0.0)), float(action_map.get("vy", 0.0))]
                        stats["strafe_samples"] += 1
                    else:
                        y_row = [
                            float(action_map.get("rotation_rate", 0.0)),
                            float(action_map.get("vx", 0.0)),
                            float(action_map.get("vy", 0.0)),
                        ]
                        stats["heading_strafe_samples"] += 1

                    input_dim = len(x_row)
                    prev_input_dim = inferred_input_dims.get(mode)
                    if prev_input_dim is None:
                        inferred_input_dims[mode] = input_dim
                    else:
                        if prev_input_dim != input_dim:
                            raise ValueError(f"inconsistent input_dim for {mode}: got {input_dim}, expected {prev_input_dim}")

                    x_by_mode[mode].append(x_row)
                    y_by_mode[mode].append(y_row)

                    meta = metadata_counters[mode]
                    meta["history_len"][int(history_len)] += 1
                    if "min_turn_rate_dps" in row:
                        meta["min_turn_rate_dps"][float(row["min_turn_rate_dps"])] += 1
                    if "inner_deadband_dps" in row:
                        meta["inner_deadband_dps"][float(row["inner_deadband_dps"])] += 1
                    if "active_log_rate_hz" in row:
                        meta["active_log_rate_hz"][float(row["active_log_rate_hz"])] += 1

                    stats["valid"] += 1
                except Exception:
                    stats["invalid"] += 1

    datasets: Dict[str, Dataset] = {}
    for mode in (MODE_HEADING, MODE_STRAFE, MODE_HEADING_STRAFE):
        if x_by_mode[mode]:
            datasets[mode] = Dataset(
                np.asarray(x_by_mode[mode], dtype=np.float64),
                np.asarray(y_by_mode[mode], dtype=np.float64),
            )

    metadata_summary: Dict[str, Dict[str, object]] = {}
    for mode, counters in metadata_counters.items():
        summary: Dict[str, object] = {}
        for field, counter in counters.items():
            if not counter:
                summary[field] = {"value": None, "distinct": []}
                continue
            # Most-common value wins; expose all observed values for transparency.
            dominant_value, _ = counter.most_common(1)[0]
            distinct = sorted(
                [{"value": v, "count": int(c)} for v, c in counter.items()],
                key=lambda d: -d["count"],
            )
            summary[field] = {"value": dominant_value, "distinct": distinct}
        metadata_summary[mode] = summary

    return datasets, stats, inferred_input_dims, metadata_summary


def split_train_val(dataset: Dataset, val_ratio: float = 0.2, seed: int = 123) -> Tuple[Dataset, Dataset]:
    n = dataset.x.shape[0]
    if n < 2:
        return dataset, dataset

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_n = max(1, int(n * val_ratio))

    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    if train_idx.size == 0:
        train_idx = val_idx

    train = Dataset(dataset.x[train_idx], dataset.y[train_idx])
    val = Dataset(dataset.x[val_idx], dataset.y[val_idx])
    return train, val


def compute_norm(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def next_index(output_dir: Path, mode_suffix: str) -> int:
    existing = sorted(output_dir.glob(f"model_{mode_suffix}_*.json"))
    if not existing:
        return 1

    last = existing[-1].stem
    try:
        return int(last.split("_")[-1]) + 1
    except Exception:
        return len(existing) + 1


class TrainerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Navigation Log Trainer")
        self.root.geometry("760x720")
        self.root.minsize(760, 680)
        self.tooltips: List[ToolTip] = []

        self.selected_files: List[str] = []
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "trained_models"))
        self.model_name = tk.StringVar(value="")
        self.hidden_width = tk.StringVar(value="64")
        self.hidden_layers = tk.StringVar(value="2")
        self.epochs = tk.StringVar(value="120")
        self.batch_size = tk.StringVar(value="64")
        self.learning_rate = tk.StringVar(value="0.001")
        self.activation = tk.StringVar(value="relu")

        self._build_ui()
        self._refresh_path_preview()
        self.model_name.trace_add("write", lambda *_: self._refresh_path_preview())
        self.output_dir.trace_add("write", lambda *_: self._refresh_path_preview())
        # Force a full layout pass before any modal dialog steals focus.
        # Without this, nested ttk.LabelFrame widgets on Windows can render at
        # zero height until a user-initiated event (e.g. Browse) forces a
        # redraw — which hid the Train button on first launch.
        self.root.update_idletasks()
        self.root.after(400, self.select_files)

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        file_frame = ttk.LabelFrame(self.root, text="Training Logs")
        file_frame.pack(fill="x", **pad)

        select_btn = ttk.Button(file_frame, text="Select Log Files", command=self.select_files)
        select_btn.grid(row=0, column=0, sticky="w", **pad)
        self.file_label = ttk.Label(file_frame, text="No files selected")
        self.file_label.grid(row=0, column=1, sticky="w", **pad)

        cfg_frame = ttk.LabelFrame(self.root, text="Model Settings")
        cfg_frame.pack(fill="x", **pad)

        ttk.Label(cfg_frame, text="Hidden Width:").grid(row=0, column=0, sticky="e", **pad)
        hidden_width_entry = ttk.Entry(cfg_frame, textvariable=self.hidden_width, width=10)
        hidden_width_entry.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cfg_frame, text="Hidden Layers:").grid(row=0, column=2, sticky="e", **pad)
        hidden_layers_entry = ttk.Entry(cfg_frame, textvariable=self.hidden_layers, width=10)
        hidden_layers_entry.grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cfg_frame, text="Epochs:").grid(row=1, column=0, sticky="e", **pad)
        epochs_entry = ttk.Entry(cfg_frame, textvariable=self.epochs, width=10)
        epochs_entry.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cfg_frame, text="Batch Size:").grid(row=1, column=2, sticky="e", **pad)
        batch_size_entry = ttk.Entry(cfg_frame, textvariable=self.batch_size, width=10)
        batch_size_entry.grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cfg_frame, text="Learning Rate:").grid(row=2, column=0, sticky="e", **pad)
        learning_rate_entry = ttk.Entry(cfg_frame, textvariable=self.learning_rate, width=10)
        learning_rate_entry.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(cfg_frame, text="Activation:").grid(row=3, column=0, sticky="e", **pad)
        act_frame = ttk.Frame(cfg_frame)
        act_frame.grid(row=3, column=1, columnspan=3, sticky="w")
        activation_radios: List[ttk.Radiobutton] = []
        for label, val in [("ReLU", "relu"), ("Tanh", "tanh"), ("Leaky ReLU", "leaky_relu")]:
            rb = ttk.Radiobutton(act_frame, text=label, variable=self.activation, value=val)
            rb.pack(side="left", padx=6)
            activation_radios.append(rb)

        out_frame = ttk.LabelFrame(self.root, text="Output")
        out_frame.pack(fill="x", **pad)

        ttk.Label(out_frame, text="Save Dir:").grid(row=0, column=0, sticky="e", **pad)
        output_dir_entry = ttk.Entry(out_frame, textvariable=self.output_dir, width=58)
        output_dir_entry.grid(row=0, column=1, sticky="w", **pad)
        browse_btn = ttk.Button(out_frame, text="Browse", command=self.choose_output_dir)
        browse_btn.grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(out_frame, text="Model Name:").grid(row=1, column=0, sticky="e", **pad)
        model_name_entry = ttk.Entry(out_frame, textvariable=self.model_name, width=58)
        model_name_entry.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(out_frame, text="Full Path:").grid(row=2, column=0, sticky="ne", **pad)
        self.path_preview_var = tk.StringVar(value="")
        path_preview_label = ttk.Label(
            out_frame, textvariable=self.path_preview_var, wraplength=520, justify="left", foreground="#555"
        )
        path_preview_label.grid(row=2, column=1, columnspan=2, sticky="w", **pad)

        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(side="bottom", fill="both", expand=True, **pad)

        self.status_text = tk.Text(status_frame, height=10, wrap="word")
        self.status_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.status_text.configure(state="disabled")

        action_frame = ttk.Frame(self.root)
        action_frame.pack(side="bottom", fill="x", **pad)
        self.train_button = ttk.Button(action_frame, text="Train Models", command=self.start_training)
        self.train_button.pack(anchor="w", padx=8)

        self.tooltips.extend(
            [
                ToolTip(select_btn, "Choose one or more simulator .jsonl log files for training."),
                ToolTip(self.file_label, "Shows how many log files are currently selected."),
                ToolTip(hidden_width_entry, "Neurons per hidden layer. Larger values can fit more complex behavior."),
                ToolTip(hidden_layers_entry, "Number of hidden layers in the policy network."),
                ToolTip(epochs_entry, "Number of full training passes over the data."),
                ToolTip(batch_size_entry, "Samples per optimization step."),
                ToolTip(learning_rate_entry, "Step size for model updates during training."),
                ToolTip(output_dir_entry, "Folder where trained model and metrics files are saved."),
                ToolTip(browse_btn, "Pick the output folder for model artifacts."),
                ToolTip(
                    model_name_entry,
                    "Optional custom stem for the saved files. Leave empty to auto-number as model_<mode>_NNN.json. "
                    "When multiple modes train in one run, the mode suffix is appended automatically.",
                ),
                ToolTip(
                    path_preview_label,
                    "Resolved save path(s) for the current model name and save directory.",
                ),
                ToolTip(self.train_button, "Start training models for all modes found in selected logs."),
                ToolTip(self.status_text, "Live training progress, parsed data stats, and saved file paths."),
            ]
        )
        for rb in activation_radios:
            self.tooltips.append(ToolTip(rb, "Activation function used in hidden layers."))

    def log(self, msg: str) -> None:
        def _append() -> None:
            self.status_text.configure(state="normal")
            self.status_text.insert("end", msg + "\n")
            self.status_text.see("end")
            self.status_text.configure(state="disabled")

        self.root.after(0, _append)

    def select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select log files",
            filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")],
        )
        if paths:
            self.selected_files = list(paths)
            self.file_label.config(text=f"{len(self.selected_files)} files selected")
            self.log(f"Selected {len(self.selected_files)} files")

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def _sanitize_model_stem(self, raw: str) -> str:
        stem = raw.strip()
        if stem.lower().endswith(".json"):
            stem = stem[:-5]
        bad = '<>:"/\\|?*'
        return "".join("_" if c in bad else c for c in stem)

    def _resolve_model_paths(self, mode_suffix: str) -> Tuple[Path, Path, Path]:
        output_dir = Path(self.output_dir.get())
        stem = self._sanitize_model_stem(self.model_name.get())
        if stem:
            base = f"{stem}_{mode_suffix}"
            return (
                output_dir / f"model_{base}.json",
                output_dir / f"metrics_{base}.json",
                output_dir / f"params_{base}.json",
            )
        idx = next_index(output_dir, mode_suffix) if output_dir.exists() else 1
        return (
            output_dir / f"model_{mode_suffix}_{idx:03d}.json",
            output_dir / f"metrics_{mode_suffix}_{idx:03d}.json",
            output_dir / f"params_{mode_suffix}_{idx:03d}.json",
        )

    def _refresh_path_preview(self) -> None:
        try:
            suffixes = ["heading", "strafe", "heading_strafe"]
            lines: List[str] = []
            for s in suffixes:
                model_p, _, _ = self._resolve_model_paths(s)
                lines.append(f"{s}: {model_p}")
            self.path_preview_var.set("\n".join(lines))
        except Exception as e:
            self.path_preview_var.set(f"(path preview unavailable: {e})")

    def _read_int(self, value: str, name: str, min_val: int = 1) -> int:
        try:
            out = int(value)
        except Exception as e:
            raise ValueError(f"{name} must be an integer") from e
        if out < min_val:
            raise ValueError(f"{name} must be >= {min_val}")
        return out

    def _read_float(self, value: str, name: str, min_val: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception as e:
            raise ValueError(f"{name} must be a number") from e
        if out <= min_val:
            raise ValueError(f"{name} must be > {min_val}")
        return out

    def start_training(self) -> None:
        if not self.selected_files:
            messagebox.showwarning("No files", "Select one or more log files first")
            return

        try:
            _ = self._read_int(self.hidden_width.get(), "Hidden width", 1)
            _ = self._read_int(self.hidden_layers.get(), "Hidden layers", 1)
            _ = self._read_int(self.epochs.get(), "Epochs", 1)
            _ = self._read_int(self.batch_size.get(), "Batch size", 1)
            _ = self._read_float(self.learning_rate.get(), "Learning rate", 0.0)
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self.train_button.config(state="disabled")
        t = threading.Thread(target=self._train_pipeline, daemon=True)
        t.start()

    def _save_artifacts(
        self,
        mode: str,
        model: NumpyMLP,
        input_dim: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        history: Dict[str, List[float]],
        cfg: Dict[str, object],
        n_samples: int,
        mode_meta: Optional[Dict[str, object]] = None,
    ) -> Tuple[Path, Path]:
        mode_suffix_map = {
            MODE_HEADING: "heading",
            MODE_STRAFE: "strafe",
            MODE_HEADING_STRAFE: "heading_strafe",
        }
        mode_suffix = mode_suffix_map.get(mode, "other")
        model_path, metrics_path, params_path = self._resolve_model_paths(mode_suffix)

        memory_steps = _infer_memory_steps(input_dim=input_dim, mode=mode)
        output_layout = _action_keys_for_mode(mode)

        model_blob = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "model_mode": _model_mode_name(mode),
            "input_dim": int(input_dim),
            "output_dim": int(model.biases[-1].shape[1]),
            "memory_steps": int(memory_steps),
            "whisker_dim": 11,
            "input_layout": _build_input_layout(mode=mode, memory_steps=memory_steps),
            "output_layout": output_layout,
            "history_includes_drive": bool(memory_steps > 1 and "drive_speed" in output_layout),
            "hidden_layers": int(cfg["hidden_layers"]),
            "hidden_width": int(cfg["hidden_width"]),
            "activation": str(cfg["activation"]),
            "weights": [w.tolist() for w in model.weights],
            "biases": [b.tolist() for b in model.biases],
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        }

        metrics_blob = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "samples": n_samples,
            "config": cfg,
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        }

        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_blob, f)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_blob, f)

        params_blob = self._build_params_blob(
            mode=mode,
            mode_suffix=mode_suffix,
            memory_steps=int(memory_steps),
            input_dim=int(input_dim),
            output_layout=list(output_layout),
            activation=str(cfg["activation"]),
            model_path=model_path,
            mode_meta=mode_meta or {},
        )
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params_blob, f, indent=2)

        self._last_params_path = params_path
        return model_path, metrics_path

    def _build_params_blob(
        self,
        mode: str,
        mode_suffix: str,
        memory_steps: int,
        input_dim: int,
        output_layout: List[str],
        activation: str,
        model_path: Path,
        mode_meta: Dict[str, object],
    ) -> Dict[str, object]:
        def dominant(field: str, default: float) -> float:
            info = mode_meta.get(field) if isinstance(mode_meta, dict) else None
            if isinstance(info, dict):
                val = info.get("value")
                if val is not None:
                    return float(val)
            return float(default)

        min_turn = dominant("min_turn_rate_dps", 0.0)
        inner_db = dominant("inner_deadband_dps", 0.0)
        log_rate_hz = dominant("active_log_rate_hz", 10.0)

        output_units = {
            "rotation_rate": "deg/s",
            "drive_speed": "m/s",
            "vx": "m/s",
            "vy": "m/s",
        }
        output_signals = [{"name": k, "unit": output_units.get(k, "")} for k in output_layout]

        action_slice_order_map = {
            MODE_HEADING: ["drive_speed", "rotation_rate"],
            MODE_STRAFE: ["vx", "vy"],
            MODE_HEADING_STRAFE: ["rotation_rate", "vx", "vy"],
        }

        saturation = {
            "rotation_rate_dps": 40.0,
            "drive_speed_mps": 0.40,
            "vx_mps": 0.40,
            "vy_mps": 0.40,
        }

        recommended_hysteresis = {
            "rotation_rate_enter_dps": float(min_turn) if min_turn > 0.0 else None,
            "rotation_rate_exit_dps": float(min_turn * 0.6) if min_turn > 0.0 else None,
        }

        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "schema_version": 1,
            "mode": mode,
            "mode_suffix": mode_suffix,
            "model_file": model_path.name,
            "memory_steps": int(memory_steps),
            "input_dim": int(input_dim),
            "output_dim": len(output_layout),
            "log_rate_hz": float(log_rate_hz),
            "activation": activation,
            "whisker": {
                "count": 11,
                "max_length_m": 0.50,
                "angles_deg_left_to_right": [90, 60, 45, 30, 15, 0, -15, -30, -45, -60, -90],
            },
            "input_signals": {
                "per_timestep_state": [
                    {"name": "whisker_lengths", "unit": "m", "count": 11, "range": [0.0, 0.50]},
                    {"name": "heading_to_target", "unit": "deg", "sign": "+ = target to robot LEFT"},
                ],
                "past_action_slice_order": action_slice_order_map.get(mode, []),
            },
            "output_signals": output_signals,
            "sign_conventions": {
                "frame": "right-hand, z-up",
                "rotation_rate": "+ = CCW (turn LEFT)",
                "drive_speed": "+ = forward (robot x-axis)",
                "vx": "+ = forward (robot x-axis)",
                "vy": "+ = robot LEFT (robot y-axis)",
                "heading_to_target": "+ = target on robot LEFT",
            },
            "saturation_limits": saturation,
            "stiction": {
                "min_turn_rate_dps": min_turn,
                "inner_deadband_dps": inner_db,
                "notes": (
                    "Apply the same two-threshold snap at runtime on rotation_rate before "
                    "publishing to motors. Source: training logs (dominant value across samples)."
                ),
            },
            "recommended_hysteresis": recommended_hysteresis,
            "source_metadata_counters": mode_meta,
        }

    def _log_metadata_summary(self, mode_label: str, mode_meta: Dict[str, object]) -> None:
        if not mode_meta:
            return
        for field, info in mode_meta.items():
            if not isinstance(info, dict):
                continue
            distinct = info.get("distinct") or []
            if len(distinct) > 1:
                pretty = ", ".join(f"{d['value']}x{d['count']}" for d in distinct)
                self.log(f"[{mode_label}] {field} inconsistent across logs: {pretty} (dominant wins)")

    def _train_one_mode(
        self,
        mode: str,
        dataset: Dataset,
        cfg: Dict[str, object],
        mode_meta: Optional[Dict[str, object]] = None,
    ) -> None:
        train_set, val_set = split_train_val(dataset)

        x_mean, x_std = compute_norm(train_set.x)
        y_mean, y_std = compute_norm(train_set.y)

        x_train = apply_norm(train_set.x, x_mean, x_std)
        y_train = apply_norm(train_set.y, y_mean, y_std)
        x_val = apply_norm(val_set.x, x_mean, x_std)
        y_val = apply_norm(val_set.y, y_mean, y_std)

        output_dim = dataset.y.shape[1]
        input_dim = dataset.x.shape[1]
        model = NumpyMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=int(cfg["hidden_layers"]),
            hidden_width=int(cfg["hidden_width"]),
            activation=str(cfg["activation"]),
            seed=42,
        )

        self.log(f"Training {mode}: {dataset.x.shape[0]} samples")
        history = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=int(cfg["epochs"]),
            batch_size=int(cfg["batch_size"]),
            learning_rate=float(cfg["learning_rate"]),
            seed=123,
        )

        model_path, metrics_path = self._save_artifacts(
            mode=mode,
            model=model,
            input_dim=input_dim,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            history=history,
            cfg=cfg,
            mode_meta=mode_meta or {},
            n_samples=int(dataset.x.shape[0]),
        )

        self.log(f"Saved model: {model_path}")
        self.log(f"Saved metrics: {metrics_path}")
        params_path = getattr(self, "_last_params_path", None)
        if params_path is not None:
            self.log(f"Saved params: {params_path}")
        self.log(
            f"{mode} final losses -> train: {history['train_loss'][-1]:.6f}, "
            f"val: {history['val_loss'][-1]:.6f}"
        )

    def _train_pipeline(self) -> None:
        try:
            cfg: Dict[str, object] = {
                "hidden_width": float(self._read_int(self.hidden_width.get(), "Hidden width", 1)),
                "hidden_layers": float(self._read_int(self.hidden_layers.get(), "Hidden layers", 1)),
                "epochs": float(self._read_int(self.epochs.get(), "Epochs", 1)),
                "batch_size": float(self._read_int(self.batch_size.get(), "Batch size", 1)),
                "learning_rate": self._read_float(self.learning_rate.get(), "Learning rate", 0.0),
                "activation": self.activation.get(),
            }

            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            self.log("Reading selected log files...")
            datasets, stats, inferred_input_dims, metadata_summary = parse_logs(self.selected_files)
            self.log(
                "Parsed logs: "
                f"files={stats['files']}, lines={stats['lines']}, valid={stats['valid']}, invalid={stats['invalid']}"
            )
            self.log(
                "Mode counts: "
                f"heading={stats['heading_samples']}, "
                f"strafe={stats['strafe_samples']}, "
                f"heading_strafe={stats['heading_strafe_samples']}"
            )
            if inferred_input_dims:
                self.log(
                    "Inferred input dims: "
                    + ", ".join(f"{m}={d}" for m, d in sorted(inferred_input_dims.items()))
                )

            if not datasets:
                raise ValueError("No valid samples found in selected files")

            for mode_key, mode_label in (
                (MODE_HEADING, "heading_drive"),
                (MODE_STRAFE, "xy_strafe"),
                (MODE_HEADING_STRAFE, "heading_strafe"),
            ):
                if mode_key in datasets:
                    self._log_metadata_summary(mode_label, metadata_summary.get(mode_key, {}))
                    self._train_one_mode(mode_key, datasets[mode_key], cfg, metadata_summary.get(mode_key, {}))
                else:
                    self.log(f"Skipping {mode_label}: no samples")

            self.log("Training complete")
            self.root.after(0, lambda: messagebox.showinfo("Done", "Training complete"))
        except Exception as e:
            self.log(f"ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.root.after(0, lambda: self.train_button.config(state="normal"))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = TrainerApp()
    app.run()


if __name__ == "__main__":
    main()
