import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_logs(file_paths: List[str]) -> Tuple[Dict[str, Dataset], Dict[str, int], Dict[str, int]]:
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

    return datasets, stats, inferred_input_dims


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
        self.root.geometry("760x560")
        self.tooltips: List[ToolTip] = []

        self.selected_files: List[str] = []
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "trained_models"))
        self.hidden_width = tk.StringVar(value="64")
        self.hidden_layers = tk.StringVar(value="2")
        self.epochs = tk.StringVar(value="120")
        self.batch_size = tk.StringVar(value="64")
        self.learning_rate = tk.StringVar(value="0.001")
        self.activation = tk.StringVar(value="relu")

        self._build_ui()
        self.root.after(150, self.select_files)

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

        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", **pad)
        self.train_button = ttk.Button(action_frame, text="Train Models", command=self.start_training)
        self.train_button.pack(anchor="w", padx=8)

        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(fill="both", expand=True, **pad)

        self.status_text = tk.Text(status_frame, height=14, wrap="word")
        self.status_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.status_text.configure(state="disabled")

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
        output_dir: Path,
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
    ) -> Tuple[Path, Path]:
        mode_suffix_map = {
            MODE_HEADING: "heading",
            MODE_STRAFE: "strafe",
            MODE_HEADING_STRAFE: "heading_strafe",
        }
        mode_suffix = mode_suffix_map.get(mode, "other")
        idx = next_index(output_dir, mode_suffix)

        model_path = output_dir / f"model_{mode_suffix}_{idx:03d}.json"
        metrics_path = output_dir / f"metrics_{mode_suffix}_{idx:03d}.json"

        model_blob = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "input_dim": int(input_dim),
            "output_dim": int(model.biases[-1].shape[1]),
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

        return model_path, metrics_path

    def _train_one_mode(
        self,
        mode: str,
        dataset: Dataset,
        cfg: Dict[str, object],
        output_dir: Path,
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
            output_dir=output_dir,
            mode=mode,
            model=model,
            input_dim=input_dim,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            history=history,
            cfg=cfg,
            n_samples=int(dataset.x.shape[0]),
        )

        self.log(f"Saved model: {model_path}")
        self.log(f"Saved metrics: {metrics_path}")
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
            datasets, stats, inferred_input_dims = parse_logs(self.selected_files)
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

            if MODE_HEADING in datasets:
                self._train_one_mode(MODE_HEADING, datasets[MODE_HEADING], cfg, output_dir)
            else:
                self.log("Skipping heading_drive: no samples")

            if MODE_STRAFE in datasets:
                self._train_one_mode(MODE_STRAFE, datasets[MODE_STRAFE], cfg, output_dir)
            else:
                self.log("Skipping xy_strafe: no samples")

            if MODE_HEADING_STRAFE in datasets:
                self._train_one_mode(MODE_HEADING_STRAFE, datasets[MODE_HEADING_STRAFE], cfg, output_dir)
            else:
                self.log("Skipping heading_strafe: no samples")

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
