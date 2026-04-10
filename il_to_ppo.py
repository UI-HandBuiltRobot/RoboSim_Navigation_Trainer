"""Convert an Imitation Learning JSON model to a Stable-Baselines3 PPO ZIP.

The converter bakes IL input normalisation (x_mean / x_std) into the first
linear layer and IL output denormalisation (y_mean / y_std) + PPO physical
action scale into the output layer so that the SB3 network operates directly
on raw observations and produces normalised [-1, 1] actions — exactly the
interface expected by NavigationEnv.

Usage (command line):
    python il_to_ppo.py path/to/il_model.json path/to/output.zip

Programmatic:
    from il_to_ppo import convert_il_to_ppo
    convert_il_to_ppo("model.json", "output.zip")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from nav_env import NavigationEnv

# Physical scales mapping normalised PPO action [-1, 1] → physical units.
# Must match sim_core.Robot.apply_normalized_action (GAMEPAD_MAX constants).
_PHYSICAL_SCALES: Dict[str, np.ndarray] = {
    # IL output order for heading_drive is [drive_speed, rotation_rate].
    "heading_drive": np.array([0.40, 40.0], dtype=np.float64),
    "xy_strafe":     np.array([0.40, 0.40], dtype=np.float64),
}

# Mapping from IL output order -> NavigationEnv action order.
# heading_drive IL output: [drive, rotation]
# env action order:       [rotation, drive]
_IL_TO_ENV_ACTION_ORDER: Dict[str, np.ndarray] = {
    "heading_drive": np.array([1, 0], dtype=np.int64),
    "xy_strafe": np.array([0, 1], dtype=np.int64),
}

# Action dimension per control mode (PPO action space size).
_N_ACTION: Dict[str, int] = {"heading_drive": 2, "xy_strafe": 2}

# State features per timestep: 11 whisker lengths + 1 heading-to-target.
_STATE_DIM = 12


def _compute_history_window(input_dim: int, n_action: int) -> int:
    """Infer history_window N from the IL model's input_dim.

    The IL obs layout is:  obs = [s_0, a_0, s_1, ..., a_{N-2}, s_{N-1}]
    where each s_i has _STATE_DIM features and each a_i has n_action features.
    Total dim = _STATE_DIM*N + n_action*(N-1)
               = (_STATE_DIM + n_action)*N - n_action

    Rearranging: N = (input_dim + n_action) / (_STATE_DIM + n_action)
    """
    numerator = input_dim + n_action
    denominator = _STATE_DIM + n_action
    if numerator % denominator != 0:
        raise ValueError(
            f"Cannot derive an integer history_window from input_dim={input_dim}, "
            f"n_action={n_action}. "
            f"Expected input_dim = {_STATE_DIM}*N + {n_action}*(N-1) for some N≥1."
        )
    return numerator // denominator


def convert_il_to_ppo(il_path: str | Path, zip_out_path: str | Path) -> None:
    """Load an IL JSON model and write a weight-initialised SB3 PPO ZIP.

    The value network (critic) retains its random SB3 initialisation.

    Args:
        il_path: Path to the IL model JSON file produced by train_mlp.py.
        zip_out_path: Destination path. SB3 always appends '.zip', so pass the
                      path either with or without that suffix — both are handled.
    """
    il_path = Path(il_path)
    zip_out_path = Path(zip_out_path)
    # SB3 adds '.zip' automatically; strip a trailing suffix to avoid e.g. 'x.zip.zip'.
    if zip_out_path.suffix.lower() == ".zip":
        zip_out_path = zip_out_path.with_suffix("")

    with open(il_path, "r", encoding="utf-8") as f:
        il: dict = json.load(f)

    mode: str = str(il.get("mode", "heading_drive"))
    if mode not in _PHYSICAL_SCALES:
        raise ValueError(
            f"IL model mode '{mode}' is not supported for PPO conversion. "
            f"Supported modes: {list(_PHYSICAL_SCALES.keys())}."
        )

    n_action: int      = _N_ACTION[mode]
    input_dim: int     = int(il["input_dim"])
    hidden_width: int  = int(il["hidden_width"])
    hidden_layers: int = int(il["hidden_layers"])

    weights = [np.array(w, dtype=np.float64) for w in il["weights"]]
    biases  = [np.array(b, dtype=np.float64).squeeze() for b in il["biases"]]
    x_mean  = np.array(il["x_mean"], dtype=np.float64).squeeze()
    x_std   = np.array(il["x_std"],  dtype=np.float64).squeeze()
    y_mean  = np.array(il["y_mean"], dtype=np.float64).squeeze()
    y_std   = np.array(il["y_std"],  dtype=np.float64).squeeze()

    # Validate layer count: hidden_layers hidden matrices + 1 output matrix.
    expected = hidden_layers + 1
    if len(weights) != expected:
        raise ValueError(
            f"IL JSON has {len(weights)} weight matrices but expected {expected} "
            f"(hidden_layers={hidden_layers} + 1 output)."
        )

    history_window = _compute_history_window(input_dim, n_action)
    net_arch = [hidden_width] * hidden_layers
    physical_scales = _PHYSICAL_SCALES[mode]

    config = {"control_mode": mode, "history_window": history_window}
    vec_env = DummyVecEnv([lambda: NavigationEnv(config=config, seed=0)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=net_arch, activation_fn=torch.nn.ReLU),
        verbose=0,
    )

    # ------------------------------------------------------------------ #
    # Weight transfer                                                     #
    # ------------------------------------------------------------------ #
    # IL forward pass:                                                    #
    #   x_norm = (x - x_mean) / x_std                                    #
    #   h = relu(x_norm @ W0 + b0)                                       #
    #   ... (intermediate hidden layers) ...                              #
    #   out_norm = h @ W_last + b_last                                    #
    #   physical = out_norm * y_std + y_mean                              #
    #   ppo_action = physical / physical_scales                           #
    #                                                                     #
    # We bake these transforms into the first and last SB3 layers so SB3  #
    # receives raw observations and outputs normalised actions directly.  #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        policy_net = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net

        linear_layers = [m for m in policy_net if isinstance(m, torch.nn.Linear)]

        # --- Input layer: bake x_mean / x_std ---
        # New layer computes:  x @ W0_new + b0_new
        # = (x - x_mean)/x_std @ W0 + b0
        W0 = weights[0]  # (input_dim, hidden_width)
        b0 = biases[0]   # (hidden_width,)
        safe_std = np.where(x_std == 0.0, 1.0, x_std)
        W0_new = W0 / safe_std[:, np.newaxis]
        b0_new = b0 - (x_mean / safe_std) @ W0
        linear_layers[0].weight.copy_(torch.from_numpy(W0_new.T.astype(np.float32)))
        linear_layers[0].bias.copy_(torch.from_numpy(b0_new.astype(np.float32)))

        # --- Intermediate hidden layers: direct transpose copy ---
        for i in range(1, hidden_layers):
            layer = linear_layers[i]
            layer.weight.copy_(torch.from_numpy(weights[i].T.astype(np.float32)))
            layer.bias.copy_(torch.from_numpy(biases[i].astype(np.float32)))

        # --- Output layer: bake y_mean / y_std + physical scale ---
        # IL: out_phys (IL-order) = (h @ W_last + b_last) * y_std + y_mean
        # We convert to PPO normalised action in IL order first, then reorder to
        # NavigationEnv action order before writing action_net weights.
        W_last = weights[-1]  # (hidden_width, n_output)
        b_last = biases[-1]   # (n_output,)
        output_scale = y_std / physical_scales
        W_last_new = W_last * output_scale[np.newaxis, :]
        b_last_new = b_last * output_scale + y_mean / physical_scales

        order = _IL_TO_ENV_ACTION_ORDER.get(mode, np.arange(n_action))
        W_last_new = W_last_new[:, order]
        b_last_new = b_last_new[order]
        action_net.weight.copy_(torch.from_numpy(W_last_new.T.astype(np.float32)))
        action_net.bias.copy_(torch.from_numpy(b_last_new.astype(np.float32)))

    vec_env.close()
    model.save(str(zip_out_path))
    final_path = str(zip_out_path) + ".zip"
    print(
        f"IL → PPO conversion complete.\n"
        f"  Source  : {il_path}\n"
        f"  Output  : {final_path}\n"
        f"  Mode    : {mode}  history_window={history_window}  net_arch={net_arch}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python il_to_ppo.py <il_model.json> <output.zip>")
        sys.exit(1)
    convert_il_to_ppo(sys.argv[1], sys.argv[2])
