# RoboSim MLP Model — Implementation Spec

This document is intended to be handed to a separate implementation that
loads and runs a trained model JSON. It is derived directly from the
trainer source code (`train_mlp.py`, `sim_core.py`, `headless_sim.py`)
and is authoritative for all three control modes.

---

## 1. Model JSON top-level fields

| Field | Type | Description |
|---|---|---|
| `mode` | string | `"heading_drive"` \| `"xy_strafe"` \| `"heading_strafe"` |
| `model_mode` | string | Short alias: `"heading"` \| `"strafe"` \| `"heading_strafe"` |
| `input_dim` | int | Total length of the flat input vector fed to the network |
| `output_dim` | int | Number of output channels (see Section 4) |
| `memory_steps` | int | Number of timesteps `N` in the history window (≥ 1) |
| `whisker_dim` | int | Always `11` |
| `input_layout` | string[] | Human-readable slot names, length == `input_dim` |
| `output_layout` | string[] | Output channel names in order, length == `output_dim` |
| `hidden_layers` | int | Number of hidden layers |
| `hidden_width` | int | Neurons per hidden layer |
| `activation` | string | `"relu"` \| `"tanh"` \| `"leaky_relu"` |
| `weights` | float[][][] | Per-layer weight matrices, shape `[fan_in, fan_out]` |
| `biases` | float[][][] | Per-layer bias row vectors, shape `[1, fan_out]` |
| `x_mean` | float[][] | Input normalisation mean, shape `[1, input_dim]` |
| `x_std` | float[][] | Input normalisation std dev, shape `[1, input_dim]` |
| `y_mean` | float[][] | Output un-normalisation mean, shape `[1, output_dim]` |
| `y_std` | float[][] | Output un-normalisation std dev, shape `[1, output_dim]` |

`len(weights) == len(biases) == hidden_layers + 1`.

---

## 2. Sign & unit conventions

All externally-visible quantities use a **right-hand, z-up** yaw convention.

| Quantity | Unit | Sign convention |
|---|---|---|
| `whisker_lengths[i]` | metres | Always ≥ 0; equals `WHISKER_MAX_LEN_M` (0.50 m) when nothing detected |
| `heading_to_target` | degrees | **Positive = target to robot's LEFT** (CCW positive) |
| `rotation_rate` | deg/s | **Positive = turn LEFT** (CCW positive) |
| `drive_speed` | m/s | Positive = forward |
| `vx` | m/s | Positive = forward (along robot x-axis) |
| `vy` | m/s | Positive = robot's LEFT (along robot y-axis) |

Physical saturation limits (used by the simulator and for clamping
real-robot commands):

| Channel | Limit |
|---|---|
| `rotation_rate` | ±40.0 deg/s |
| `drive_speed` | ±0.40 m/s |
| `vx` | ±0.40 m/s |
| `vy` | ±0.40 m/s |

---

## 3. Input feature vector

### 3.1 Whisker sensor layout

11 ray-cast distance sensors. Stored in the feature vector in this fixed
index order, always left-to-right relative to the robot:

| Index | Angle off forward (right-hand) | Physical direction |
|---|---|---|
| 0 | +90° | robot LEFT |
| 1 | +60° | |
| 2 | +45° | |
| 3 | +30° | |
| 4 | +15° | |
| 5 | 0° | straight FORWARD |
| 6 | −15° | |
| 7 | −30° | |
| 8 | −45° | |
| 9 | −60° | |
| 10 | −90° | robot RIGHT |

Values are raw metres in `[0.0, 0.50]`. Do not pre-normalise; the stored
`x_mean` / `x_std` handle all normalisation.

### 3.2 Flat feature construction

For a history window of `N` timesteps (`memory_steps` from the blob),
labelled oldest-first as `t[0] … t[N-1]` where `t[N-1]` is the current
step:

```
feature_vector = []
for i in 0 .. N-1:
    feature_vector += t[i].whisker_lengths   # 11 floats, metres
    feature_vector += [t[i].heading_to_target]  # 1 float, degrees
    if i < N-1:                               # NOT appended for the last (current) step
        feature_vector += action_slice(t[i])  # physical units — see Section 3.3
```

`len(feature_vector) == input_dim == N*12 + (N-1)*A`
where `A` is the action dimension for the mode (2 or 3).

### 3.3 Action slice ordering inside the feature vector

The past-action fields appended for each non-final step follow this order,
**in physical units (not normalised)**:

| Mode | Action slice order | A |
|---|---|---|
| `heading_drive` | `drive_speed`, `rotation_rate` | 2 |
| `xy_strafe` | `vx`, `vy` | 2 |
| `heading_strafe` | `rotation_rate`, `vx`, `vy` | 3 |

Note the `heading_drive` order is **drive_speed first, rotation_rate second**.
This is the canonical training order. It differs from the order used by
`Robot.apply_normalized_action` inside the simulator (which takes
`[rotation_rate_norm, drive_speed_norm]`), but that function is never
exposed externally.

### 3.4 Input dimensionality table

| Mode | A | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| `heading_drive` | 2 | 12 | 26 | 68 | 138 |
| `xy_strafe` | 2 | 12 | 26 | 68 | 138 |
| `heading_strafe` | 3 | 12 | 27 | 72 | 147 |

To infer `N` from the blob: `N = (input_dim + A) / (12 + A)`.

### 3.5 Zero-padding on episode start

At the start of an episode the history deque has fewer than `N` real
observations. Pad the deque on the **left** (oldest positions) with zero
entries:

```
zero_entry = {
    whisker_lengths: [0.0] * 11,   # NOT 0.5 — actual zero
    heading_to_target: 0.0,
    action: {each_key: 0.0}
}
```

Use the same zero-padding at deployment; the training distribution was
built with these exact zeros. Any different padding value will shift the
normalised inputs out of distribution.

---

## 4. Output vector

Model output is in **physical units** after un-normalisation (not `[−1, 1]`).

| Mode | Output order | Dim |
|---|---|---|
| `heading_drive` | `drive_speed` (m/s), `rotation_rate` (deg/s) | 2 |
| `xy_strafe` | `vx` (m/s), `vy` (m/s) | 2 |
| `heading_strafe` | `rotation_rate` (deg/s), `vx` (m/s), `vy` (m/s) | 3 |

The `output_layout` field in the blob lists these names in order and is
the authoritative source. Always cross-check your indexing against it.

Clamp each channel to the limits in Section 2 before sending to motors.

---

## 5. Inference procedure

```python
# x_raw  : float array, shape (input_dim,) or (1, input_dim)
# blob   : parsed model JSON (weights/biases/x_mean/x_std/y_mean/y_std/activation)

LEAKY_ALPHA = 0.01

def apply_activation(z, activation):
    if activation == "tanh":
        return tanh(z)
    elif activation == "leaky_relu":
        return where(z > 0, z, LEAKY_ALPHA * z)
    else:  # "relu" (default)
        return maximum(0.0, z)

def infer(x_raw, blob):
    x_mean = blob["x_mean"]                          # shape (1, input_dim)
    x_std  = where(abs(blob["x_std"]) < 1e-6,
                   1.0, blob["x_std"])               # shape (1, input_dim)
    y_mean = blob["y_mean"]                          # shape (1, output_dim)
    y_std  = where(abs(blob["y_std"]) < 1e-6,
                   1.0, blob["y_std"])               # shape (1, output_dim)

    a = (x_raw - x_mean) / x_std                    # standard-score input

    for i in range(len(blob["weights"]) - 1):        # hidden layers
        z = a @ blob["weights"][i] + blob["biases"][i]
        a = apply_activation(z, blob["activation"])

    z_out = a @ blob["weights"][-1] + blob["biases"][-1]   # linear output layer
    y_norm = z_out
    y_phys = y_norm * y_std + y_mean                 # un-normalise → physical units

    return y_phys.flatten()   # length output_dim, physical units
```

**Critical**: the return value is physical units. Do NOT pass it through a
`tanh` or clip to `[−1, 1]` as if it were a normalised policy output.

---

## 6. History buffer management at runtime

```
On episode reset:
    deque.clear()
    fill deque with (memory_steps - 1) zero entries   # oldest first

Each control step:
    1. Read sensors → build current_state = {whisker_lengths, heading_to_target}
    2. Append {state: current_state, action: last_issued_command} to deque
       (use zero action if no command has been issued yet)
    3. Flatten deque to feature vector per Section 3.2
    4. Call infer(feature_vector, blob)
    5. Clamp output to physical limits (Section 2)
    6. Send clamped command to motors
    7. Store clamped command as last_issued_command for next step
```

The `action` stored with each step is the command that was **actually
sent** after clamping, not the raw network output.

---

## 7. Normalisation details

`x_mean` and `x_std` are computed from the **training split only** (80%
of shuffled data). `std` values below `1e-6` are replaced with `1.0` to
prevent division by zero.

Because early-episode observations include zero-padded history slots, the
per-feature means for older-history positions are slightly biased toward
zero. This is intentional — it matches the zero-padding at inference.

The `y_mean` / `y_std` are computed the same way over the label (action)
distribution. They encode the typical magnitude and centre of each output
channel, so the linear output layer can predict near-zero via near-zero
activations.

---

## 8. Checklist for a new consumer implementation

- [ ] Load `activation` from blob; do not hardcode ReLU.
- [ ] Feature vector constructed with whiskers **before** heading at each step.
- [ ] Action slices use the mode-specific order in Section 3.3 (note `heading_drive` is `drive_speed` first).
- [ ] Action slices in physical units (m/s, deg/s) — **not** normalised.
- [ ] No action slice appended for the final (current) timestep.
- [ ] History deque seeded with **zero** entries (not `WHISKER_MAX` or any other value).
- [ ] Input normalised with `(x - x_mean) / x_std` using blob's stored stats.
- [ ] Output un-normalised with `y_norm * y_std + y_mean` → physical units.
- [ ] Output clamped to ±40 deg/s (rotation), ±0.40 m/s (translation) before motor send.
- [ ] `heading_to_target` sign: positive = target on robot's LEFT.
- [ ] `rotation_rate` sign: positive = turn LEFT (CCW).
- [ ] `vy` sign: positive = robot's LEFT.
- [ ] `len(feature_vector) == input_dim` verified before every forward pass.
- [ ] `output_layout` field cross-checked against consumer's channel mapping.
