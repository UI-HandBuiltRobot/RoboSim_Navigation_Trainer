# RoboSim Navigation Trainer — Data & Model Spec

This document is the authoritative description of how the simulator organizes
and reports data, how that data becomes training features for the MLP
trainer, and what a trained model JSON promises to a downstream consumer
(e.g. a real robot).

All quantities are in SI units unless noted. All angles are in degrees.

---

## 1. Coordinate conventions

### 1.1 World frame

- Right-handed, **z-up**. The simulator is 2-D, so only x and y are modeled.
- Origin at the bottom-left corner of the room.
- Room extents: `x ∈ [0, ROOM_WIDTH]`, `y ∈ [0, ROOM_HEIGHT]`
  (`ROOM_WIDTH = 5.0 m`, `ROOM_HEIGHT = 3.0 m`).

### 1.2 Robot body frame

- `x_body` points forward (nose of the robot).
- `y_body` points to the robot's left.
- `z_body` points up (out of the floor).
- At `heading_deg = 0`, the robot faces `+y_world`. So `x_body` = `+y_world`
  and `y_body` = `-x_world`.

### 1.3 Yaw convention (important — changed)

**All externally-visible yaw quantities are right-hand positive about
`+z_up`:**

- `rotation_rate > 0` ⇒ robot turns **left** (CCW as seen from above).
- `heading_to_target > 0` ⇒ target lies to the robot's **left**.
- Reducing a positive `heading_to_target` therefore requires a positive
  `rotation_rate`.

Internally `Robot.heading_deg` is still stored as a CW-from-`+y_world` scalar
(it preserves the legacy relationship `forward = (sin θ, cos θ)`). The
integrator negates `rotation_rate` when applying it, so this is purely an
implementation detail; no external caller ever sees or produces a left-hand
yaw value.

> Migration note: logs recorded before this change used the opposite sign
> for `heading_to_target` and `rotation_rate`. They must be re-collected (or
> mirrored at parse time) before being fed to the current trainer; mixing
> old and new logs will teach the network two opposite policies.

---

## 2. Sensors

### 2.1 Whiskers

- 11 ray-cast distance sensors on the robot body.
- Angles (degrees off the forward axis, in the right-hand yaw convention):
  `[+90, +60, +45, +30, +15, 0, −15, −30, −45, −60, −90]`.
  (Equivalently, in the stored `WHISKER_ANGLES` array this is
  `[-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]`, evaluated against the
  legacy CW-positive internal heading; index 0 always points to the robot's
  left, index 5 straight ahead, index 10 to the right.)
- Origin offset: each ray originates 0.12 m forward of the robot centre.
- Values are clamped to `[0.0, WHISKER_MAX_LEN_M]` (0.50 m). A value equal
  to `WHISKER_MAX_LEN_M` means "nothing in range".
- Units: metres. Raw, **not normalised** in the log.

Each logged row exposes whisker values as a JSON array ordered exactly as
above (left → forward → right), stored under `whisker_lengths`.

### 2.2 Heading to target

- `heading_to_target_deg` — signed angle, degrees, right-hand positive,
  wrapped to `(-180, +180]`.
- Computed by `Robot.compute_heading_to_target(tx, ty)`.
- A value of 0 means the target is directly in front of the robot.

---

## 3. Control modes and drive commands

There are three control modes. Each defines its own `drive_command` schema
and its own action dimensionality.

| Mode              | Keys (in order)                         | Dim | Physical scales                        |
|-------------------|-----------------------------------------|-----|----------------------------------------|
| `heading_drive`   | `drive_speed`, `rotation_rate`          | 2   | ±0.40 m/s, ±40 deg/s                   |
| `xy_strafe`       | `vx`, `vy`                              | 2   | ±0.40 m/s, ±0.40 m/s                   |
| `heading_strafe`  | `rotation_rate`, `vx`, `vy`             | 3   | ±40 deg/s, ±0.40 m/s, ±0.40 m/s        |

Notes:

- `drive_speed` is forward speed along `x_body` (m/s, positive = forward).
- `vx`, `vy` are body-frame translational velocities (m/s). `vy` is
  positive toward the robot's **left**.
- `rotation_rate` is the yaw rate in deg/s, right-hand positive
  (positive = turn left).
- The physical scale table is authoritative for all three pipelines:
  simulator, headless sim, trainer, and benchmark GUI. It lives in
  `Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS` / `Robot.GAMEPAD_MAX_ROTATE_RATE_DPS`
  (and the mirrored `headless_sim._PHYSICAL_SCALES`). Anything that
  de-normalises a policy action must use these constants.

### 3.1 Normalised policy actions

`Robot.apply_normalized_action(action)` takes a vector in `[-1, 1]` and
multiplies componentwise by the physical scales above. The component order
matches the row in the table **except** that `heading_drive` takes the
normalised vector as `[rotation_rate_norm, drive_speed_norm]` — rotation
first, then drive. (This is the one place in the codebase where the order
differs from the logged/training order; see Section 5.2.)

---

## 4. Log format (JSON Lines, schema v2)

Files live under `logs/<mode_suffix>_mem<N>/run_###_<mode_suffix>_mem<N>.jsonl`
where `mode_suffix ∈ {heading, strafe, heading_strafe}`. Legacy v1 rows
(flat, no `history` key, `history_len == 1`) are still parseable by
`train_mlp.parse_logs`.

One row per logged timestep:

```json
{
  "schema_version": 2,
  "timestamp": 1775869589.657,
  "mode": "heading_strafe",
  "history_len": 5,
  "active_log_rate_hz": 5.0,
  "history": [
    {"mode": "heading_strafe",
     "whisker_lengths": [... 11 floats ...],
     "heading_to_target": 0.0,
     "action": {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}},
    ...
    {"mode": "heading_strafe",
     "whisker_lengths": [... 11 floats ...],
     "heading_to_target": 42.755,
     "action": {"rotation_rate": 0.001, "vx": 1.4e-5, "vy": 0.0}}
  ]
}
```

Rules:

- `history` has length `history_len`, oldest first. The **last** entry is
  the most recent timestep.
- Early-in-episode rows are padded on the left with zero entries
  (`_zero_timestep_record`). A zero entry has zero whiskers, zero heading,
  and a zero action dict. The trainer uses exactly the same zero-padding;
  a real-robot consumer must do likewise or the first few inferences will
  mis-match the training distribution.
- `whisker_lengths` and `heading_to_target` are raw (metres, degrees).
- Each `action` dict always contains exactly the keys for that mode, in
  any order — the parser re-orders them canonically.
- Logging cadence is adaptive (see `_current_log_interval`). It does not
  affect data shape, only density.
- Snapshots and logs are only written on success (goal reached). Collision
  failures are discarded at the log level; they can be queued separately
  via the DAgger "train-me" mechanism.

---

## 5. Trainer features and labels

All of the following lives in `train_mlp.py`.

### 5.1 Canonical action-key order (per mode)

```
heading_drive   → ["drive_speed", "rotation_rate"]
heading_strafe  → ["rotation_rate", "vx", "vy"]
xy_strafe       → ["vx", "vy"]
```

This is the single source of truth; it is used for **both** feature
flattening and label construction, and is echoed as `output_layout` in the
saved model JSON.

### 5.2 Feature vector (`_flatten_history_features`)

For a row with `history_len = N` entries in chronological order
`h[0] ... h[N-1]`:

```
for i in 0 .. N-1:
    append h[i].whisker_lengths   # 11 floats, metres
    append h[i].heading_to_target # 1 float, degrees
    if i < N-1:                   # action present for all but last step
        for k in action_keys_for_mode(mode):
            append h[i].action[k] # physical units
```

So per timestep the "state slice" is always 12 wide
(`STATE_INPUT_DIM = 11 + 1`), and inter-step "action slices" have width
`|action_keys|`. The resulting input dimensionality is:

```
input_dim = N * STATE_INPUT_DIM + (N - 1) * |action_keys|
          = N * 12 + (N - 1) * A
```

| Mode             | A | N=1 | N=5 | N=10 |
|------------------|---|-----|-----|------|
| heading_drive    | 2 | 12  | 68  | 138  |
| heading_strafe   | 3 | 12  | 72  | 147  |
| xy_strafe        | 2 | 12  | 68  | 138  |

`input_layout` in the model JSON spells out each slot explicitly
(`"t-4.whiskers"`, `"t-4.heading_to_target"`, `"t-4.action.drive_speed"`,
..., `"t+0.whiskers"`, `"t+0.heading_to_target"`). Consumers should verify
they're feeding features in exactly this order.

### 5.3 Label vector

The label for a row is the **last** step's action, in the mode's canonical
order:

```
heading_drive   → [drive_speed, rotation_rate]          # physical
heading_strafe  → [rotation_rate, vx, vy]               # physical
xy_strafe       → [vx, vy]                              # physical
```

### 5.4 Normalisation

Per-feature mean/std computed **only from the training split** and saved
alongside the model:

```
x_mean, x_std  ← shape (1, input_dim),  std floored at EPS (1e-6)
y_mean, y_std  ← shape (1, output_dim)
```

Both inputs and outputs are standard-scored during training. Inference
reverses the output transform after the forward pass — see Section 6.2.

Because zero-padded history steps contribute real data points to the
statistics, early-history features have `x_mean` shifted toward zero
relative to mid-episode statistics. This is intentional as long as
inference uses the same zero-padding convention; do not substitute a
different padding value at deployment.

---

## 6. Model JSON schema

Written by `TrainerApp._save_artifacts`:

```jsonc
{
  "saved_at": "2026-04-13T12:34:56",
  "mode": "heading_drive" | "xy_strafe" | "heading_strafe",
  "model_mode": "heading" | "strafe" | "heading_strafe",
  "input_dim":  <int>,
  "output_dim": <int>,
  "memory_steps": <int>,       // N above
  "whisker_dim": 11,
  "input_layout":  [<str>, ...],
  "output_layout": [<str>, ...],  // same as action_keys_for_mode
  "history_includes_drive": <bool>,
  "hidden_layers": <int>,
  "hidden_width":  <int>,
  "activation": "relu" | "tanh" | "leaky_relu",
  "weights": [ [[..]], ... ],  // per-layer float matrices, (fan_in, fan_out)
  "biases":  [ [[..]], ... ],  // per-layer float row vectors, (1, fan_out)
  "x_mean": [[..]], "x_std": [[..]],
  "y_mean": [[..]], "y_std": [[..]]
}
```

A companion `metrics_*.json` captures training curves and the full config
used for the run; it's not required at inference.

### 6.1 Network topology

Sizes are `[input_dim, hidden_width, ..., hidden_width, output_dim]` with
`hidden_layers` hidden layers. Hidden layers use the activation named in
the blob; the output layer is linear. The loss is MSE on the standard-scored
label.

### 6.2 Inference (reference implementation)

```python
a = (x_raw - x_mean) / x_std
for i in range(len(weights) - 1):
    z = a @ weights[i] + biases[i]
    a = {
        "relu":       lambda z: np.maximum(0.0, z),
        "tanh":       np.tanh,
        "leaky_relu": lambda z: np.where(z > 0.0, z, 0.01 * z),
    }[activation](z)
y_norm = a @ weights[-1] + biases[-1]
y_phys = y_norm * y_std + y_mean            # physical units
```

> **The model's output is in physical units (m/s and deg/s), not
> normalised `[-1, 1]`.** A real-robot consumer that expects normalised
> commands must divide `drive_speed` / `vx` / `vy` by 0.40 and
> `rotation_rate` by 40.0 — or better, scale by
> `Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS` / `Robot.GAMEPAD_MAX_ROTATE_RATE_DPS`.
> This is the single most common source of "magnitudes don't match
> simulator" bugs on deployment.

The inference path in `simulator.py` and `benchmark_gui.py` previously
hard-coded the ReLU activation regardless of the value stored in the blob.
Both now honour the `activation` field; re-export any model saved before
this change if it was trained with `tanh` or `leaky_relu`.

---

## 7. End-to-end data flow (portability audit)

### 7.1 At the simulator (training data producer)

1. Robot state is advanced by `Robot.step(dt, obstacles)`.
2. `Robot.update_heading_to_target(target)` refreshes `heading_to_target_deg`
   using the right-hand sign convention (Section 1.3).
3. `SimulatorApp._capture_timestep_record(mode)` snaps a dict with raw
   whiskers, raw heading, and the **currently active** `drive_command`.
4. `_collect_log_entry` bundles the last `history_len` captures (left-padded
   with zeros) and appends the row to `in_memory_log`. The row is only
   flushed to disk if the episode reaches the goal.

### 7.2 At the trainer

1. `parse_logs` reads rows, normalises action-dict ordering via
   `_action_keys_for_mode(mode)`, left-pads short histories, and emits
   `(x_row, y_row)` pairs per row.
2. Per-mode datasets are split 80/20. `x_mean/x_std` and `y_mean/y_std`
   are computed **only from the training split** — do not re-compute these
   from the full log on a real robot.
3. A `NumpyMLP` is trained with MSE on the standard-scored data.
4. The blob described in Section 6 is serialised as `model_<mode>_###.json`.

### 7.3 At the headless evaluator (apples-to-apples sim benchmarking)

1. `HeadlessSimulator` builds observations in the **same** layout as
   training (`_build_observation`) — raw whiskers + raw heading + past
   physical actions, zero-padded on reset.
2. `benchmark_gui._predict_il_physical` runs the forward pass per
   Section 6.2 and returns physical units.
3. `_to_env_normalized_action` divides by the physical scales to obtain
   an action in `[-1, 1]`, which `Robot.apply_normalized_action` expects.
   Component order here is the only place we re-map `heading_drive` from
   the training-side order `[drive_speed, rotation_rate]` to the env-side
   order `[rotation_rate, drive_speed]`. Nothing outside `benchmark_gui.py`
   or `_update_command_from_model` is allowed to do this remap; the
   training pipeline and the logs are always in the Section 5.1 order.

### 7.4 At a real-robot consumer

A node that loads `model_<mode>_###.json` must implement the following
pipeline exactly, otherwise simulator/real-robot behaviour will diverge:

1. Match the **yaw convention** (Section 1.3). Whatever platform the
   robot uses, `rotation_rate > 0` must mean CCW (turn left), and the
   measurement that becomes `heading_to_target` must be positive when the
   target is on the left.
2. Collect sensor readings in metres, ordered left → forward → right
   (see Section 2.1). Clamp to `[0, 0.5]`.
3. Maintain a rolling deque of length `memory_steps` (from the blob).
   Seed it with zero entries at start-up. After each inference, append
   the `(state, action)` pair just produced; oldest falls off the front.
4. Flatten exactly per Section 5.2 using `input_layout` as the reference,
   in physical units for both state and past actions.
5. Run the forward pass with the blob's stored `activation`. Unstandardise
   the output with `y_std` / `y_mean`.
6. Output is a vector of physical commands in `output_layout` order:
   m/s for translational channels, deg/s for the rotation channel. Do
   **not** assume `[-1, 1]`.
7. Clamp per-channel to the physical scales in Section 3 before sending
   to motor controllers. The simulator clamps at these limits too, so the
   training distribution is bounded by them; ignoring the clamp can
   produce out-of-distribution inputs on subsequent steps.
8. Close the loop: the action just issued becomes the next step's
   "previous action" in the history buffer. Use the physical command that
   was actually sent to motors (after clamping), not any upstream target.

If any of (1)–(8) is skipped, the symptom is typically a large magnitude
or sign mismatch versus the simulator — even though the network weights
are identical.

---

## 8. Invariants to preserve

These are the cross-file contracts that the consistency audit in this pass
validated. Breaking any of them will silently corrupt training or inference:

- `STATE_INPUT_DIM = 12` everywhere (`train_mlp.py`, `headless_sim.py`,
  `simulator.py`, `benchmark_gui.py`).
- `_action_keys_for_mode` / `_command_keys_for_mode` produce identical
  lists for identical modes (they do — the three implementations are
  byte-identical strings).
- `_PHYSICAL_SCALES` in `headless_sim.py` must match the magnitudes used
  by `Robot.apply_normalized_action`.
- `_ACTION_FEATURE_ORDER` exists solely to re-map `heading_drive` from
  the physical-action order `[rotation_rate, drive_speed]` to the trainer
  order `[drive_speed, rotation_rate]`. The other two modes use identity
  permutations.
- The formula `history = (input_dim + A) // (STATE_DIM + A)` is repeated
  in `train_mlp._infer_memory_steps`, `benchmark_gui._history_from_dim`,
  and `simulator._current_model_memory_len`. All three must agree.
- The inference forward pass must use the activation stored in the blob
  (`relu` / `tanh` / `leaky_relu`).
- Zero-padding policy (value = 0 for every field) must be identical at
  training time and at inference time.
