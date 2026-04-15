# RoboSim MLP — ROS Deployment Spec

This document is for the engineer implementing the ROS node that runs a
trained RoboSim MLP model on the real robot. It is the authoritative
reference for the runtime layer: sidecar parameters, input assembly,
inference, stiction snap, hysteresis, and motor publishing.

For the pure inference math (weight matrices, activation selection,
normalization), the companion file [MODEL_SPEC.md](MODEL_SPEC.md) is
authoritative. This document cross-references it where relevant.

---

## 1. Files produced by training

Every completed training run writes three files into the output directory:

| File | Purpose |
|---|---|
| `model_<name>_<mode>.json` | Weights, biases, activation, normalization stats, input/output layout. Consumed by the inference path. |
| `metrics_<name>_<mode>.json` | Training curves + config. Informational only. |
| `params_<name>_<mode>.json` | **Deployment parameters**. Consumed by the ROS node at startup. This file. |

`<mode>` is one of `heading`, `strafe`, `heading_strafe`.

Load both the model JSON and the params JSON at node startup. The two
files are paired — do not mix a model with a params sidecar from a
different training run.

---

## 2. `params_*.json` schema

```jsonc
{
  "saved_at": "2026-04-14T15:32:10",
  "schema_version": 1,
  "mode": "heading_drive",
  "mode_suffix": "heading",
  "model_file": "model_robosim_v1_heading.json",
  "memory_steps": 5,
  "input_dim": 68,
  "output_dim": 2,
  "log_rate_hz": 10.0,
  "activation": "relu",

  "whisker": {
    "count": 11,
    "max_length_m": 0.50,
    "angles_deg_left_to_right": [90, 60, 45, 30, 15, 0, -15, -30, -45, -60, -90]
  },

  "input_signals": {
    "per_timestep_state": [
      { "name": "whisker_lengths",    "unit": "m",   "count": 11, "range": [0.0, 0.50] },
      { "name": "heading_to_target",  "unit": "deg", "sign": "+ = target to robot LEFT" }
    ],
    "past_action_slice_order": ["drive_speed", "rotation_rate"]
  },

  "output_signals": [
    { "name": "drive_speed",   "unit": "m/s"   },
    { "name": "rotation_rate", "unit": "deg/s" }
  ],

  "sign_conventions": {
    "frame": "right-hand, z-up",
    "rotation_rate":     "+ = CCW (turn LEFT)",
    "drive_speed":       "+ = forward (robot x-axis)",
    "vx":                "+ = forward (robot x-axis)",
    "vy":                "+ = robot LEFT (robot y-axis)",
    "heading_to_target": "+ = target on robot LEFT"
  },

  "saturation_limits": {
    "rotation_rate_dps": 40.0,
    "drive_speed_mps":    0.40,
    "vx_mps":             0.40,
    "vy_mps":             0.40
  },

  "stiction": {
    "min_turn_rate_dps":  30.0,
    "inner_deadband_dps": 3.0,
    "notes": "..."
  },

  "recommended_hysteresis": {
    "rotation_rate_enter_dps": 30.0,
    "rotation_rate_exit_dps":  18.0
  },

  "source_metadata_counters": { "...": "distinct values observed across training logs" }
}
```

Any field can be overridden at the ROS-node parameter server level, but
the defaults in the sidecar match what the model was trained against and
should be preferred unless there is a specific reason to diverge.

If `stiction.min_turn_rate_dps == 0.0`, the training data was collected
without stiction shaping. The model will still emit small rates that die
on the real motors — in that case you **must** choose reasonable values
manually (`30.0` / `3.0` is a sane starting point for most hobby
brushed-DC platforms).

---

## 3. Input assembly

Build the feature vector exactly as specified in
[MODEL_SPEC.md §3](MODEL_SPEC.md). Summary:

- `N = memory_steps`, `A` = action dim (2 or 3 depending on mode).
- Oldest-first history window of `N` entries. Each entry = 11 whisker
  lengths (m) + `heading_to_target` (deg). For all non-final entries,
  also append the past action slice in the order listed under
  `input_signals.past_action_slice_order` in the sidecar.
- The final (current) entry gets no action appended.
- `len(feature_vector) == input_dim == N*12 + (N-1)*A`.
- **Pad the deque on the left at episode start with copies of the first
  real observation's state and a zero action** — see the runtime loop
  below. Do **not** zero-fill the whiskers (`whiskers=0.0` would mean
  "obstacle at distance 0 in every direction") and do **not** use `0.5`
  (max whisker length, also out of distribution).

Runtime loop:

```
on episode reset / startup:
    load model.json, params.json (once)
    read sensors -> initial_state = {whisker_lengths, heading_to_target}
    pad_entry = {state: initial_state, action: zero_action}
    deque = [pad_entry] * (N - 1)
    deque.append({state: initial_state, action: zero_action})
    last_cmd = zero_action_for_mode

each control tick (at log_rate_hz):
    1. read sensors -> current_state = {whisker_lengths, heading_to_target}
    2. deque.append({state: current_state, action: last_cmd})
       if len(deque) > N: deque.popleft()
    3. feature_vector = flatten(deque)
    4. y_phys = infer(feature_vector)              # MODEL_SPEC §5
    5. y_phys = clamp_to_saturation_limits(y_phys)
    6. y_phys.rotation = apply_rotation_pipeline(  # §5 below
           y_phys.rotation, min_dps, inner_db, exit_dps, state)
    7. publish_cmd(y_phys)
    8. last_cmd = y_phys                           # store post-processed command
```

The action stored in `last_cmd` is the command **actually sent after all
post-processing** (clamp + stiction + hysteresis), not the raw model
output. This matches how the simulator builds training data.

Tick at `log_rate_hz` (typically 10 Hz). Inference at a higher rate than
the training cadence shifts the input distribution out of what the model
was trained on.

---

## 4. Inference

Covered fully in [MODEL_SPEC.md §5](MODEL_SPEC.md). Key points:

- Output is in **physical units** after un-normalization, **not**
  `[-1, 1]`. Do not apply `tanh` or normalize outputs.
- Read `activation` from the model blob. Do not hardcode ReLU — models
  may be trained with `relu`, `tanh`, or `leaky_relu` (α=0.01).
- Clamp every channel to the limits in `saturation_limits` before any
  downstream post-processing.

---

## 5. Steering pipeline — combined dual-band + hysteresis (rotation only)

**Single function**, applied after the saturation clamp. Replaces the
earlier-spec layout that ran a stateless dual-band snap *before* a
separate hysteresis pass — that ordering is broken because the dual-band
snaps every sub-min input up to ±min, leaving the feather band
unreachable.

The combined pipeline has two regimes selected by a one-bit
`was_turning` state, seeded `False` on episode reset:

```
not_turning:                              | turning:
  |rate| <  inner_db          -> 0        |   |rate| <  exit_dps         -> 0  (drop turn)
  inner_db <= |rate| < min    -> ±min     |   |rate| >= exit_dps         -> rate (feather + passthrough)
  |rate| >= min               -> rate     |
  (any nonzero output sets was_turning)   |
```

So once a turn is established by crossing `min`, the rate is allowed to
**feather down** to `exit_dps` without snapping back up. Below `exit_dps`,
the turn is dropped. Stiction protection on entry remains: a fresh,
sub-min command still snaps up to overcome static friction.

```python
def apply_rotation_pipeline(rate_dps, min_dps, inner_db, exit_dps, state):
    if min_dps <= 0.0:
        return rate_dps                             # disabled
    mag = abs(rate_dps)
    if state.was_turning:
        if mag < exit_dps:
            state.was_turning = False
            return 0.0
        return rate_dps                             # feather or passthrough
    # not turning
    if mag < inner_db or mag == 0.0:
        return 0.0
    state.was_turning = True
    if mag < min_dps:
        return math.copysign(min_dps, rate_dps)
    return rate_dps
```

This matches `Robot.apply_rotation_pipeline` in `sim_core.py` exactly —
the simulator runs the same function in model-driven episodes, so what
you see in the sim is what you get on the real robot.

### Parameters

All four come from the params sidecar:

| Field | Source | Default if 0 |
|---|---|---|
| `min_dps` | `params.stiction.min_turn_rate_dps` | feature disabled |
| `inner_db` | `params.stiction.inner_deadband_dps` | 0 (no deep dead zone) |
| `exit_dps` | `params.recommended_hysteresis.rotation_rate_exit_dps` | `0.6 × min_dps` |
| (informational) | `params.recommended_hysteresis.hysteresis_band_dps` | derived = `min - exit` |

`exit_dps` MUST be `< min_dps` and `> 0`. A reasonable feather window is
`min - exit` somewhere in `[0.10 × min, 0.50 × min]` — i.e., 10–50% of
the min turn rate. Too narrow → chatter returns. Too wide → the robot
coasts on an unwanted rotation past the goal heading.

### State management

- `was_turning` is **per-axis** (just rotation here, but if you add the
  same pipeline to translation channels it needs its own bit).
- Reset to `False` on every episode start, scenario change, or any time
  the policy is paused/resumed.
- Persist the bit across control ticks; it's the only stateful element
  in the inference path.

### Why this stays out of training data

The training pipeline (in the simulator, when a human pilots with
keyboard or gamepad) applies only the **stateless** dual-band snap, not
this combined pipeline. The reason: hysteresis introduces a hidden
dependence on the prior command, but the model's input vector has no
"was_turning" feature. If we let the bit influence labels, the model
would have to guess at hidden state from current observations alone —
worse than just letting it emit raw rates and post-processing at
runtime. Apply this combined pipeline only at deployment (and for
model-driven preview in the simulator).

Translation speeds (`drive_speed`, `vx`, `vy`) are not stiction-shaped
in the current pipeline; brushless and well-geared brushed platforms
typically don't need it. If yours does, add a symmetric pipeline keyed
off separate sidecar fields.

---

## 7. Publishing to motors

Publish whatever message type your motor driver expects. If it's
`geometry_msgs/Twist`:

| Twist field | Source | Unit conversion |
|---|---|---|
| `linear.x`  | `drive_speed` (heading_drive) or `vx` (strafe/heading_strafe) | m/s, no change |
| `linear.y`  | 0 (heading_drive) or `vy` (strafe/heading_strafe) | m/s, no change. **Sign: +y = left**, matches ROS REP-103 |
| `angular.z` | `rotation_rate` | **deg/s → rad/s**: multiply by `π/180`. **Sign: +z = CCW = turn left**, matches ROS REP-103 |

ROS REP-103 body frame (x-forward, y-left, z-up) matches RoboSim's
right-hand convention exactly, so no sign flips are required. If your
motor driver uses a different convention (e.g. +z = CW), invert
`angular.z` at the boundary — **not** earlier in the pipeline.

---

## 8. Startup checklist

- [ ] Loaded `model_*.json` AND matching `params_*.json` from the same
  training run (check `saved_at` timestamps match).
- [ ] `params.memory_steps == (model.input_dim + A) / (12 + A)` for the
  mode's action dim `A`. If not, the files don't match.
- [ ] Control loop ticks at `params.log_rate_hz`.
- [ ] History deque pre-seeded with `(memory_steps − 1)` **copies of the first real observation's state with zero action** — NOT literal whiskers=0 (which would mean "obstacles everywhere").
- [ ] `last_cmd` initialized to zero action for the mode.
- [ ] Inference output treated as physical units (no `tanh`, no rescale).
- [ ] Activation string read from model blob, not hardcoded.
- [ ] Saturation clamp applied first, then the **single combined**
  `apply_rotation_pipeline` (§5), then publish. Do NOT chain a separate
  stiction snap and hysteresis pass — the dual-band snap would prevent
  the feather band from ever being reached.
- [ ] `was_turning` state initialized False; reset to False on every
  episode start.
- [ ] `last_cmd` stores the **post-processed** command.
- [ ] All sign conventions cross-checked against
  `params.sign_conventions` — especially `heading_to_target` input sign.
- [ ] ROS Twist `angular.z` converted from deg/s → rad/s.

---

## 9. Failure modes observed during development

| Symptom | Likely cause |
|---|---|
| Robot refuses to turn, but logs show nonzero `rotation_rate` commands | Stiction not applied, or `min_turn_rate_dps` set below motor's actual stiction threshold. |
| Robot chatters / ratchets near the threshold | Hysteresis not applied, or `exit_dps` too close to `enter_dps`. |
| Robot turns the wrong way | Sign mismatch. Check `angular.z` conversion and any upstream `heading_to_target` sign reversal. |
| Outputs are ~10× smaller than expected | Output treated as normalized `[-1, 1]` — forgot to un-normalize. |
| Turns initiated but robot moves in a small circle forever | `last_cmd` not updated, so the memory window encodes a stale action state. |
| Behavior diverges from simulator after 1–2 seconds | Control loop rate does not match `log_rate_hz`, so input distribution is off. |

Each of these has been seen in practice. If the deployed behavior
diverges from interactive simulator behavior, work the list top-down
before re-training.
