# RoboSim Navigation Trainer

RoboSim Navigation Trainer is an imitation-learning framework for obstacle-aware
robot navigation. It pairs an interactive 2D simulator with a NumPy MLP trainer
and a headless benchmark evaluator, so you can collect demonstrations, train a
policy, benchmark it, and close the loop with DAgger-style corrective data.

Looking for a guided walkthrough? See [QUICKSTART.md](QUICKSTART.md).

## Installation

```bash
pip install -r requirements.txt
```

The trainer and evaluator are pure NumPy. The simulator uses `pygame` for
rendering; the trainer and benchmark GUIs use `tkinter` (bundled with most
Python builds). A gamepad is optional but recommended for demonstrations.

## Launcher

Start here. The launcher is a small Tkinter window that opens any of the three
main utilities in its own Python process, so they can run side-by-side.

```bash
python utility_launcher.py
```

Buttons:

- **Open Simulator** — starts [simulator.py](simulator.py) for interactive
  driving, demonstration logging, and model-in-the-loop testing.
- **Open MLP Trainer** — starts [train_mlp.py](train_mlp.py) for training a
  policy from one or more JSONL logs.
- **Open Benchmark Utility** — starts [benchmark_gui.py](benchmark_gui.py) for
  generating fixed map sets and running batch model evaluations.

Each button spawns an independent process rooted at the project directory, so
you can keep the launcher open while any combination of tools runs.

If you prefer the command line, each utility can be invoked directly:

```bash
python simulator.py
python train_mlp.py
python benchmark_gui.py
```

---

## Simulator — [simulator.py](simulator.py)

The simulator is the primary interactive workspace. You can drive manually via
keyboard or gamepad, log demonstrations, load trained models, and build
targeted DAgger queues for corrective data collection.

### Top-bar controls

- **Run** — starts a fresh randomized episode, or replays the next queued
  DAgger case when the queue is non-empty.
- **FAIL** — while a model is driving, snapshots the current scenario and
  appends it to the DAgger queue for later human takeover.
- **Obstacles −/+** — changes the number of obstacles generated per episode.
- **Enable Logging** — toggles whether demonstration samples are written to
  disk during manual driving.
- **Control Mode** — switches between Heading+Drive, XY Strafe, and
  Heading+Strafe.
- **Vis Mode** — cycles through rendering styles (see below).
- **Pick / Reload** — selects or reloads the JSON model used in model-driven
  episodes for the active control mode.
- **Robot View** — toggles a robot-aligned camera framing for comfortable
  manual driving.

### Preferences panel

Opens a sub-panel with the settings that affect logged-sample shape:

- **History (N)** — number of timesteps in each sample's history window.
  Determines `memory_steps` and therefore model input dimensionality.
- **Rate Hz** — effective logging frequency in Hz.

Both values are recorded in every log sample so the trainer can verify
consistency across input files.

### Control modes

Choose before driving; a model trained in one mode cannot be used in another.

- **Heading + Drive** — outputs `[drive_speed, rotation_rate]`
  (forward speed in m/s + yaw rate in deg/s).
- **XY Strafe** — outputs `[vx, vy]`
  (planar body-frame velocity in m/s).
- **Heading + Strafe** — outputs `[rotation_rate, vx, vy]`
  (yaw rate + planar body-frame velocity).

Sign convention is right-hand, z-up: positive rotation rate turns left,
positive `vy` moves left, positive `heading_to_target` means the target is
on the robot's left. See [SPEC.md](SPEC.md) and
[MODEL_SPEC.md](MODEL_SPEC.md) for the full data-flow specification.

### Visualization modes

Rendering only; underlying physics and sensors are identical in every mode.

- **All** — full-scene view for debugging map generation and geometry.
- **Action Radius** — emphasizes the robot's local decision zone and reduces
  clutter on dense maps.
- **Detected** — shows obstacles as the whiskers sense them, useful for
  inspecting perception timing.
- **Sparse Sensing** — hides walls and undetected obstacles; only whisker
  intersection points are visible. Best for judging what the policy actually
  "sees" through the memory window.

### Model-driven episodes and the DAgger loop

Load a JSON model with **Pick** and select a model-driven run from the
control menu. The model then issues commands each step while the simulator
renders the result. This is where DAgger happens:

1. Watch the model drive.
2. When it makes a mistake, hit **FAIL** (or let it crash) — the scenario is
   frozen and queued in `dagger_snapshots/`.
3. After collecting some queued cases, press **Run** with Enable Logging on.
   The simulator will replay each queued case and hand control to you so you
   can demonstrate the correct recovery.
4. Retrain, adding the new logs to your dataset.

This is complementary to batch benchmarking:

- Simulator is interactive and diagnostic — good for discovering failures and
  collecting corrective data.
- Benchmark is statistical and batch-oriented — good for apples-to-apples
  model comparison.

### Output directories

- `logs/` — demonstration logs (JSONL, schema v2).
- `dagger_snapshots/` — queued failure scenarios for takeover runs.
- `trained_models/` — consumed by the simulator when loading models.
- `benchmark_maps/` and `benchmark_reports/` — produced by the evaluator.

---

## MLP Trainer — [train_mlp.py](train_mlp.py)

Trains a small NumPy MLP from one or more simulator JSONL logs. No PyTorch
required. The trained model is a self-contained JSON file that encodes
weights, biases, per-feature normalization, and metadata.

### Typical flow

1. Launch the trainer.
2. Add one or more JSONL logs from `logs/`. The trainer validates that all
   inputs share the same control mode, history length, and sample rate.
3. Configure architecture, optimizer, and training schedule.
4. Set **Model Name** and confirm the **Full Path** preview.
5. Click **Train**. A live loss curve updates each epoch.
6. The trainer writes `<name>_<mode>.json` (model) and `<name>_<mode>_metrics.json`
   (training metrics) into the Save Dir.

### Key configuration

- **Control mode** — must match the logs. Training halts if they mismatch.
- **Hidden layers / Hidden width** — network size. Start small.
- **Activation** — `relu`, `tanh`, or `leaky_relu` (alpha 0.01). The chosen
  activation is stored in the model JSON and applied by every downstream
  consumer (simulator, benchmark GUI, real-robot runtime).
- **Learning rate, Batch size, Epochs** — standard optimizer knobs.
- **Validation split** — fraction held out for the plotted validation curve.
- **Model Name** — filename stem (sanitized). The trainer appends the mode
  suffix (`_heading`, `_strafe`, `_heading_strafe`) automatically.
- **Save Dir** — output folder; the preview label shows the exact resolved
  path that will be written.

### Outputs

- `<name>_<mode>.json` — the model blob. Top-level fields include `mode`,
  `input_dim`, `output_dim`, `memory_steps`, `activation`, `weights`,
  `biases`, and per-channel normalization stats. Full schema in
  [MODEL_SPEC.md](MODEL_SPEC.md).
- `<name>_<mode>_metrics.json` — loss history, validation metrics, and
  training configuration.

### Notes

- Normalization stats (`x_mean`, `x_std`, `y_mean`, `y_std`) are computed from
  the training split only and baked into the model JSON. Downstream
  consumers never need to recompute them.
- Outputs are in **physical units** (m/s, deg/s) after un-normalization, not
  the `[-1, 1]` range seen during training. Clamp them at the motor boundary
  (±0.40 m/s, ±40 deg/s).

---

## Benchmark Evaluator — [benchmark_gui.py](benchmark_gui.py)

Headless, batch-oriented model evaluation against a fixed map set. Designed
for apples-to-apples comparison between model versions.

### Typical flow

1. Launch the benchmark GUI.
2. **Generate Maps** — produces a reproducible set of scenario JSON files
   under `benchmark_maps/`. Maps are control-mode agnostic: robot pose,
   obstacles, and goal placement are fixed regardless of which model you
   evaluate later.
3. **Load Model** — pick a trained JSON model.
4. **Run** — the simulator steps each map headlessly under the model's
   control. Live metrics update as runs complete.
5. **Save Report** — writes a summary into `benchmark_reports/` with success
   rate, time-to-goal, collision counts, and per-map outcomes.

### Why headless benchmark evaluation matters

- **Repeatability** — identical seeds and maps across runs.
- **Comparability** — every model is scored against the same workload.
- **Throughput** — hundreds of episodes evaluate in seconds.
- **Decision support** — pick the winner before an interactive rollout.

Use interactive evaluation in the simulator for qualitative judgment and
DAgger data collection; use the benchmark GUI for quantitative model
selection.

---

## Further reading

- [SPEC.md](SPEC.md) — authoritative data-flow specification (coordinates,
  sensors, log schema, trainer pipeline, invariants).
- [MODEL_SPEC.md](MODEL_SPEC.md) — standalone implementation spec for porting
  the inference path to another runtime (e.g., a real robot).
