# Quick Start

This walkthrough takes you from a fresh clone to a trained, benchmarked, and
DAgger-refined model in one sitting. Plan on about 30–60 minutes the first
time through.

See [README.md](README.md) for the reference documentation of each tool.

---

## 0. Install

```bash
pip install -r requirements.txt
```

Plug in a gamepad if you have one. Keyboard works too, but gamepad
demonstrations are smoother and produce better training data.

---

## 1. Open the simulator

From the project root:

```bash
python utility_launcher.py
```

Click **Open Simulator**.

---

## 2. Configure memory and rendering

Open the **Prefs** panel on the simulator toolbar and set:

- **History (N)** = `5`
- **Vis Mode** = `Sparse Sensing`

`History=5` gives the policy enough temporal context to judge motion
without blowing up input dimensionality. `Sparse Sensing` hides walls and
undetected obstacles, so you see exactly what the policy sees — this is
invaluable when judging whether a behavior is a sensing problem or a
policy problem.

---

## 3. Set obstacle density

Use **Obstacles +** until the counter reads `20`. This is a reasonable
mid-difficulty map for training data collection.

---

## 4. Drive and log

1. Confirm the control mode is **Heading + Drive** (the most common starting
   point).
2. Toggle **Enable Logging** on.
3. Press **Run** to spawn a fresh episode.
4. Drive the robot to the goal with the gamepad. The left stick drives,
   the right stick (or the twist axis) yaws.
5. When an episode ends, press **Run** again for the next one.

Aim for around 20 clean runs to start. Each run is appended to a JSONL file
in `logs/`.

---

## 5. Train a model

1. From the launcher, click **Open MLP Trainer**.
2. **Add Logs** → select the JSONL files you just recorded.
3. Confirm the mode auto-detects as `heading_drive`.
4. Choose a sensible architecture (2–3 hidden layers, ~64–128 units wide,
   `relu` activation is a fine default).
5. Set **Model Name** — e.g. `robosim_v1` — and check the **Full Path**
   preview.
6. Click **Train**.

When training finishes, you will have
`trained_models/robosim_v1_heading.json` and a matching metrics file.

---

## 6. Build a benchmark map set

1. From the launcher, click **Open Benchmark Utility**.
2. Click **Generate Maps**. This writes a fixed, reproducible set of
   scenarios into `benchmark_maps/`.

Keep this map set around — future models should be benchmarked against the
same set so their scores are directly comparable.

---

## 7. Evaluate the model

In the benchmark GUI:

1. **Load Model** → pick `robosim_v1_heading.json`.
2. **Run** — each map is executed headlessly. Success rate, time-to-goal,
   and collision counts populate as runs complete.
3. **Save Report** — writes the numbers into `benchmark_reports/`.

---

## 8. Watch the model drive

Back in the simulator:

1. Set the control mode to **Heading + Drive**.
2. **Pick** → `robosim_v1_heading.json`.
3. Switch the control source to model-driven and press **Run**.

Observe the behavior. Success on the benchmark set does not guarantee
good behavior on arbitrary new layouts, and this is where qualitative
judgment matters.

---

## 9. DAgger: fail, queue, replay

When the model eventually misbehaves:

- Press **FAIL** the moment it makes a mistake, **or** let it crash. Either
  way, the scenario is snapshotted into the DAgger queue
  (`dagger_snapshots/`).
- After queuing a handful of cases, toggle **Enable Logging** back on and
  press **Run**. The simulator replays each queued scenario with you in
  control — demonstrate the correct recovery.
- When the queue is empty, retrain the MLP with the combined
  original + DAgger logs and re-benchmark.

This is the core imitation-learning loop: drive, train, benchmark, replay
the failures, retrain.

---

## Tips

- **Start around 20 runs, then let DAgger stress-test the model.** A small
  clean baseline is usually enough to produce a reasonable first policy.
  The DAgger queue is a much more efficient source of difficult training
  data than pure random exploration.
- **Periodically evaluate against the same benchmark map set.** Regenerating
  maps invalidates all prior comparisons. Keep one canonical map set per
  project so scores across model versions are directly comparable.
- **Heading control mode is the most common.** If your controller
  implements strafing, try the Heading+Strafe model — but note that whether
  it beats pure Heading depends strongly on the training data. A strafe
  policy trained from hesitant demonstrations will often underperform a
  clean heading policy.
- **You must record training data separately for each model type.** A
  Heading+Drive log cannot be used to train a strafe model, and vice
  versa. Do a dedicated collection session in the mode you intend to
  deploy.
