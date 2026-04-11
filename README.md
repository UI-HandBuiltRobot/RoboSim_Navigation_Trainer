# RoboSim Navigation Trainer

RoboSim Navigation Trainer supports an imitation-learning workflow for robot navigation with three core utilities:

- simulator.py for interactive control, data collection, and model-in-the-loop testing
- train_mlp.py for training JSON policy models from simulator logs
- benchmark_gui.py for repeatable headless model evaluation on shared maps

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch tools directly:

```bash
python simulator.py
python train_mlp.py
python benchmark_gui.py
```

Or use the utility launcher:

```bash
python utility_launcher.py
```

## Simulator

The simulator is the primary interactive workspace. You can drive manually, log demonstrations, load trained models, and build targeted DAgger queues for corrective data collection.

### Core options

- Run: starts a new randomized episode, or loads the next queued train-me case when available
- FAIL: while in model-driving mode, snapshots the current case for later human takeover and replay
- Obstacles (-/+): controls environment density per episode
- Enable Logging: toggles whether demonstration samples are written to disk
- History (-/+): sets temporal window length recorded in each sample
- Rate Hz (-/+): sets active logging frequency
- Pick / Reload: selects or reloads the model for the current control mode
- Robot View: toggles robot-aligned framing for manual driving comfort

### Control modes

- Heading+Drive: predicts translational speed and rotation rate
- XY Strafe: predicts planar x/y velocity
- Hdg+Strafe: predicts rotation plus planar x/y velocity

Train and evaluate models in the same mode they were trained for.

### Rendering modes and why they exist

The simulator includes multiple obstacle rendering modes to support different operator tasks:

- All: maximizes full-scene awareness for debugging map generation and geometry edge cases
- Action Radius: emphasizes the local decision zone around the robot and reduces visual clutter
- Detected: displays obstacles as they are sensed, useful for understanding whisker-triggered perception timing
- Sparse Sensing: shows sensed points over short temporal memory, helpful for diagnosing history-window effects and perception-to-action coupling

These modes change visualization only, not underlying physics.

### Model-driven episodes and DAgger workflow

You can load a trained JSON model and let it drive inside the interactive simulator. This is useful for:

- quick qualitative checks of behavior
- spotting failure patterns in context
- queuing difficult scenes into the train-me flow for human corrective demonstrations

That loop is an interactive DAgger-style process: observe a model failure, queue the case, then collect human takeover data on that exact scenario.

This differs from benchmarking in benchmark_gui.py:

- simulator.py is interactive and diagnostic, optimized for discovery and corrective data collection
- benchmark_gui.py is statistical and batch-oriented, optimized for apples-to-apples model comparison

### Output directories

- logs/ stores demonstration logs
- dagger_snapshots/ stores queued train-me scenarios
- trained_models/ stores trained model and metrics artifacts

## Model Training

Use train_mlp.py to train models from simulator logs.

### Typical flow

1. Select one or more JSONL logs from logs/
2. Configure architecture and optimization settings
3. Start training and monitor training/validation behavior
4. Save model JSON and metrics JSON into trained_models/

### Hyperparameter tips

- Hidden width and depth:
	- start modest (for example 2 to 3 layers with moderate width) to reduce overfitting on small datasets
	- increase capacity only if both train and validation curves underfit
- Learning rate:
	- if loss is noisy or diverges, reduce learning rate first
	- if convergence is extremely slow and stable, increase cautiously
- Batch size:
	- larger batches improve gradient stability but can smooth away useful variation
	- smaller batches can generalize better but may require lower learning rates
- Epochs:
	- track validation trend and stop when improvement plateaus
	- avoid training far past the best validation checkpoint
- History window alignment:
	- ensure the history length used in logging aligns with what you expect the model to consume
	- changing memory length changes input structure and affects behavior significantly
- Data balance:
	- include successful trajectories and failure-recovery demonstrations
	- overrepresenting easy maps can produce brittle models

## Model Evaluation

Evaluation can be done two ways: interactive in simulator.py and batch/statistical in benchmark_gui.py.

### Interactive evaluation in simulator.py

Load a model and watch it drive in live randomized scenes. This is best for:

- understanding behavior and failure modes in context
- identifying specific corrective demonstrations to collect
- validating whether control feel is acceptable for deployment intent

### Batch evaluation in benchmark_gui.py

benchmark_gui.py is designed for reproducible comparison across models.

1. Generate a benchmark map set once
2. Run one model over all maps in headless simulation
3. Save report metrics
4. Repeat with another model on the same map set

Maps are scenario-only and control-mode agnostic: robot pose, obstacles, and goal placement are fixed regardless of model mode. This supports fair direct comparison because each candidate model is tested against identical environments.

### Why headless benchmark evaluation matters

- repeatability: identical seeds and maps across runs
- comparability: success rate, time, and collision metrics are measured on the same workload
- throughput: many episodes can be evaluated quickly without manual intervention
- decision support: helps choose between model versions before interactive rollout