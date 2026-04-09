# RoboSim Navigation Trainer

RoboSim Navigation Trainer is a desktop toolkit for collecting navigation behavior data, training control models, and calibrating whisker geometry from camera images.

It includes three user-facing applications:

- `simulator.py` - interactive simulator and data collection app
- `train_mlp.py` - model trainer for simulator logs
- `whisker_calibration_tool.py` - camera/image calibration GUI

## Quick Start

1. Install Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app you need:

```bash
python simulator.py
python train_mlp.py
python whisker_calibration_tool.py
```

## Simulator

The simulator is where you generate episodes, collect logs, run trained models, and queue difficult cases for human takeover training.

### What You Do In The Simulator

- Configure control mode and input source
- Run episodes manually (keyboard/gamepad) or with a trained model
- Save successful runs as training logs
- Queue failed model runs for train-me replay

### Top Controls and User Impact

These controls now include hover data tips in the UI.

| Control | What It Does | Likely Effect On Trained Models |
|---|---|---|
| Run | Starts a fresh episode, or loads the next queued train-me case | More run variety improves coverage and generalization |
| FAIL | In Model input mode, queues the current case for human takeover | Helps target model weaknesses with focused corrective examples |
| Obstacles (- / +) | Decrease/increase obstacle count in generated episodes | More obstacles typically produce stronger obstacle-avoidance behavior, but can make training distribution narrower if overused |
| History (- / +) | Change logging memory window (1-10) | Longer history can improve smoothness and anticipation; too short may react late |
| Rate Hz (- / +) | Change active logging sample rate | Higher rate captures finer motion detail; lower rate gives lighter, coarser datasets |
| Control Mode: Heading+Drive | Heading + forward/back command style | Produces models that output drive speed and turn rate |
| Control Mode: XY Strafe | Translation-only command style | Produces models that output planar velocity components |
| Control Mode: Hdg+Strafe | Heading + planar strafe command style | Produces models with full planar control outputs |
| Input Source: Keyboard | Manual control from keys | Human demonstrations become supervision data for training |
| Input Source: Gamepad | Manual control from joystick | Can capture smoother demonstrations than keyboard in many cases |
| Input Source: Model | Uses the selected trained model to control robot | Useful for validation and for collecting failure cases |
| Pick | Choose a specific model JSON for current mode | Lets you compare model versions directly |
| Reload | Reload selected model or latest model for current mode | Fast iteration after retraining |
| Enable Logging | Turns training log capture on/off | Disabled means no new training samples are saved |
| Robot View | Toggles robot-aligned camera view | Improves operator orientation during manual collection |

### Simulator Panel (Right Side)

The panel shows current model status, queue status, drive command values, whisker lengths, heading-to-target, collision flag, and robot pose. Use it as a live quality check while collecting demonstrations.

### Data Output From Simulator

- Training logs: `logs/`
- Train-me snapshots: `dagger_snapshots/`
- Optional model files loaded from: `trained_models/`

## Model Trainer

The trainer GUI reads one or more simulator log files and trains mode-specific models.

### Typical Workflow

1. Click **Select Log Files** and choose `.jsonl` logs.
2. Set model settings (hidden width/layers, epochs, batch size, learning rate, activation).
3. Choose output directory.
4. Click **Train Models**.
5. Watch progress and final metrics in the Status panel.

### What The Trainer Produces

For each mode with available samples, the trainer saves:

- `model_<mode>_<index>.json` - deployable model artifact
- `metrics_<mode>_<index>.json` - training/validation loss history and summary

Saved by default under `trained_models/`.

### End-User Notes

- If a mode has no samples in selected logs, that mode is skipped.
- Mixed-quality logs can degrade results; curate runs when possible.
- Matching simulator control mode and model mode is important when testing.

## Whisker Calibration GUI

This app helps you calibrate whisker rays against camera imagery so distance mapping is grounded to your setup.

### Typical Workflow

1. Connect to a camera (1920x1080 check is built in), or load an existing calibration.
2. Snap an image.
3. Select a whisker angle button.
4. Left-click points on the image and enter known distance (mm) for each point.
5. Repeat until each whisker has enough points.
6. Click **Done** for the active whisker.
7. Save calibration JSON.

### Practical Tips

- Use clean, consistent measurement references.
- Add points across near/mid/far distances for each whisker.
- Re-check whiskers with sparse or noisy points before saving.

## Files and Folders

- `simulator.py` - simulator and data collection
- `train_mlp.py` - model training GUI
- `whisker_calibration_tool.py` - whisker calibration GUI
- `logs/` - captured run logs
- `trained_models/` - trained model artifacts and metrics
- `dagger_snapshots/` - queued train-me snapshots

## Troubleshooting

- No logs being saved: verify **Enable Logging** is checked and runs are reaching the goal.
- Model not loading: check that selected model matches current control mode.
- No camera found in calibration app: verify camera availability and that no other app is locking it.
- UI controls crowded at small window sizes: increase the app window size.

## License / Usage

Use this project for experimentation, data collection, and model iteration in your own navigation workflows.