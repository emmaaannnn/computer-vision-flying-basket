# Computer Vision Flying Basket

This repository is the local computer-vision foundation for the Flying Basket honours thesis project. The immediate goal is to validate the vision pipeline on a laptop camera before replacing the input source with the DJI Tello video stream.

The first development milestone is a local webcam loop that:

- captures frames from the laptop camera;
- applies the planned YUV plus CLAHE pre-processing stage;
- runs person detection for the supervisor target;
- estimates geometric depth from bounding-box height;
- overlays diagnostics so the pipeline can be inspected in real time.

## Python environment

This project uses a local `.venv` virtual environment for all `pip` installs and runtime commands.

### Activate the virtual environment

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

When the environment is active, your shell prompt should show `(.venv)`.

### Install dependencies into `.venv`

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

1. Activate `.venv`.
2. Install the dependencies.
3. Run the local camera pipeline:

```bash
python -m src.main --camera-index 0
```

If your default webcam is not camera `0`, try another index such as `1`.

For a camera and pre-processing smoke test without YOLO inference, run:

```bash
python -m src.main --camera-index 0 --disable-inference --show-preprocessed
```

For person detection, use the default command above. The first inference-enabled run may download the baseline Ultralytics model weights automatically.

To calibrate the geometric distance constant against a measured position, stand at a known distance and run:

```bash
python -m src.main --camera-index 0 --reference-distance-m 1.5
```

When the person box is stable, note the on-screen `scale suggest` value and reuse it in later runs with `--geom-scale`.

## Calibration note

The current geometric ranging constant is an initial placeholder intended for laptop validation only. Before using the output for any control logic, calibrate `--geom-scale` against a measured subject distance and the observed bounding-box height.

One practical workflow is:

1. Mark a measured distance such as `1.5 m` from the laptop camera.
2. Run `python -m src.main --camera-index 0 --reference-distance-m 1.5`.
3. Hold a steady pose until the `scale suggest` overlay settles.
4. Re-run with that suggested value via `--geom-scale` and verify the reported depth at `1.0 m`, `1.5 m`, and `2.0 m`.

## Planned pipeline structure

- `src/camera.py`: camera ingestion abstraction for local webcam testing.
- `src/preprocessing.py`: YUV conversion and CLAHE enhancement.
- `src/detector.py`: person detection wrapper for YOLO inference.
- `src/ranging.py`: O(1) geometric depth estimation from bounding-box height.
- `src/pipeline.py`: orchestration and on-screen diagnostics.
- `src/main.py`: local executable entrypoint for laptop-camera validation.

## Current scope

The present codebase is intentionally laptop-first. It is designed to let you verify the computer-vision loop, tune the pre-processing, and inspect detections before introducing DJI Tello transport, asynchronous UDP ingestion, or downstream control loops.
