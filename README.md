# Computer Vision Flying Basket

This repository is a simple DJI Tello video + YOLO setup for the Flying Basket project.

The main goal is now:

- connect to the Tello Wi-Fi;
- stream the drone camera to the laptop;
- run YOLO26n on those frames;
- show the annotated video on screen.

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
3. Connect your laptop to the Tello Wi-Fi network.
4. Start the Tello video + YOLO viewer:

```bash
python -m src.tello_yolo --model-name ./yolo26n.pt
```

If you want the simplest possible run, you can omit the model flag and let it use the default `./yolo26n.pt` path.

## DJI Tello video setup

The Tello is controlled over its own Wi-Fi network, not USB. Plugging it into the laptop only charges the battery.

### 1. Connect to the Tello network

1. Power on the Tello.
2. On macOS, open Wi-Fi and connect to the drone SSID, which usually looks like `Tello-XXXX`.
3. Keep the drone on a flat surface with the propellers clear.

If you want to confirm the link from the terminal, run:

```bash
ping -c 3 192.168.10.1
```

### 2. Start the video stream check

Run this to confirm the laptop can receive the drone camera and process it with YOLO:

```bash
python -m src.tello_yolo --model-name ./yolo26n.pt
```

This will:

- connect to the Tello SDK;
- turn on the drone video stream;
- read frames from the Tello camera;
- run YOLO26n on the laptop;
- draw the detections in a display window.

## Model benchmarking

The repository currently includes these local nano checkpoints for comparison:

- `./yolov8n.pt`
- `./yolo11n.pt`
- `./yolo26n.pt`

Use an explicit local path for `--model-name` when benchmarking. This keeps the run pinned to the checkpoint in the repository root instead of letting Ultralytics substitute a newer alias.

To compare the three local models with the Tello camera stream, run these commands one at a time:

```bash
python -m src.tello_yolo --model-name ./yolov8n.pt
python -m src.tello_yolo --model-name ./yolo11n.pt
python -m src.tello_yolo --model-name ./yolo26n.pt
```

The loop timing and detection stats appear in the live Tello window itself.

If you want the exact screenshot-style technical overlay, add `--show-technical-diagnostics`:

```bash
python -m src.tello_yolo --model-name ./yolov8n.pt --show-technical-diagnostics
python -m src.tello_yolo --model-name ./yolo11n.pt --show-technical-diagnostics
python -m src.tello_yolo --model-name ./yolo26n.pt --show-technical-diagnostics
```

On macOS or Linux, you can step through the three local models with a shell loop. Close the display window after each run to move to the next model:

```bash
for model in ./yolov8n.pt ./yolo11n.pt ./yolo26n.pt; do
	python -m src.tello_yolo --model-name "$model"
done
```

## Code layout

The repository is now split into two clear areas:

- `src/vision/`: the YOLO person detector.
- `src/tello/`: the Tello video stream adapter and the Tello YOLO viewer.

The root entrypoints stay as thin wrappers so the existing commands still work:

- `src/tello_yolo.py` -> `src.tello.yolo_stream`

The main files you will usually tweak are:

- `src/vision/detector.py`: YOLO person detection.
- `src/tello/video_stream.py`: Tello camera connection and frame handling.
- `src/tello/yolo_stream.py`: the display loop that runs YOLO on the Tello frames.
