# Computer Vision Flying Basket

DJI Tello human-following mode with one simple command line.
## What you choose

- `--detector`: `opencv`, `yolo11n`, or `yolo26n`
- `--distance-cm`: how far from the person you want the drone to stay
- `--height-cm`: how high from the ground you want the drone to stay
- `--log-file`: where to write the CSV log

## Setup

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
ping -c 3 192.168.10.1
```

## Commands

OpenCV baseline:

```bash
python -m src.tello_follow --detector opencv --distance-cm 120 --height-cm 120
```

YOLO11n:

```bash
python -m src.tello_follow --detector yolo11n --distance-cm 120 --height-cm 120
```

YOLO26n:

```bash
python -m src.tello_follow --detector yolo26n --distance-cm 120 --height-cm 120
```

Add takeoff when you are ready to fly:

```bash
python -m src.tello_follow --detector yolo26n --distance-cm 120 --height-cm 120 --takeoff
```

If you keep pasting the older launcher, this still works:

```bash
python -m src.tello_yolo --distance-cm 120 --height-cm 120
```

## Why no model path?

You should not need to pass a path for normal use. The detector picks the default checkpoint automatically:

- `opencv` uses the built-in OpenCV human detector.
- `yolo11n` uses `./yolo11n.pt`.
- `yolo26n` uses `./yolo26n.pt`.

Use `--model-path` only if you want to override the default checkpoint.

## Logs

Every run writes a CSV file to `logs/tello_follow.csv` unless you choose a different `--log-file`.

## Testing order

1. Run OpenCV first without takeoff.
2. If OpenCV misses you, switch to YOLO11n.
3. If YOLO11n is still weak, try YOLO26n.
4. Only add `--takeoff` after the detector and height settings look right.

## Notes

- Press `q` to quit the window.
- If you want to override the checkpoint manually, add `--model-path`.
# Computer Vision Flying Basket

DJI Tello follow mode with one shared command line for OpenCV, YOLO11n, and YOLO26n.

## Quick Start

1. Activate the virtual environment.
2. Install dependencies.
3. Connect to the Tello Wi-Fi network.
4. Start with the simplest detector first.

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
ping -c 3 192.168.10.1
```

## Commands

OpenCV baseline:

```bash
python -m src.tello_follow --detector opencv --target-height-cm 120
```

YOLO11n:

```bash
python -m src.tello_follow --detector yolo11n --target-height-cm 120
```

YOLO26n:

```bash
python -m src.tello_follow --detector yolo26n --target-height-cm 120
```

Same run, but with takeoff:

```bash
python -m src.tello_follow --detector yolo26n --target-height-cm 120 --takeoff
```

Compatibility command if you still paste the older launcher:

```bash
python -m src.tello_yolo --target-height-cm 120
```

If you ever want to override the checkpoint manually, use `--model-path`:

```bash
python -m src.tello_follow --detector yolo26n --model-path ./yolo26n.pt --target-height-cm 120
```

## Testing Order

1. Run OpenCV first without takeoff.
2. If OpenCV misses you, switch to YOLO11n.
3. If YOLO11n is still weak, try YOLO26n.
4. Only add `--takeoff` when the detector output and height target look correct.

## Logs

Each run writes `logs/tello_follow.csv` with detection boxes, current height, target height, and command values.

## Notes

- `--detector` selects the default model, so `--model-path` is optional.
- `--model-path` is only for manual overrides or benchmarking.
- Press `q` to quit the window.
