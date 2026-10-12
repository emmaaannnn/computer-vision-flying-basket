# Computer Vision Flying Basket

DJI Tello human-following mode with one command line for OpenCV, YOLOv8n,
YOLO11n, and YOLO26n.

## Setup

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the MediaPipe hand-landmark model once while you have internet access.
The file is used locally when connected to the drone:

```bash
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
```

Connect to the Tello Wi-Fi network before starting. You can check the
connection with:

```bash
ping -c 3 192.168.10.1
```

## Run

Choose a detector and set the desired follow distance and height:

```bash
python -m src.tello_follow --detector opencv --distance-cm 120 --height-cm 120
python -m src.tello_follow --detector yolov8n --distance-cm 120 --height-cm 120
python -m src.tello_follow --detector yolo11n --distance-cm 120 --height-cm 120
python -m src.tello_follow --detector yolo26n --distance-cm 120 --height-cm 120
```

By default, YOLO detectors use their corresponding checkpoint in the project
directory: `yolov8n.pt`, `yolo11n.pt`, or `yolo26n.pt`. Override it with
`--model-path`, for example:

```bash
python -m src.tello_follow --detector yolov8n --model-path ./yolov8n.pt --distance-cm 120 --height-cm 120
```

The drone does not take off unless you pass `--takeoff`. Check the detector and
controls without takeoff first:

```bash
python -m src.tello_follow --detector yolov8n --distance-cm 120 --height-cm 120 --takeoff
```

The older launcher remains available and defaults to YOLO26n:

```bash
python -m src.tello_yolo --distance-cm 120 --height-cm 120
```

## Person lock and distance estimate

Before lock-on, the video shows detected people but the drone does not follow
anyone. Hold an open palm upright toward the camera for about one second. The
program associates the hand with the closest unambiguous person detection and
locks onto that person. Lower your hand; you do not need to keep the signal up.
Show the same open-palm signal again to toggle the lock off.

While locked, the program matches detections to the selected person's previous
position instead of choosing the largest person in every frame. If the target
is missing or multiple detections are similarly plausible, it sends zero
velocity commands until it can match the target unambiguously. This is
best-effort visual tracking, not guaranteed identity recognition: it does not
recognize your face, and people crossing or occluding one another can still
confuse image-based tracking. Test in a clear area and be ready to stop/land the
drone.

Forward/backward control compares the detected person's height in pixels with
an estimated target height. The estimate assumes a 220-pixel-tall person at
120 cm and scales it for the requested distance. This is a rough image-size
heuristic, not calibrated depth measurement; camera resolution, framing, and
body pose affect it. MediaPipe Hands is used for open-palm detection.

## CSV logs

Each run creates a new CSV under `logs/`, named with the detector and local
date/time, such as `yolov8n_20261012_175137_123456.csv`. A row is written and
flushed as each video frame is processed, so the log grows during the run.

To use a custom filename prefix while still creating a unique file per run:

```bash
python -m src.tello_follow --detector yolov8n --log-file logs/field-test.csv
```

This creates a file such as `logs/field-test_yolov8n_20261012_175137_123456.csv`.
The CSV includes lock state, per-frame detections, bounding boxes, pixel
errors, height readings, and velocity commands.

Press `q` or Escape to quit.
