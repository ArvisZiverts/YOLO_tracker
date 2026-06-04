# YOLO Vehicle Tracker

Train a custom **YOLOv8** detector on a vehicle dataset, then run **multi-object tracking** on video to draw bounding boxes, persistent track IDs, and movement trajectories — and export the full trajectory data to CSV.

The whole pipeline (training, tracking, and both end-to-end) lives in a single script: [`vehicle-tracker.py`](vehicle-tracker.py).

---

## Features

- **Custom object detection** — fine-tune YOLOv8 on your own labeled vehicle images.
- **Multi-object tracking** — YOLO + [ByteTrack](trackers/bytetrack.yaml) for stable, persistent IDs across frames.
- **Trajectory visualization** — each tracked vehicle gets a unique color, a labeled box (`class ID:n`), and a poly-line trail showing its path.
- **CSV export** — per-frame `frame, track_id, class, confidence, cx, cy` written to `outputs/trajectory.csv`.
- **Annotated video output** — written to `outputs/tracked.mp4`.
- **Single central config** — all hyperparameters live in one `CONFIG` dict at the top of the script.
- **Apple Silicon ready** — defaults to the `mps` device (easy to switch to `cuda` or `cpu`).

---

## Dataset

The included dataset (`dataset/`) is a 6-class vehicle dataset in YOLO format.

| ID | Class       |
|----|-------------|
| 0  | car         |
| 1  | threewheel  |
| 2  | bus         |
| 3  | truck       |
| 4  | motorbike   |
| 5  | van         |

| Split | Images |
|-------|--------|
| Train | 2,100  |
| Val   | 900    |

Layout (defined in [`dataset/data.yaml`](dataset/data.yaml)):

```
dataset/
├── data.yaml          # path / train / val / class names
├── classes.txt        # class list
├── images/
│   ├── train/         # 2100 images
│   └── val/           # 900 images
└── labels/
    ├── train/         # YOLO-format .txt labels
    └── val/
```

Each label line is `class_id cx cy w h` with normalized coordinates.

---

## Project structure

```
.
├── vehicle-tracker.py        # main script: train / track / both
├── fix_labels.py             # helper to strip out-of-range class IDs from labels
├── dataset/                  # images, labels, data.yaml
├── trackers/
│   └── bytetrack.yaml        # ByteTrack tracker config
├── videos/
│   └── input.mp4             # sample input video
├── outputs/                  # generated: tracked.mp4 + trajectory.csv
├── runs/                     # YOLO training runs (weights, metrics, plots)
├── yolov8n.pt                # pretrained YOLOv8 nano weights
└── yolov8m.pt                # pretrained YOLOv8 medium weights
```

A trained model is already provided at `runs/detect/images/weights/best.pt`.

---

## Installation

Requires **Python 3.11+**.

```bash
pip install ultralytics opencv-python numpy pandas
```

`ultralytics` pulls in PyTorch automatically. On Apple Silicon the `mps` backend is used by default; on an NVIDIA GPU set `"device": "cuda"` in `CONFIG`, or use `"cpu"` if you have no GPU.

---

## Configuration

All tunable parameters are in the `CONFIG` dict near the top of `vehicle-tracker.py`:

```python
CONFIG = {
    # TRAINING
    "epochs": 15,
    "imgsz": 640,
    "batch": 4,
    "device": "mps",
    "model": "yolov8m.pt",

    # TRACKING
    "conf": 0.65,          # detection confidence threshold
    "iou": 0.45,           # NMS IoU threshold
    "max_trail": 150,      # max trajectory length
    "tracker": "bytetrack.yaml",

    # CLASSES TO TRACK (None = all)
    "classes": ["truck", "bus", "car"],
}
```

By default tracking is limited to `truck`, `bus`, and `car`. Set `"classes": None` to track every class the model knows.

---

## Usage

The script has three subcommands.

### 1. Train

```bash
python vehicle-tracker.py images \
  --data dataset/data.yaml \
  --model yolov8m.pt
```

Trains YOLOv8 and saves the best weights to `runs/detect/images/weights/best.pt`.

### 2. Track

```bash
python vehicle-tracker.py track \
  --weights runs/detect/images/weights/best.pt \
  --source videos/input.mp4
```

Add `--show` to display the annotated frames live (press `Esc` to stop).

### 3. Both (train, then track)

```bash
python vehicle-tracker.py both \
  --data dataset/data.yaml \
  --source videos/input.mp4 \
  --model yolov8m.pt
```

Runs training, then automatically tracks using the freshly trained `best.pt`.

---

## Outputs

After a tracking run, `outputs/` contains:

| File              | Description                                                          |
|-------------------|----------------------------------------------------------------------|
| `tracked.mp4`     | Input video with boxes, labels, IDs, and trajectory trails.          |
| `trajectory.csv`  | Per-detection rows: `frame, track_id, class, confidence, cx, cy`.    |

The CSV can be loaded directly with pandas for further analysis (counting, speed estimation, heatmaps, etc.):

```python
import pandas as pd
df = pd.read_csv("outputs/trajectory.csv")
```

---

## How it works

1. **Detection** — YOLOv8 runs on each frame, optionally filtered to the configured classes.
2. **Tracking** — `model.track(..., persist=True, tracker="bytetrack.yaml")` assigns and maintains a stable `track_id` per vehicle across frames.
3. **Trajectories** — each track's box centers are stored in a rolling `deque` and drawn as a poly-line trail; colors are seeded deterministically from the track ID.
4. **Filtering** — boxes larger than 80% of the frame are skipped to suppress spurious giant detections.
5. **Export** — annotated frames are written to video and detection rows accumulated into the CSV.

ByteTrack behavior is tuned in [`trackers/bytetrack.yaml`](trackers/bytetrack.yaml) (match thresholds, track buffer for occlusion, etc.).

---

## Helper: `fix_labels.py`

A small one-off utility that removes label lines whose class ID falls outside the valid `0–5` range, keeping the label files consistent with `data.yaml`. Adjust the `label_dir` paths inside it to match your folder layout before running.

---

## Notes

- The pretrained `yolov8n.pt` (nano) and `yolov8m.pt` (medium) weights are committed for convenience. Nano is faster; medium is more accurate.
- Source video should be at least ~30 seconds for a meaningful tracking demo.
- Training runs (metrics, confusion matrix, sample predictions) accumulate under `runs/detect/`.
