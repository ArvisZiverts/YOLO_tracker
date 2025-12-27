"""
vehicle_yolo_tracker.py
====================================================

VIENS FAILS iesniegšanai (PyCharm / Google Colab stils).

IZPILDA UZDEVUMU:
✔ Objekta izvēle (truck, bus, car u.c.)
✔ YOLOv8 apmācība uz sava dataset
✔ YOLO + tracker (ByteTrack / BoT-SORT)
✔ Video (>30s) apstrāde
✔ Objekta trajektorija (x,y)
✔ Saglabā video + CSV
✔ Viss konfigurējams (epochs, batch, classes...)

Bibliotēkas:
    pip install ultralytics opencv-python numpy pandas

----------------------------------------------------
PALAIŠANA:

1) APMĀCĪBA
python vehicle-tracker.py images \
 --data dataset/data.yaml \
 --model yolov8m.pt \

2) TRACKING + TRAJEKTORIJA
python vehicle-tracker.py track \
 --weights runs/detect/images/weights/best.pt \
 --source videos/input.mp4 \

3) VISU KOPĀ
python vehicle-tracker.py both \
 --data dataset/data.yaml \
 --source videos/input.mp4 \

----------------------------------------------------
REZULTĀTI:
outputs/tracked.mp4
outputs/trajectory.csv
"""

import argparse
from collections import defaultdict ,deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ====================================================
# CENTRAL CONFIG (EDIT ONLY HERE)
# ====================================================

CONFIG = {
    # TRAINING
    "epochs": 15,
    "imgsz": 640,
    "batch": 4,
    "device": "mps",
    #model": "yolov8n.pt",
    "model": "yolov8m.pt",

    # TRACKING
    "conf": 0.65,
    "iou": 0.45,
    "max_trail": 150,
    "tracker": "bytetrack.yaml",

    # CLASSES TO TRACK (None = all)
    "classes": ["truck", "bus" ,"car"]
}


# ====================================================
# Palīgfunkcijas
# ====================================================

def get_color(track_id):
    track_id = int(track_id)
    np.random.seed(track_id + 12345)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# ====================================================
# YOLO APMĀCĪBA
# ====================================================

def train_yolo(args):
    print(">>> Sākas YOLO apmācība...")
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=CONFIG["device"],
        project="runs/detect",
        name="images",
        pretrained=True
    )

    print(">>> Apmācība pabeigta!")
    print(">>> Modelis saglabāts: runs/detect/images/weights/best.pt")


# ====================================================
# TRACKING + TRAJEKTORIJA
# ====================================================

def track_video(args):
    print(">>> Sākas tracking...")

    model = YOLO(args.weights)

    class_ids = None
    if CONFIG["classes"]:
        class_ids = [
            idx for idx, name in model.names.items()
            if name.lower() in [c.lower() for c in CONFIG["classes"]]
        ]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Nevar atvērt video failu")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    video_out = out_dir / "tracked.mp4"
    traj_out = out_dir / "trajectory.csv"

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    trajectories = defaultdict(lambda: deque(maxlen=25))
    csv_rows = []

    results = model.track(
        source=args.source,
        stream=True,
        persist=True,
        tracker=CONFIG["tracker"],
        conf=CONFIG["conf"],
        iou=CONFIG["iou"],
        classes=class_ids,
        max_det=100
    )

    for frame_idx, r in enumerate(results):
        frame = r.orig_img.copy()

        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [-1] * len(boxes)

            for box, conf, cls, tid in zip(boxes, confs, classes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = box_center(box)

                if tid >= 0:
                    trajectories[tid].append((cx, cy))

                    xs = [p[0] for p in trajectories[tid]]
                    ys = [p[1] for p in trajectories[tid]]
                    smooth_center = (sum(xs) // len(xs), sum(ys) // len(ys))

                color = get_color(int(tid))
                label = f"{model.names[cls]} ID:{tid}"

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w > 0.8 * w or box_h > 0.8 * h:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                csv_rows.append([frame_idx, tid, model.names[cls], conf, cx, cy])

        for tid, pts in trajectories.items():
            if len(pts) > 1:
                cv2.polylines(
                    frame,
                    [np.array(list(pts), dtype=np.int32)],
                    False,
                    get_color(tid),
                    2
                )

        writer.write(frame)

        if args.show:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Saglabā trajektoriju CSV
    pd.DataFrame(
        csv_rows,
        columns=["frame", "track_id", "class", "confidence", "cx", "cy"]
    ).to_csv(traj_out, index=False)

    print(">>> Tracking pabeigts")
    print(f">>> Video: {video_out}")
    print(f">>> Trajektorija: {traj_out}")


# ====================================================
# CLI
# ====================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO Vehicle Tracking (1 fails)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # TRAIN
    p_train = sub.add_parser("images")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", default=CONFIG["model"])

    # TRACK
    p_track = sub.add_parser("track")
    p_track.add_argument("--weights", required=True)
    p_track.add_argument("--source", required=True)
    p_track.add_argument("--show", action="store_true")

    # BOTH
    p_both = sub.add_parser("both")
    p_both.add_argument("--data", required=True)
    p_both.add_argument("--source", required=True)
    p_both.add_argument("--model", default=CONFIG["model"])
    p_both.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if args.cmd == "images":
        train_yolo(args)

    elif args.cmd == "track":
        track_video(args)

    elif args.cmd == "both":
        train_yolo(args)
        args.weights = "runs/detect/images/weights/best.pt"
        track_video(args)


if __name__ == "__main__":
    main()