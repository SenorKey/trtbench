"""
Real-time person detection — Day 1 baseline.

Loads pretrained YOLOv8n, runs inference on a video source (file or webcam),
and displays annotated frames with a live FPS readout. Acts as the reference
PyTorch implementation we'll later compare against ONNX Runtime and TensorRT.

Usage:
    python detect_live.py --source 0                 # default webcam
    python detect_live.py --source video.mp4         # video file
    python detect_live.py --source 0 --device cuda   # force GPU
    python detect_live.py --source 0 --device mps    # Apple Silicon GPU
"""
import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time person detection demo.")
    p.add_argument("--source", default="0",
                   help="Video source: integer for webcam index, or path to a video file.")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights (downloads automatically on first run).")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold for detections.")
    p.add_argument("--device", default="auto",
                   help="Inference device: auto, cpu, cuda, or mps.")
    return p.parse_args()


def resolve_device(requested):
    """Pick the best available device when 'auto', otherwise honor the request."""
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def open_source(source):
    """Open a webcam (numeric source) or a video file."""
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Loading {args.model} on device: {device}")

    model = YOLO(args.model)
    cap = open_source(args.source)
    frame_times = deque(maxlen=30)  # rolling window for FPS smoothing

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=args.conf,
                device=device,
                verbose=False,
            )
            annotated = results[0].plot()

            frame_times.append(time.perf_counter() - t0)
            avg = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg if avg > 0 else 0.0
            cv2.putText(
                annotated,
                f"{fps:5.1f} FPS  |  {device}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Person Detection (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
