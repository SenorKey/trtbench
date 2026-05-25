"""
Export YOLOv8n to TensorRT engines (FP32 and FP16).

TensorRT engines are GPU-architecture-specific. The engines built here are
tuned for the GPU they're built on (Ada Lovelace / RTX 4080) and will need
to be rebuilt for different GPU families. This is the tradeoff for the
ahead-of-time compilation that makes TRT fast.

Build time is a few minutes per engine the first time, since the TRT builder
benchmarks candidate kernel implementations to pick the fastest one.
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

DEFAULT_ENGINE = Path("yolov8n.engine")

for precision_label, half in [("fp32", False), ("fp16", True)]:
    print(f"\n=== Building {precision_label.upper()} engine ===")
    model = YOLO("yolov8n.pt")
    model.export(format="engine", half=half, imgsz=640, dynamic=False)

    target = Path(f"yolov8n_{precision_label}.engine")
    if target.exists():
        target.unlink()
    shutil.move(str(DEFAULT_ENGINE), str(target))
    print(f"Saved: {target}")

print("\nDone.")