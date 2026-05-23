# trtbench

Real-time object detection deployment pipeline benchmarking YOLOv8 across
PyTorch, ONNX Runtime, and TensorRT. Cross-platform development (macOS / Windows + RTX 4080) with p50/p95/p99 latency methodology.

> **Status:** In active development. Day 1 baseline (live PyTorch inference) is
> working; ONNX export, benchmark harness, and TensorRT engine results are
> coming next. Full methodology and results table will land in the README when
> benchmarks are complete.

## Current state

- Live person detection via pretrained YOLOv8n with a real-time FPS overlay
- Runs on macOS (CPU / MPS) and Windows (CUDA)
- Device auto-detection; configurable via CLI flags

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python detect_live.py --source 0   # webcam; or pass a video file path
```

For CUDA on Windows, install PyTorch with the CUDA wheel before
`pip install -r requirements.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Roadmap

- [x] Day 1 — Working PyTorch inference loop with FPS overlay
- [x] Day 2 — ONNX export and PyTorch parity check
- [x] Day 3 — Benchmark harness (warmup, sync, p50/p95/p99)
- [x] Day 4 — Cross-machine benchmarks (Mac CPU/MPS, Windows CPU/CUDA)
- [ ] Day 5 — TensorRT engine build (FP32, FP16, optional INT8)
- [ ] Day 6 — Full methodology write-up and results

## Scope note

Uses pretrained COCO weights — no training was performed in this project.
The focus is the deployment and benchmarking pipeline, not model accuracy.
