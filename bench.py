#!/usr/bin/env python3
"""
bench.py — Benchmark YOLOv8n inference across PyTorch and ONNX Runtime.

Measures p50 / p95 / p99 latency and throughput for the model forward pass
on a fixed input tensor. Output is a structured JSON file suitable for
combining across machines and plotting later.

Methodology
-----------
- We benchmark *pure model inference*: preprocessing and postprocessing are
  excluded, and the input is placed on the target device before timing
  starts. This isolates the runtime's compute cost from any data-loading
  overhead and makes runtimes directly comparable.
- For CUDA backends we call torch.cuda.synchronize() before timing start
  and before timing stop. CUDA kernel launches are asynchronous; without
  synchronization, time.perf_counter() measures only the launch latency,
  not the actual compute.
- A warmup phase runs N_WARMUP iterations untimed to absorb cuDNN
  autotuning, lazy init, and GPU clock ramp.
- The PyTorch model has Conv+BN fused (model.fuse()) to match what
  Ultralytics does at deployment and what the ONNX export already contains.

Usage
-----
    # Run everything available on this machine:
    python bench.py

    # Restrict to one backend / device:
    python bench.py --backend pytorch --device cuda
    python bench.py --backend onnxruntime --device cpu

    # Override paths and iteration counts:
    python bench.py --weights yolov8n.pt --onnx yolov8n.onnx \
                    --n-warmup 30 --n-runs 300 --out results_win.json
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# System info
# --------------------------------------------------------------------------


def get_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        info["torch"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers_available"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime"] = None
        
    try:
        import tensorrt as trt

        info["tensorrt"] = trt.__version__
    except ImportError:
        info["tensorrt"] = None
    return info


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class PyTorchBackend:
    name = "pytorch"

    def __init__(self, weights_path: str, device: str):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = device
        if device == "cuda":
            torch.backends.cudnn.benchmark = True

        # YOLO(...) is the high-level wrapper; .model is the underlying
        # nn.Module DetectionModel. We call it directly so we time only
        # the forward pass, not the wrapper's preprocessing.
        ckpt = YOLO(weights_path)
        self.model = ckpt.model.to(device).eval()
        if hasattr(self.model, "fuse"):
            self.model = self.model.fuse()

    def prepare(self, x_np: np.ndarray):
        return self.torch.from_numpy(x_np).to(self.device)

    def infer(self, x):
        with self.torch.no_grad():
            return self.model(x)

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            # MPS sync is available in modern PyTorch; guard for safety.
            sync = getattr(self.torch.mps, "synchronize", None)
            if sync is not None:
                sync()

    def describe(self) -> dict[str, Any]:
        return {"device": self.device}


class ONNXBackend:
    name = "onnxruntime"

    def __init__(self, onnx_path: str, device: str):
        import onnxruntime as ort

        providers_map = {
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }
        if device not in providers_map:
            raise ValueError(f"ONNX backend doesn't support device '{device}'")

        self.device = device
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=providers_map[device]
        )
        self.input_name = self.session.get_inputs()[0].name
        # Surface which providers actually loaded — if CUDA falls back to
        # CPU silently you want to see it here, not discover it from numbers
        # that look wrong.
        self.active_providers = self.session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.active_providers:
            print(
                f"  WARNING: requested CUDA but session loaded with "
                f"{self.active_providers}. Did you install onnxruntime-gpu?",
                file=sys.stderr,
            )

    def prepare(self, x_np: np.ndarray):
        return x_np

    def infer(self, x):
        return self.session.run(None, {self.input_name: x})

    def sync(self):
        # session.run() blocks until outputs are materialised, so no explicit
        # sync is needed from the caller's perspective.
        pass

    def describe(self) -> dict[str, Any]:
        return {"device": self.device, "active_providers": self.active_providers}

class TensorRTBackend:
    """Run a TensorRT engine via the native Python API.

    GPU buffers are allocated through torch.cuda (no pycuda dependency),
    and inference is dispatched with execute_async_v3 on a dedicated stream.
    The engine's input dtype (FP32 / FP16) is detected from engine metadata,
    and the input tensor is cast accordingly inside prepare() — outside the
    timing window — so the benchmark measures pure inference cost.
    """

    name = "tensorrt"

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.engine_path = engine_path
        self.device = "cuda"  # TRT engines are NVIDIA-only

        # Deserialize the engine.
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        # Ultralytics .engine files are not raw TRT engines: they prepend a
        # 4-byte little-endian length followed by a JSON metadata blob (class
        # names, input shape, etc., used by AutoBackend to rehydrate the
        # model). The raw engine bytes follow that header. We strip it
        # before deserializing so we can use the native TRT runtime directly
        # without going through AutoBackend's wrapper overhead.
        with open(engine_path, "rb") as f:
            meta_len = int.from_bytes(f.read(4), byteorder="little")
            metadata_json = f.read(meta_len).decode("utf-8")
            engine_bytes = f.read()
        self.ultralytics_metadata = json.loads(metadata_json)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }

        # Discover I/O tensors; pre-allocate and bind output buffers.
        self.input_name: str | None = None
        self.output_tensors: dict[str, Any] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            trt_dtype = self.engine.get_tensor_dtype(name)
            if trt_dtype not in dtype_map:
                raise RuntimeError(f"Unsupported TRT dtype on {name}: {trt_dtype}")
            dtype = dtype_map[trt_dtype]
            shape = tuple(self.engine.get_tensor_shape(name))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name is not None:
                    raise RuntimeError("Multi-input engines are not supported")
                self.input_name = name
                self.input_dtype = dtype
                self.input_shape = shape
            else:
                buf = torch.empty(shape, dtype=dtype, device="cuda")
                self.output_tensors[name] = buf
                self.context.set_tensor_address(name, buf.data_ptr())

        if self.input_name is None:
            raise RuntimeError("Engine has no input tensor")

        self.precision = "fp16" if self.input_dtype == torch.float16 else "fp32"
        self.stream = torch.cuda.Stream()

    def prepare(self, x_np: np.ndarray):
        # Move to GPU and cast to engine's expected dtype outside the timing
        # window. Hold the tensor alive on self so its data pointer stays
        # valid across the timed inferences.
        self._input = self.torch.from_numpy(x_np).to(
            device="cuda", dtype=self.input_dtype
        )
        self.context.set_tensor_address(self.input_name, self._input.data_ptr())
        return self._input

    def infer(self, x):
        # Buffers are already bound; we only dispatch.
        self.context.execute_async_v3(self.stream.cuda_stream)

    def sync(self):
        self.stream.synchronize()

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "engine_path": self.engine_path,
        }

# --------------------------------------------------------------------------
# Benchmark loop
# --------------------------------------------------------------------------


def run_benchmark(
    backend, input_np: np.ndarray, n_warmup: int, n_runs: int
) -> list[float]:
    x = backend.prepare(input_np)

    # Warmup
    for _ in range(n_warmup):
        backend.infer(x)
    backend.sync()

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(n_runs):
        backend.sync()
        t0 = time.perf_counter()
        backend.infer(x)
        backend.sync()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return latencies_ms


def summarize(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": p50,
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "throughput_fps_p50": 1000.0 / p50,
    }


# --------------------------------------------------------------------------
# Config selection
# --------------------------------------------------------------------------


def available_configs(
    weights_path: Path,
    onnx_path: Path,
    engine_fp32_path: Path,
    engine_fp16_path: Path,
) -> list[tuple[str, str, str | None]]:
    """Return (backend, device, variant) tuples we can run on this machine."""
    configs: list[tuple[str, str, str | None]] = []
    try:
        import torch

        if weights_path.exists():
            configs.append(("pytorch", "cpu", None))
            if torch.cuda.is_available():
                configs.append(("pytorch", "cuda", None))
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                configs.append(("pytorch", "mps", None))
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        if onnx_path.exists():
            configs.append(("onnxruntime", "cpu", None))
            if "CUDAExecutionProvider" in ort.get_available_providers():
                configs.append(("onnxruntime", "cuda", None))
    except ImportError:
        pass
    try:
        import tensorrt  # noqa: F401

        if engine_fp32_path.exists():
            configs.append(("tensorrt", "cuda", "fp32"))
        if engine_fp16_path.exists():
            configs.append(("tensorrt", "cuda", "fp16"))
    except ImportError:
        pass
    return configs


def build_backend(
    name: str,
    device: str,
    variant: str | None,
    weights_path: Path,
    onnx_path: Path,
    engine_fp32_path: Path,
    engine_fp16_path: Path,
):
    if name == "pytorch":
        return PyTorchBackend(str(weights_path), device)
    if name == "onnxruntime":
        return ONNXBackend(str(onnx_path), device)
    if name == "tensorrt":
        path = engine_fp32_path if variant == "fp32" else engine_fp16_path
        return TensorRTBackend(str(path))
    raise ValueError(f"Unknown backend: {name}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("yolov8n.onnx"))
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument(
        "--engine-fp32", type=Path, default=Path("yolov8n_fp32.engine")
    )
    parser.add_argument(
        "--engine-fp16", type=Path, default=Path("yolov8n_fp16.engine")
    )
    parser.add_argument(
        "--variant",
        choices=["fp32", "fp16"],
        help="For TensorRT, restrict to one precision (default: all available)",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnxruntime", "tensorrt"],
        help="Restrict to a single backend (default: all available)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Restrict to a single device (default: all available)",
    )
    
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # Same input tensor across every config — fairness depends on it.
    input_np = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)

    configs = available_configs(
        args.weights, args.onnx, args.engine_fp32, args.engine_fp16
    )
    if args.backend:
        configs = [c for c in configs if c[0] == args.backend]
    if args.device:
        configs = [c for c in configs if c[1] == args.device]
    if args.variant:
        configs = [c for c in configs if c[2] == args.variant]

    if not configs:
        print(
            "No configurations available. Check --weights / --onnx / --engine-* "
            "paths and that torch / onnxruntime / tensorrt are installed.",
            file=sys.stderr,
        )
        return 1

    print(f"Running {len(configs)} configuration(s): {configs}")
    print(
        f"Input: (1, 3, {args.imgsz}, {args.imgsz})  "
        f"warmup={args.n_warmup}  runs={args.n_runs}"
    )
    print()

    results: list[dict[str, Any]] = []
    for backend_name, device, variant in configs:
        label = f"{backend_name}/{device}" + (f"/{variant}" if variant else "")
        print(f"  {label:<24}", end="", flush=True)
        try:
            backend = build_backend(
                backend_name,
                device,
                variant,
                args.weights,
                args.onnx,
                args.engine_fp32,
                args.engine_fp16,
            )
            latencies = run_benchmark(backend, input_np, args.n_warmup, args.n_runs)
            stats = summarize(latencies)
            results.append(
                {
                    "backend": backend_name,
                    "device": device,
                    "variant": variant,
                    "stats": stats,
                    "extra": backend.describe(),
                    "raw_latencies_ms": latencies,
                }
            )
            print(
                f"p50={stats['p50_ms']:7.2f}ms  "
                f"p95={stats['p95_ms']:7.2f}ms  "
                f"p99={stats['p99_ms']:7.2f}ms  "
                f"fps={stats['throughput_fps_p50']:7.1f}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {e}")
            results.append(
                {
                    "backend": backend_name,
                    "device": device,
                    "variant": variant,
                    "error": str(e),
                }
            )

    payload = {
        "config": {
            "weights": str(args.weights),
            "onnx": str(args.onnx),
            "input_shape": [1, 3, args.imgsz, args.imgsz],
            "n_warmup": args.n_warmup,
            "n_runs": args.n_runs,
            "seed": args.seed,
        },
        "system": get_system_info(),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
