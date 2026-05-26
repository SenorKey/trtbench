#!/usr/bin/env python3
"""Combine per-machine benchmark JSONs into a unified comparison table.

Usage:
    python make_table.py results_win.json results_mac.json
    python make_table.py results_win.json results_mac.json --csv results.csv
    python make_table.py results_win.json results_mac.json \\
        --labels "RTX 4080 (Windows)" "Apple Silicon (macOS)"
"""

import argparse
import csv
import json
from pathlib import Path


BACKEND_DISPLAY = {
    "pytorch": "PyTorch",
    "onnxruntime": "ONNX Runtime",
    "tensorrt": "TensorRT",
}


def infer_label(system: dict, fallback: str) -> str:
    """Derive a short, recognizable machine label from the system block."""
    os_name = system.get("platform", "").split("-")[0] or "Unknown OS"
    if system.get("cuda_available") and system.get("cuda_device"):
        return f'{system["cuda_device"]} ({os_name})'
    if system.get("mps_available"):
        return f"Apple Silicon ({os_name})"
    return fallback


def runtime_label(result: dict) -> str:
    """Build a human-readable runtime label, e.g. 'PyTorch (CUDA)' or
    'TensorRT (CUDA, FP16)' for engines built at reduced precision."""
    backend = BACKEND_DISPLAY.get(result["backend"], result["backend"])
    device = result["device"].upper()
    variant = result.get("variant")
    if variant:
        return f"{backend} ({device}, {variant.upper()})"
    return f"{backend} ({device})"


def collect_rows(json_paths, labels):
    rows = []
    for path, label in zip(json_paths, labels):
        data = json.loads(Path(path).read_text())
        machine = label or infer_label(data["system"], Path(path).stem)
        for r in data["results"]:
            s = r["stats"]
            rows.append({
                "machine": machine,
                "runtime": runtime_label(r),
                "p50_ms": s["p50_ms"],
                "p95_ms": s["p95_ms"],
                "p99_ms": s["p99_ms"],
                "fps_p50": s["throughput_fps_p50"],
            })
    # Sort by machine, then by p50 ascending — fastest config per machine first.
    rows.sort(key=lambda r: (r["machine"], r["p50_ms"]))
    return rows


def to_markdown(rows) -> str:
    headers = ["Machine", "Runtime", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Throughput (FPS @ p50)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append(
            f'| {r["machine"]} | {r["runtime"]} | '
            f'{r["p50_ms"]:.2f} | {r["p95_ms"]:.2f} | {r["p99_ms"]:.2f} | '
            f'{r["fps_p50"]:.1f} |'
        )
    return "\n".join(lines)


def to_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["machine", "runtime", "p50_ms", "p95_ms", "p99_ms", "fps_p50"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="benchmark JSON files")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="machine labels (one per input); auto-inferred if omitted")
    parser.add_argument("--csv", default=None, help="also write CSV to this path")
    args = parser.parse_args()

    labels = args.labels if args.labels is not None else [None] * len(args.inputs)
    if len(labels) != len(args.inputs):
        parser.error(f"--labels must have {len(args.inputs)} values (one per input file)")

    rows = collect_rows(args.inputs, labels)
    print(to_markdown(rows))
    if args.csv:
        to_csv(rows, args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
