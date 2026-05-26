#!/usr/bin/env python3
"""Generate a two-panel latency distribution chart from benchmark JSONs.

Reads raw_latencies_ms from a benchmark results file (typically
results_win.json) and produces violin plots split into GPU and CPU panels.
The split solves the dynamic-range problem: GPU latencies (~1-3 ms) and
CPU latencies (~7-10 ms) on a shared axis would crush the GPU detail
where the interesting story lives.

Usage:
    python make_chart.py results_win.json
    python make_chart.py results_win.json --out latency_distributions.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Display labels matching the README results table.
RUNTIME_LABELS = {
    ("pytorch", "cpu", None): "PyTorch (CPU)",
    ("pytorch", "cuda", None): "PyTorch (CUDA)",
    ("onnxruntime", "cpu", None): "ONNX Runtime (CPU)",
    ("onnxruntime", "cuda", None): "ONNX Runtime (CUDA)",
    ("tensorrt", "cuda", "fp32"): "TensorRT FP32",
    ("tensorrt", "cuda", "fp16"): "TensorRT FP16",
}

# Order within each panel — fastest at the top of the panel so the
# reader's eye lands on the headline configuration first.
GPU_ORDER = [
    ("tensorrt", "cuda", "fp16"),
    ("tensorrt", "cuda", "fp32"),
    ("pytorch", "cuda", None),
    ("onnxruntime", "cuda", None),
]
CPU_ORDER = [
    ("onnxruntime", "cpu", None),
    ("pytorch", "cpu", None),
]


def load_distributions(path: Path) -> dict:
    """Build a {(backend, device, variant): [latencies_ms]} mapping."""
    data = json.loads(path.read_text())
    out = {}
    for r in data["results"]:
        if "raw_latencies_ms" not in r:
            continue  # error rows have no latencies
        key = (r["backend"], r["device"], r.get("variant"))
        out[key] = r["raw_latencies_ms"]
    return out


def plot_panel(ax, distributions, order, title):
    data = [distributions[k] for k in order if k in distributions]
    labels = [RUNTIME_LABELS[k] for k in order if k in distributions]

    parts = ax.violinplot(
        data,
        vert=False,
        showmeans=False,
        showmedians=True,
        widths=0.8,
    )
    # Style violins: muted fill, darker edge.
    for body in parts["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_edgecolor("#2A4A7A")
        body.set_alpha(0.7)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_edgecolor("#2A4A7A")
            parts[key].set_linewidth(1.2)

    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Latency per inference (ms)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()  # fastest at top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="benchmark JSON (e.g. results_win.json)"
    )
    parser.add_argument("--out", type=Path, default=Path("latency_distributions.png"))
    args = parser.parse_args()

    distributions = load_distributions(args.input)

    fig, (ax_gpu, ax_cpu) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        gridspec_kw={"height_ratios": [len(GPU_ORDER), len(CPU_ORDER)]},
    )

    plot_panel(ax_gpu, distributions, GPU_ORDER, "GPU configurations")
    plot_panel(ax_cpu, distributions, CPU_ORDER, "CPU configurations")

    fig.suptitle(
        "YOLOv8n inference latency distribution (RTX 4080, 200 timed runs per config)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
