"""Verify ONNX Runtime output matches PyTorch within FP32 tolerance."""

import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO


def main():
    # Underlying nn.Module — bypasses Ultralytics' preprocessing wrapper
    yolo = YOLO("yolov8n.pt")
    pt_model = yolo.model
    pt_model.eval()

    ort_session = ort.InferenceSession(
        "yolov8n.onnx",
        providers=["CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name

    rng = np.random.default_rng(seed=42)
    n_trials = 5
    rtol, atol = 1e-3, 1e-4  # relative for box coords, absolute for scores

    worst_max = 0.0
    for i in range(n_trials):
        x_np = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)

        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np))
        if isinstance(pt_out, (tuple, list)):
            pt_out = pt_out[0]
        pt_out = pt_out.cpu().numpy()

        ort_out = ort_session.run(None, {input_name: x_np})[0]

        assert (
            pt_out.shape == ort_out.shape
        ), f"shape mismatch: PT {pt_out.shape} vs ORT {ort_out.shape}"

        diff = np.abs(pt_out - ort_out)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        p99_diff = float(np.percentile(diff, 99))
        worst_max = max(worst_max, max_diff)

        close = np.allclose(pt_out, ort_out, rtol=rtol, atol=atol)
        status = "PASS" if close else "FAIL"
        print(
            f"[{status}] trial {i}: "
            f"max|Δ|={max_diff:.2e}  p99|Δ|={p99_diff:.2e}  mean|Δ|={mean_diff:.2e}"
        )
        assert close, f"parity failed at trial {i} (rtol={rtol}, atol={atol})"

    print(f"\n✓ parity check passed across {n_trials} trials")
    print(f"  worst-case max|Δ|: {worst_max:.2e}")
    print(f"  tolerances: rtol={rtol}, atol={atol}")


if __name__ == "__main__":
    main()
