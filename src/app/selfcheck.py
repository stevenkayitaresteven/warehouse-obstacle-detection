"""
Self-check for the demo: proves the installed environment can run it.

    python -m src.app.selfcheck

1. Loads the trained PPO policy and drives one simulated episode.
2. Renders a frame (OpenCV drawing).
3. Builds the Gradio UI, starts the web server on a free port, fetches the
   page over HTTP and shuts the server down again.

Prints "SELF-CHECK PASSED" and exits 0 on success (used by run_demo.py
--check and by CI on Windows, macOS and Linux).

    python -m src.app.selfcheck --perception [--require-depth]

also runs the real-image tab's pipeline (fine-tuned YOLO11n, and Depth
Anything V2 if it can be downloaded) on an example photo.
"""

from __future__ import annotations

import sys
import time
import urllib.request

import numpy as np


def check_perception(require_depth=False):
    from pathlib import Path

    from PIL import Image

    from src.perception.live_pipeline import analyse

    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "assets" / "examples" /
                                "example_1.jpg").convert("RGB"))
    cache = {}
    vis, md = analyse(img, cache)
    n = int(md.split("**Detector:**")[1].split("—")[1].split("detection")[0])
    print(f"[perception] detector {cache['detector_name']}: {n} detections; "
          f"depth model: {'loaded' if cache.get('depth') else 'NOT available'}")
    assert n > 0, "fine-tuned detector found nothing on the example photo"
    if require_depth:
        assert cache.get("depth") is not None, cache.get("depth_error")
        assert vis.shape[1] == 2 * img.shape[1], "depth panel missing"


def main():
    t0 = time.time()
    if "--perception" in sys.argv:
        check_perception("--require-depth" in sys.argv)
    from src.app.sim_session import SimSession

    sim = SimSession()
    for _ in range(60):
        sim.step()
        if sim.done:
            break
    frame = sim.frame()
    assert frame.ndim == 3 and frame.shape[2] == 3, frame.shape
    assert frame.dtype == np.uint8
    print(f"[1/3] simulation + PPO policy OK ({sim.env.steps} steps, "
          f"frame {frame.shape[1]}x{frame.shape[0]})")

    import gradio as gr

    from src.app.app import build_ui, free_port

    demo = build_ui()
    demo.queue()
    print(f"[2/3] Gradio {gr.__version__} UI built")

    port = free_port("127.0.0.1", 7870)
    demo.launch(server_name="127.0.0.1", server_port=port,
                prevent_thread_lock=True, inbrowser=False, quiet=True)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/",
                                    timeout=30) as r:
            body = r.read()
            assert r.status == 200, r.status
            assert len(body) > 500
    finally:
        demo.close()
    print(f"[3/3] web server answered on port {port}")
    print(f"SELF-CHECK PASSED in {time.time() - t0:.1f} s "
          f"(Python {sys.version.split()[0]}, {sys.platform})")


if __name__ == "__main__":
    main()
