"""
Plot the robot-view fine-tuning of the detector (epochs 2 -> 8).

Reads results/metrics/yolo_robotview_results.csv (per-epoch validation)
and results/metrics/detector_finetune.json (held-out tests), writes
results/graphs/yolo_robotview_finetune.png.

    python -m scripts.plot_detector_finetune
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "metrics" / "yolo_robotview_results.csv"
JS = ROOT / "results" / "metrics" / "detector_finetune.json"
OUT = ROOT / "results" / "graphs" / "yolo_robotview_finetune.png"


def main():
    rows = list(csv.DictReader(open(CSV)))
    rep = json.loads(JS.read_text())
    e2 = rep["training"]["validation_epoch2"]
    ep = [2] + [int(r["epoch"]) for r in rows]
    m50 = [e2["mAP50"]] + [float(r["metrics/mAP50(B)"]) for r in rows]
    m95 = [e2["mAP50_95"]] + [float(r["metrics/mAP50-95(B)"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    a = axes[0]
    a.plot(ep, [100 * v for v in m50], "o-", color="#2a7fd4", label="mAP@50")
    a.plot(ep, [100 * v for v in m95], "s-", color="#7a5195",
           label="mAP@50-95")
    a.axvline(2.5, color="#999", ls="--")
    a.text(2.55, 100 * min(m95) + 1, "continued here\n(recovered weights)",
           fontsize=8, color="#555")
    a.set_xlabel("epoch")
    a.set_ylabel("%")
    a.set_title("Validation (real + robot-view val images)")
    a.legend(loc="lower right")
    a.grid(alpha=0.3)

    held = rep["heldout_tests"]["results"]
    names = [("original", "original\n(real photos only)", "#bbbbbb"),
             ("epoch2", "epoch 2", "#9ecae1"),
             ("epoch8", "epoch 8", "#2a7fd4")]
    a = axes[1]
    groups = [("sim_heldout", "robot-view frames\n(unseen objects)"),
              ("real_test", "real test photos")]
    w = 0.26
    for j, (key, label, col) in enumerate(names):
        vals = [100 * held[key][g]["mAP50"] for g, _ in groups]
        xs = [i + (j - 1) * w for i in range(len(groups))]
        a.bar(xs, vals, w, color=col, label=label.replace("\n", " "))
        for x, v in zip(xs, vals):
            a.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    a.set_xticks(range(len(groups)), [g[1] for g in groups])
    a.set_ylim(0, 125)
    a.set_ylabel("mAP@50 (%)")
    a.set_title("Held-out tests")
    a.legend(loc="upper center", ncol=3, fontsize=8)
    a.grid(alpha=0.3, axis="y")

    sim = rep["in_simulation"]["results"]
    a = axes[2]
    metrics = [("raw detector\nprecision", "raw_detector", "precision"),
               ("raw detector\nrecall", "raw_detector", "recall"),
               ("full stack\nprecision", "full_stack", "precision"),
               ("full stack\nrecall", "full_stack", "recall")]
    for j, (key, label, col) in enumerate(names[1:]):
        vals = [100 * sim[key]["driving_video"][grp][m]
                for _, grp, m in metrics]
        xs = [i + (j - 0.5) * 0.36 for i in range(len(metrics))]
        a.bar(xs, vals, 0.36, color=col, label=label)
        for x, v in zip(xs, vals):
            a.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    a.set_xticks(range(len(metrics)), [m[0] for m in metrics], fontsize=9)
    a.set_ylim(0, 125)
    a.set_ylabel("%")
    frames = sim["epoch8"]["driving_video"]["frames"]
    a.set_title(f"Robot driving held-out warehouses ({frames} frames)")
    a.legend(loc="upper center", ncol=2, fontsize=8)
    a.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
