"""
Collect the measured results into presentation/deck_data.json for the
weekly-progress deck (build with `node presentation/build_deck.js`).

    python -m presentation.collect_data
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"


def read_curve():
    pts = []
    p1 = M / "training_seed0_phase1.csv"
    p2 = M / "training_seed0_v3.csv"
    for path, cut in ((p1, 600_000), (p2, None)):
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                t = float(r["total_timesteps"])
                if cut is not None and t > cut:
                    continue
                if path == p2 and t <= 600_000:
                    continue
                pts.append((t, float(r["ep_rew_mean"]),
                            float(r["ep_len_mean"]) * 0.2))
    # down-sample to ~40 points for a readable native chart
    step = max(1, len(pts) // 40)
    pts = pts[::step] + ([pts[-1]] if pts and pts[-1] != pts[::step][-1]
                         else [])
    return pts


def read_yolo():
    """Fine-tuning curve (results.csv) + test-set evaluation."""
    out = {}
    ev = M / "yolo_eval.json"
    if ev.exists():
        out["eval"] = json.loads(ev.read_text())
    rc = M / "yolo_results.csv"
    if rc.exists():
        rows = list(csv.DictReader(open(rc)))
        key = lambda r, k: float(next(v for kk, v in r.items() if kk.strip() == k))
        out["curve"] = {
            "epoch": [int(key(r, "epoch")) for r in rows],
            "map50": [round(key(r, "metrics/mAP50(B)"), 3) for r in rows],
            "map50_95": [round(key(r, "metrics/mAP50-95(B)"), 3) for r in rows],
            "train_box_loss": [round(key(r, "train/box_loss"), 3) for r in rows],
            "val_box_loss": [round(key(r, "val/box_loss"), 3) for r in rows],
        }
    ds = ROOT / "datasets" / "processed" / "warehouse3"
    stats = M / "warehouse3_dataset.json"
    if stats.exists():
        out["dataset"] = json.loads(stats.read_text())
    return out


def main():
    nav = json.loads((M / "navigation_eval.json").read_text())
    perc = json.loads((M / "perception_eval.json").read_text())
    curve = read_curve()
    model_meta = json.loads((ROOT / "models" / "ppo_warehouse_seed0.json")
                            .read_text())
    data = {
        "total_steps_m": round(model_meta["timesteps"] / 1e6, 1),
        "curve": {
            "steps_m": [round(t / 1e6, 2) for t, _, _ in curve],
            "reward": [round(r, 1) for _, r, _ in curve],
            "ep_seconds": [round(s, 1) for _, _, s in curve],
        },
        "controllers": nav["controllers"],
        "ablation": nav["ablation"],
        "reflective": nav.get("reflective", {}),
        "shield_step_out": nav.get("shield_worker_step_out_stop_rate"),
        "episodes": nav["episodes"],
        "confusion_full": perc["confusion_full"],
        "confusion_detector_only": perc["confusion_detector_only"],
        "recall": {
            "detector_only": perc["obstacle_recall_detector_only"],
            "full": perc["obstacle_recall_full"],
            "unknown_detector_only": perc["unknown_box_recall_detector_only"],
            "unknown_full": perc["unknown_box_recall_full"],
        },
        "uncertainty": perc["uncertainty"],
        "speed": perc["speed"],
        "yolo": read_yolo(),
        "yolo_baseline": {"precision": 0.414, "recall": 0.350,
                          "map50": 0.317, "map50_95": 0.1369},
    }
    out = ROOT / "presentation" / "deck_data.json"
    out.write_text(json.dumps(data, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
