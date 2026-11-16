"""
Evaluate the real-image camera simulation (held-out objects only).

1. **Perception accuracy on driving video.** The robot drives random
   warehouses; every frame of its real-image camera is scored against the
   renderer's ground truth:
     - the raw detector (boxes at conf >= 0.25),
     - the full stack (depth fusion + gates + tracking),
     - ablations: no tracking, no map / size gates,
   plus distance error and how many unknown objects depth flagged.
2. **Navigation.** PPO + safety shield with the real-image camera in the
   loop, and the same episodes with the emulated perception.

All objects are cut-outs from the val / test photos (never used for
training). Warehouse style "c" was not used for the fine-tuning composites.

    python -m scripts.eval_real_image_sim --episodes 40 --frames-episodes 30

Writes results/metrics/real_image_eval.json.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics" / "real_image_eval.json"


class _Cached:
    """Run the detector once per frame even if several stacks ask."""

    def __init__(self, det):
        self.det, self.key, self.val = det, None, None
        self.names, self.backend = det.names, det.backend

    def __call__(self, rgb, conf=None):
        key = (id(rgb), conf)
        if key != self.key:
            self.key, self.val = key, self.det(rgb, conf)
        return self.val


def perception_worker(args):
    seeds, style = args
    from src.evaluation.sim_detection import DetectionMeter
    from src.navigation.controllers import NumpyPolicy
    from src.perception.camera_perception import (
        CameraPerception, PerceptionConfig,
    )
    from src.perception.onnx_detector import OnnxDetector
    from src.safety.shield import SafetyShield
    from src.simulation.photo_camera import PhotoCamera
    from src.simulation.warehouse_env import EnvConfig, WarehouseEnv

    det = _Cached(OnnxDetector(threads=1))
    cam = PhotoCamera(splits=("val", "test"), style=style)
    env = WarehouseEnv(EnvConfig())
    variants = {
        "full": PerceptionConfig(),
        "no_tracking": PerceptionConfig(use_tracking=False),
        "no_gates": PerceptionConfig(use_depth_gate=False,
                                     use_size_gate=False),
    }
    stacks = {k: CameraPerception(cam, det, c) for k, c in variants.items()}
    meters = {k: DetectionMeter() for k in variants}
    pol, shield = NumpyPolicy(), SafetyShield()
    for seed in seeds:
        rng = np.random.default_rng([seed, 9])
        obs, _ = env.reset(seed=seed, options={"counts": {
            "workers": int(rng.integers(2, 6)),
            "forklifts": int(rng.integers(1, 3)),
            "pallets": int(rng.integers(2, 6)),
            "boxes": int(rng.integers(1, 5))}})
        shield.reset()
        cam.reset(env, seed, style=style)
        for s in stacks.values():
            s.reset()
        for _ in range(300):
            a = pol.act(obs, env)
            a, _, _ = shield.filter(env, a)
            obs, _, term, trunc, _ = env.step(a)
            frame = cam.render(env, rng)
            for k, s in stacks.items():
                meters[k].update(frame, s.process(env, frame))
            if term or trunc:
                break
    return {k: m for k, m in meters.items()}


def merge(meters):
    from src.evaluation.sim_detection import DetectionMeter
    out = DetectionMeter()
    for m in meters:
        out.frames += m.frames
        for k in out.rec:
            out.rec[k] += m.rec[k]
        for c, n in m.n_gt.items():
            out.n_gt[c] = out.n_gt.get(c, 0) + n
        out.range_err += m.range_err
        out.unknown[0] += m.unknown[0]
        out.unknown[1] += m.unknown[1]
        out.ms += m.ms
    return out.summary()


def nav_worker(args):
    seeds, real = args
    from src.navigation.controllers import NumpyPolicy
    from src.perception.camera_perception import attach_real_camera
    from src.perception.onnx_detector import OnnxDetector
    from src.safety.shield import SafetyShield
    from src.simulation.warehouse_env import EnvConfig, WarehouseEnv
    env = WarehouseEnv(EnvConfig())
    if real:
        attach_real_camera(env, detector=OnnxDetector(threads=1))
    pol, shield = NumpyPolicy(), SafetyShield()
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        shield.reset()
        while True:
            a = pol.act(obs, env)
            a, _, _ = shield.filter(env, a)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        out.append({"seed": seed, "reached": bool(info["reached"]),
                    "collided": bool(info["collided"]),
                    "collision_with": info["collision_with"],
                    "steps": int(info["steps"]),
                    "near_misses": int(info["near_misses"]),
                    "interventions": int(shield.interventions)})
    return out, (env.vision_meter if real else None)


def nav_summary(res, dt=0.2):
    n = len(res)
    ok = [r for r in res if r["reached"]]
    coll = {}
    for r in res:
        if r["collided"]:
            coll[r["collision_with"]] = coll.get(r["collision_with"], 0) + 1
    return {"episodes": n,
            "success_rate": len(ok) / n,
            "collision_rate": sum(r["collided"] for r in res) / n,
            "timeout_rate": sum(not r["reached"] and not r["collided"]
                                for r in res) / n,
            "mean_time_to_goal_s": float(np.mean([r["steps"] * dt
                                                  for r in ok])) if ok
            else None,
            "collisions_by_class": coll}


def chunks(lst, k):
    return [lst[i::k] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--frames-episodes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=10_000)
    a = ap.parse_args()
    t0 = time.time()
    rep = {"note": "Simulation with the real-image camera. Objects are "
                   "val/test cut-outs the detector never saw. Not "
                   "physical-robot results."}
    with Pool(a.workers) as pool:
        # ---- perception on driving video, per warehouse style ----------
        rep["perception"] = {}
        seeds = list(range(700_000, 700_000 + a.frames_episodes))
        per_style = {}
        for style in ("a", "b", "c"):
            sub = seeds[{"a": 0, "b": 1, "c": 2}[style]::3]
            res = pool.map(perception_worker,
                           [(c, style) for c in chunks(sub, a.workers) if c])
            per_style[style] = res
            print(f"style {style}: {time.time() - t0:.0f} s", flush=True)
        for key in ("full", "no_tracking", "no_gates"):
            rep["perception"][key] = merge(
                [r[key] for st in per_style.values() for r in st])
        rep["perception"]["full_style_c_only"] = merge(
            [r["full"] for r in per_style["c"]])
        p = rep["perception"]
        for key in ("full", "no_tracking", "no_gates", "full_style_c_only"):
            s = p[key]["stack"]["all"]
            print(f"{key:18s} stack P {100 * s['precision']:.1f} "
                  f"R {100 * s['recall']:.1f} F1 {100 * s['f1']:.1f}")
        s = p["full"]["raw"]["all"]
        print(f"{'raw detector':18s}       P {100 * s['precision']:.1f} "
              f"R {100 * s['recall']:.1f} mAP50 {100 * s['mAP50']:.1f}")
        print("distance MAE", p["full"]["range_mae_m"], "unknown recall",
              p["full"]["unknown_recall"], "frames", p["full"]["frames"])

        # ---- navigation -------------------------------------------------
        seeds = list(range(a.seed0, a.seed0 + a.episodes))
        rep["navigation"] = {}
        for name, real in (("real_image_camera", True), ("emulated", False)):
            res = pool.map(nav_worker,
                           [(c, real) for c in chunks(seeds, a.workers) if c])
            runs = sorted([r for part, _ in res for r in part],
                          key=lambda r: r["seed"])
            rep["navigation"][name] = nav_summary(runs)
            rep["navigation"][name]["per_episode"] = runs
            if real:
                rep["navigation"]["in_loop_perception"] = merge(
                    [m for _, m in res])
            s = rep["navigation"][name]
            print(f"{name:18s} success {100 * s['success_rate']:.0f}% "
                  f"collision {100 * s['collision_rate']:.0f}% timeout "
                  f"{100 * s['timeout_rate']:.0f}%  ({time.time() - t0:.0f}"
                  f" s)", flush=True)
    rep["episodes"] = a.episodes
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
