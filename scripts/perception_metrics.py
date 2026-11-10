"""
Perception metrics in simulation:

1. Confusion matrix: true obstacle class vs. label produced by the
   perception stack (per obstacle, per frame, objects within 5.5 m that at
   least one camera ray hits). Compared for "detector only" vs "full"
   (detector + depth fusion + uncertainty + tracking).
2. Uncertainty score: normal pallets vs. stretch-wrapped (reflective)
   pallets in front of the robot, and how often the slowdown triggers.
3. Speed: perception + policy + shield time per frame (FPS) on this CPU.

    python -m scripts.perception_metrics
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.navigation.controllers import NumpyPolicy
from src.safety.shield import SafetyShield
from src.simulation.warehouse_env import (
    BOX, FORKLIFT, PALLET, PERSON, EnvConfig, WarehouseEnv,
)

ROOT = Path(__file__).resolve().parents[1]
TRUE = [PERSON, PALLET, FORKLIFT, BOX]
TRUE_NAMES = ["person", "pallet", "forklift", "unknown box"]
PRED_NAMES = ["person", "pallet", "forklift", "unknown obstacle", "missed"]


def pred_index(label):
    return {PERSON: 0, PALLET: 1, FORKLIFT: 2, BOX: 3}.get(int(label), 4)


def confusion(mode, episodes=40, seed0=30_000):
    env = WarehouseEnv(EnvConfig(perception_mode=mode))
    pol, shield = NumpyPolicy(), SafetyShield()
    M = np.zeros((4, 5), dtype=int)
    for k in range(episodes):
        obs, _ = env.reset(seed=seed0 + k)
        shield.reset()
        done = False
        while not done:
            per_obj = {}
            for d, lab, hit in zip(env.true_rays, env.ray_labels,
                                   env.ray_hits):
                if hit is None or hit.category not in TRUE or d > 5.5:
                    continue
                per_obj.setdefault(hit.uid, (hit.category, []))[1].append(
                    pred_index(lab))
            for cat, preds in per_obj.values():
                p = Counter(preds).most_common(1)[0][0]
                M[TRUE.index(cat), p] += 1
            a = pol.act(obs, env)
            a, _, _ = shield.filter(env, a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return M


def uncertainty(episodes=30, seed0=40_000):
    out = {}
    for name, refl in (("normal pallets", 0.0), ("reflective pallets", 1.0)):
        env = WarehouseEnv(EnvConfig())
        pol, shield = NumpyPolicy(), SafetyShield()
        us = []
        for k in range(episodes):
            obs, _ = env.reset(seed=seed0 + k, options={
                "reflective_prob": refl,
                "counts": {"pallets": 5, "boxes": 1, "workers": 1,
                           "forklifts": 0}})
            shield.reset()
            done = False
            while not done:
                mid = slice(7, 14)
                near_pallet = any(
                    h is not None and h.category == PALLET
                    for h in env.ray_hits[mid]) and \
                    env.true_rays[mid].min() < 3.0
                if near_pallet:
                    us.append(env.percep.uncertainty)
                a = pol.act(obs, env)
                a, _, _ = shield.filter(env, a)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
        us = np.array(us)
        out[name] = {
            "frames": int(len(us)),
            "mean": float(us.mean()),
            "slow_rate": float((us >= 0.35).mean()),
            "stop_rate": float((us >= 0.75).mean()),
            "hist": np.histogram(us, bins=10, range=(0, 1))[0].tolist(),
        }
    return out


def speed(frames=2000):
    env = WarehouseEnv(EnvConfig())
    pol, shield = NumpyPolicy(), SafetyShield()
    obs, _ = env.reset(seed=1)
    t_perc = t_pol = 0.0
    for _ in range(frames):
        t0 = time.perf_counter()
        a = pol.act(obs, env)
        a, _, _ = shield.filter(env, a)
        t1 = time.perf_counter()
        obs, _, term, trunc, _ = env.step(a)   # physics + perception
        t2 = time.perf_counter()
        t_pol += t1 - t0
        t_perc += t2 - t1
        if term or trunc:
            obs, _ = env.reset()
    return {"policy_plus_shield_ms": 1000 * t_pol / frames,
            "sim_plus_perception_ms": 1000 * t_perc / frames,
            "fps": frames / (t_pol + t_perc)}


def main():
    rep = {"note": "Simulation only. Detector branch is simulated "
                   "(recall 80-97 %), not the trained YOLO model."}
    for mode in ("detector_only", "full"):
        M = confusion(mode)
        rep[f"confusion_{mode}"] = {"rows_true": TRUE_NAMES,
                                    "cols_pred": PRED_NAMES,
                                    "matrix": M.tolist()}
        print(mode)
        print(M)
        found = 1 - M[:, 4].sum() / M.sum()
        print(f"  obstacle recall (anything found): {100 * found:.1f}%")
        rep[f"obstacle_recall_{mode}"] = float(found)
        rep[f"unknown_box_recall_{mode}"] = float(1 - M[3, 4] / M[3].sum())
    rep["uncertainty"] = uncertainty()
    print(json.dumps(rep["uncertainty"], indent=1))
    rep["speed"] = speed()
    print(rep["speed"])
    p = ROOT / "results" / "metrics" / "perception_eval.json"
    p.write_text(json.dumps(rep, indent=2))
    print("saved", p)


if __name__ == "__main__":
    main()
