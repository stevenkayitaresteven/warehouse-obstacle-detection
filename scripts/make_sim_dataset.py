"""
Render labelled frames from the real-image camera simulation.

Two uses:

* **Fine-tuning data** ("cut, paste and learn"): frames whose objects are
  cut-outs from the *train* split only. Mixed with the original real
  training photos they teach the detector the robot's viewpoint (people,
  forklifts and parcels seen from 1 m height inside a rack aisle).
* **Held-out evaluation data**: frames whose objects come from the *val /
  test* splits (real people / forklifts / parcels the detector has never
  seen) and, optionally, a warehouse style that was not used for training.

Camera poses come from real driving (the PPO policy with the safety shield
drives the robot through random warehouses) mixed with random poses.

    python -m scripts.make_sim_dataset --out datasets/processed/simcam \\
        --split train --frames 2400 --sprites train --styles a,b
    python -m scripts.make_sim_dataset --out datasets/processed/simcam \\
        --split test --frames 600 --sprites val,test --styles a,b,c \\
        --seed0 900000

Writes YOLO-format labels (0 person, 1 forklift, 2 package; unknown
objects stay unlabelled on purpose) and a JSON with the full ground truth
(distance, visibility) per frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.navigation.controllers import NumpyPolicy
from src.safety.shield import SafetyShield
from src.simulation.photo_camera import PhotoCamera
from src.simulation.warehouse_env import EnvConfig, WarehouseEnv

CLASS_ID = {"person": 0, "forklift": 1, "package": 2}
MIN_VISIBLE = 0.25       # label an object if at least this much is visible
MIN_SIDE = 6             # ... and it is at least this many pixels


def yolo_lines(gt, w, h):
    lines = []
    for g in gt:
        if g["class_name"] not in CLASS_ID or g["visible"] < MIN_VISIBLE:
            continue
        x0, y0, x1, y1 = g["bbox"]
        if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE + 4:
            continue
        lines.append(f"{CLASS_ID[g['class_name']]} {(x0 + x1) / 2 / w:.6f} "
                     f"{(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} "
                     f"{(y1 - y0) / h:.6f}")
    return lines


def random_counts(rng):
    return {"workers": int(rng.integers(1, 7)),
            "forklifts": int(rng.integers(0, 4)),
            "pallets": int(rng.integers(1, 7)),
            "boxes": int(rng.integers(0, 5))}


def random_pose(env, rng):
    """Teleport the robot to a free spot looking at something."""
    for _ in range(50):
        p = env._free_point(rng, (0.8, env.cfg.width - 0.8), clearance=0.6)
        if p is None:
            continue
        env.x, env.y = p
        objs = env.statics + env.dynamics
        if objs and rng.random() < 0.75:      # face a nearby object
            d = [math.hypot(o.x - env.x, o.y - env.y) for o in objs]
            near = [o for o, dd in zip(objs, d) if 1.0 < dd < 9.0]
            if near:
                o = near[int(rng.integers(len(near)))]
                env.theta = math.atan2(o.y - env.y, o.x - env.x) + float(
                    rng.uniform(-0.7, 0.7))
                return True
        env.theta = float(rng.uniform(-math.pi, math.pi))
        return True
    return False


def frames(n, seed0, sprites, styles, every=3, random_frac=0.35,
           per_episode=24, keep_empty=0.35, progress=None):
    """Yield (frame, meta) for `n` frames.

    A frame is kept only if the robot moved (0.3 m) or turned (12 degrees)
    since the last kept frame, at most `per_episode` per warehouse, and
    most frames without any object are dropped.
    """
    cam = PhotoCamera(splits=sprites, styles=styles)
    env = WarehouseEnv(EnvConfig())
    pol, shield = NumpyPolicy(), SafetyShield()
    done, ep = 0, 0

    def keep(fr, rng):
        return bool(fr.gt) or rng.random() < keep_empty

    while done < n:
        seed = seed0 + ep
        ep += 1
        rng = np.random.default_rng([seed, 5])
        obs, _ = env.reset(seed=seed, options={
            "counts": random_counts(rng),
            "reflective_prob": float(rng.uniform(0.0, 0.8))})
        shield.reset()
        cam.reset(env, seed)
        if rng.random() < random_frac:
            # a handful of random viewpoints in this warehouse
            for k in range(8):
                if done >= n or not random_pose(env, rng):
                    break
                env.omega = float(rng.choice([0.0, 0.0, 1.4, -1.4]))
                for _ in range(int(rng.integers(0, 6))):
                    env._move_dynamics()
                fr = cam.render(env, rng)
                if not keep(fr, rng):
                    continue
                yield fr, {"seed": seed, "k": k, "style": cam.style,
                           "mode": "random"}
                done += 1
        else:
            t, kept, last, idle = 0, 0, None, 0
            while done < n and kept < per_episode and idle < 40:
                a = pol.act(obs, env)
                a, _, _ = shield.filter(env, a)
                obs, _, term, trunc, _ = env.step(a)
                t += 1
                idle += 1
                moved = last is None or math.hypot(
                    env.x - last[0], env.y - last[1]) > 0.3 or abs(
                    math.remainder(env.theta - last[2], math.tau)) > 0.21
                if t % every == 0 and moved:
                    fr = cam.render(env, rng)
                    last = (env.x, env.y, env.theta)
                    if keep(fr, rng):
                        yield fr, {"seed": seed, "k": t, "style": cam.style,
                                   "mode": "drive"}
                        done += 1
                        kept += 1
                        idle = 0
                if term or trunc:
                    break
        if progress:
            progress(done, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/processed/simcam")
    ap.add_argument("--split", default="train")
    ap.add_argument("--frames", type=int, default=2400)
    ap.add_argument("--sprites", default="train")
    ap.add_argument("--styles", default="a,b")
    ap.add_argument("--seed0", type=int, default=500_000)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images" / a.split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / a.split).mkdir(parents=True, exist_ok=True)
    meta_all, inst = [], {k: 0 for k in CLASS_ID}
    gen = frames(a.frames, a.seed0, tuple(a.sprites.split(",")),
                 tuple(a.styles.split(",")), a.every,
                 progress=lambda d, n: print(f"\r{d}/{n}", end="",
                                             flush=True))
    for i, (fr, meta) in enumerate(gen):
        name = f"sim_{a.split}_{i:05d}"
        h, w = fr.rgb.shape[:2]
        cv2.imwrite(str(out / "images" / a.split / f"{name}.jpg"),
                    fr.rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        lines = yolo_lines(fr.gt, w, h)
        (out / "labels" / a.split / f"{name}.txt").write_text(
            "\n".join(lines))
        for line in lines:
            inst[list(CLASS_ID)[int(line.split()[0])]] += 1
        meta_all.append(dict(meta, name=name, gt=[
            {k: v for k, v in g.items() if not k.startswith("_")}
            for g in fr.gt]))
    (out / f"gt_{a.split}.json").write_text(json.dumps(meta_all))
    print(f"\n{a.split}: {len(meta_all)} frames, instances {inst}, "
          f"sprites {a.sprites}, styles {a.styles}")


if __name__ == "__main__":
    main()
