"""
Record backup demo videos/GIFs of the trained robot (for the presentation).

    python -m scripts.record_demo --episodes 3

Writes results/demo/demo_<scenario>.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.app.sim_session import SCENARIOS, SimSession

ROOT = Path(__file__).resolve().parents[1]


def record(scenario, layout_seed, controller="ppo_shield", max_frames=450,
           every=2, scale=0.5):
    sim = SimSession()
    sim.controller = controller
    sim.scenario = scenario
    sim.layout_seed = sim.episode_seed = layout_seed
    sim.reset()
    frames = []
    while not sim.done and sim.env.steps < max_frames:
        sim.step()
        if sim.env.steps % every == 0 or sim.done:
            f = sim.frame()
            im = Image.fromarray(f)
            im = im.resize((int(im.width * scale), int(im.height * scale)),
                           Image.BILINEAR)
            frames.append(im)
    for _ in range(8):
        frames.append(frames[-1])
    return frames, sim.outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "demo"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [
        ("Random warehouse", 12, "random"),
        ("Worker steps out in front of the robot", 11, "worker_step_out"),
        ("Fallen boxes (unknown objects) in the aisles", 21, "unknown_boxes"),
        ("Shiny stretch-wrapped pallets (reflective)", 5, "reflective"),
    ]
    for scenario, seed, tag in plan:
        assert scenario in SCENARIOS
        frames, outcome = record(scenario, seed)
        p = out / f"demo_{tag}.gif"
        frames[0].save(p, save_all=True, append_images=frames[1:],
                       duration=100, loop=0, optimize=True)
        print(f"{p.name}: {outcome} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
