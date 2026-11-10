"""
Evaluate navigation in simulation (project testing plan, Sections 15.2/15.3).

1. RL vs rules          : PPO, PPO + shield, rule-based, rule-based + shield
2. Perception ablation  : detector-only -> + depth fusion -> + uncertainty
                          -> + tracking (full system), with PPO + shield
3. Safety-shield tests  : scripted "worker steps out" scenes
4. Robustness           : PPO seeds 0/1/2 (if trained) -> mean +- std

All numbers are SIMULATION results on held-out random warehouse layouts
(seeds never used during training).

    python -m scripts.evaluate_navigation --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.evaluation.run_episodes import CONTROLLERS, evaluate
from src.navigation.controllers import NumpyPolicy
from src.safety.shield import SafetyShield
from src.simulation.warehouse_env import (
    EnvConfig, Obstacle, PERSON, WarehouseEnv,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics"


def fmt(x, pct=True):
    if x is None:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.1f}"


def shield_scenarios(n=50, policy=None):
    """A worker steps out in front of the robot at ~1 m.

    The robot drives straight down an empty aisle; at a random moment a
    worker appears 0.9-1.3 m ahead. The shield must stop in 100% of cases.
    """
    results = []
    for k in range(n):
        env = WarehouseEnv(EnvConfig(n_workers=(0, 0), n_forklifts=(0, 0),
                                     n_pallets=(0, 0), n_boxes=(0, 0)))
        obs, _ = env.reset(seed=50_000 + k)
        shield = SafetyShield()
        rng = np.random.default_rng(k)
        trigger = int(rng.integers(8, 20))
        stepped_out = False
        collided = False
        for t in range(120):
            # the policy that is being overridden always says GO
            a = 0
            if policy is not None:
                a = policy.act(obs, env)
            final, cap, reason = shield.filter(env, a)
            obs, r, term, trunc, info = env.step(final)
            if t == trigger and not stepped_out:
                d = float(rng.uniform(0.9, 1.3)) + env.cfg.robot_radius
                px = env.x + d * math.cos(env.theta)
                py = env.y + d * math.sin(env.theta)
                env.statics.append(Obstacle(PERSON, px, py, radius=0.28,
                                            uid=999))
                stepped_out = True
                env._perceive()
                obs = env._observation()
            if env.collided:
                collided = True
                break
            if term or trunc:
                break
        results.append(not collided)
    return float(np.mean(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"note": "Simulation results on held-out random layouts. "
                      "Not physical-robot results.",
              "episodes": args.episodes}

    # 1. RL vs rules -------------------------------------------------------
    print("== Controllers (full perception) ==")
    report["controllers"] = {}
    for name in CONTROLLERS:
        s, _ = evaluate(name, episodes=args.episodes)
        report["controllers"][name] = s
        print(f"{CONTROLLERS[name]:42s} success {fmt(s['success_rate'])}"
              f"  collision {fmt(s['collision_rate'])}"
              f"  timeout {fmt(s['timeout_rate'])}"
              f"  t_goal {fmt(s['mean_time_to_goal_s'], False)} s")

    # 2. Perception ablation ----------------------------------------------
    print("\n== Perception ablation (PPO + shield) ==")
    report["ablation"] = {}
    for mode in ["detector_only", "fusion", "fusion_uncertainty", "full"]:
        s, _ = evaluate("ppo_shield", episodes=args.episodes,
                        perception=mode)
        report["ablation"][mode] = s
        print(f"{mode:20s} success {fmt(s['success_rate'])}"
              f"  collision {fmt(s['collision_rate'])}"
              f"  by class {s['collisions_by_class']}")

    # Reflective-heavy scenes: uncertainty slowdown matters most here
    print("\n== Reflective hard-case scenes (all pallets stretch-wrapped) ==")
    report["reflective"] = {}
    for mode in ["fusion", "fusion_uncertainty", "full"]:
        s, _ = evaluate("ppo_shield", episodes=args.episodes,
                        perception=mode,
                        options={"reflective_prob": 1.0,
                                 "counts": {"pallets": 5, "boxes": 1,
                                            "workers": 2, "forklifts": 0}})
        report["reflective"][mode] = s
        print(f"{mode:20s} success {fmt(s['success_rate'])}"
              f"  collision {fmt(s['collision_rate'])}"
              f"  by class {s['collisions_by_class']}")

    # 3. Safety shield scripted test ---------------------------------------
    print("\n== Safety shield: worker steps out at ~1 m ==")
    ok = shield_scenarios(50)
    report["shield_worker_step_out_stop_rate"] = ok
    print(f"Shield prevented collision in {100 * ok:.0f}% of 50 scenes")

    # 4. Seeds -------------------------------------------------------------
    seeds = sorted((ROOT / "models").glob("ppo_warehouse_seed*.npz"))
    if seeds:
        print("\n== PPO + shield across training seeds ==")
        per = {}
        for p in seeds:
            s, _ = evaluate("ppo_shield", episodes=args.episodes,
                            policy_path=p)
            per[p.stem] = s
            print(f"{p.stem:24s} success {fmt(s['success_rate'])}"
                  f"  collision {fmt(s['collision_rate'])}")
        sr = [v["success_rate"] for v in per.values()]
        cr = [v["collision_rate"] for v in per.values()]
        report["seeds"] = {
            "per_seed": per,
            "success_mean": float(np.mean(sr)),
            "success_std": float(np.std(sr)),
            "collision_mean": float(np.mean(cr)),
            "collision_std": float(np.std(cr)),
        }
        print(f"mean +- std: success {100*np.mean(sr):.1f} +- "
              f"{100*np.std(sr):.1f}%, collision {100*np.mean(cr):.1f} +- "
              f"{100*np.std(cr):.1f}%")

    path = OUT / "navigation_eval.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
