"""
Shared episode runner used by the evaluation script and the demo.
"""

from __future__ import annotations

import numpy as np

from src.navigation.controllers import NumpyPolicy, RuleBasedController
from src.safety.shield import SafetyShield
from src.simulation.warehouse_env import EnvConfig, WarehouseEnv

CONTROLLERS = {
    "ppo_shield": "PPO + safety shield (full system)",
    "ppo": "PPO only (no shield)",
    "rule": "Rule-based baseline (stop if < 1 m)",
    "rule_shield": "Rule-based baseline + safety shield",
}


def build_controller(name, policy_path=None):
    if name.startswith("ppo"):
        pol = NumpyPolicy(policy_path) if policy_path else NumpyPolicy()
    else:
        pol = RuleBasedController()
    shield = SafetyShield()
    shield.enabled = name.endswith("shield")
    return pol, shield


def run_episode(env, policy, shield, seed, options=None, frames=None,
                render_every=1):
    obs, info = env.reset(seed=seed, options=options)
    shield.reset()
    done = False
    total = 0.0
    while not done:
        a = policy.act(obs, env)
        final, cap, reason = shield.filter(env, a)
        obs, r, term, trunc, info = env.step(final)
        total += r
        done = term or trunc
        if frames is not None and env.steps % render_every == 0:
            frames.append(env.render())
    return {
        "reached": bool(info["reached"]),
        "collided": bool(info["collided"]),
        "collision_with": info["collision_with"],
        "timeout": bool(trunc and not term),
        "steps": int(info["steps"]),
        "time_s": info["steps"] * env.cfg.dt,
        "near_misses": int(info["near_misses"]),
        "interventions": int(shield.interventions),
        "return": float(total),
        "min_clearance": float(env.min_clearance),
        "path_length": float(info["path_length"]),
    }


def summarise(results):
    res = list(results)
    n = len(res)
    succ = [r for r in res if r["reached"]]
    out = {
        "episodes": n,
        "success_rate": sum(r["reached"] for r in res) / n,
        "collision_rate": sum(r["collided"] for r in res) / n,
        "timeout_rate": sum(r["timeout"] for r in res) / n,
        "mean_time_to_goal_s": (float(np.mean([r["time_s"] for r in succ]))
                                if succ else None),
        "near_misses_per_episode": float(np.mean([r["near_misses"]
                                                  for r in res])),
        "shield_interventions_per_episode": float(np.mean(
            [r["interventions"] for r in res])),
        "mean_return": float(np.mean([r["return"] for r in res])),
    }
    coll = {}
    for r in res:
        if r["collided"]:
            coll[r["collision_with"]] = coll.get(r["collision_with"], 0) + 1
    out["collisions_by_class"] = coll
    return out


def evaluate(controller, episodes=100, seed0=10_000, perception="full",
             policy_path=None, cfg_kwargs=None, options=None):
    env = WarehouseEnv(EnvConfig(perception_mode=perception,
                                 **(cfg_kwargs or {})))
    pol, shield = build_controller(controller, policy_path)
    res = [run_episode(env, pol, shield, seed0 + k, options=options)
           for k in range(episodes)]
    return summarise(res), res
