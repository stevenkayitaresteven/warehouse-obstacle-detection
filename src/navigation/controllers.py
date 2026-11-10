"""
Navigation controllers:

* ``RuleBasedController`` - the simple baseline from the testing plan
  ("stop if an obstacle is closer than 1 m", otherwise steer to the path).
* ``NumpyPolicy``         - the trained PPO actor (Stable-Baselines3 MLP)
  exported to NumPy, so the demo runs without PyTorch.
* ``SB3Policy``           - thin wrapper around a Stable-Baselines3 model.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.simulation.warehouse_env import (
    GO, SLOW, STOP, TURN_LEFT, TURN_RIGHT, wrap_angle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "models" / "ppo_warehouse_policy.npz"


class RuleBasedController:
    """Hand-written reactive baseline."""

    name = "Rule-based baseline"

    def __init__(self, stop_dist=1.0):
        self.stop_dist = stop_dist

    def act(self, obs, env):
        c = env.cfg
        d = env.fused_rays
        a = env.ray_angles
        half_w = c.robot_radius + 0.1
        lateral = np.abs(d * np.sin(a))
        forward = d * np.cos(a)
        inside = (lateral < half_w) & (forward > 0)
        front = forward[inside].min() - c.robot_radius if inside.any() \
            else math.inf

        cx, cy = env.carrot()
        err = wrap_angle(math.atan2(cy - env.y, cx - env.x) - env.theta)

        if front < self.stop_dist:
            left = d[a > 0.2].mean()
            right = d[a < -0.2].mean()
            if max(left, right) < 0.8:
                return STOP
            return TURN_LEFT if left > right else TURN_RIGHT
        if abs(err) > math.radians(25):
            return TURN_LEFT if err > 0 else TURN_RIGHT
        if front < 2.0:
            return SLOW
        return GO


class NumpyPolicy:
    """PPO actor network evaluated with NumPy (deterministic argmax)."""

    name = "PPO (RL)"

    def __init__(self, path=DEFAULT_POLICY):
        data = np.load(path)
        n = int(data["n_layers"])
        self.weights = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        self.out = (data["W_out"], data["b_out"])
        self.activation = str(data["activation"]) if "activation" in data \
            else "tanh"

    def logits(self, obs):
        h = np.asarray(obs, dtype=np.float32)
        for W, b in self.weights:
            h = h @ W.T + b
            h = np.tanh(h) if self.activation == "tanh" else np.maximum(h, 0)
        W, b = self.out
        return h @ W.T + b

    def probs(self, obs):
        z = self.logits(obs)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, obs, env=None):
        return int(np.argmax(self.logits(obs)))


class SB3Policy:
    name = "PPO (RL)"

    def __init__(self, model):
        self.model = model

    def act(self, obs, env=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def export_sb3_to_numpy(model, path):
    """Export the actor of an SB3 PPO MlpPolicy to a .npz file."""
    import torch.nn as nn

    policy = model.policy
    layers = [m for m in policy.mlp_extractor.policy_net
              if isinstance(m, nn.Linear)]
    acts = [m for m in policy.mlp_extractor.policy_net
            if not isinstance(m, nn.Linear)]
    activation = "tanh" if (acts and isinstance(acts[0], nn.Tanh)) else "relu"
    arrays = {"n_layers": np.array(len(layers)),
              "activation": np.array(activation)}
    for i, layer in enumerate(layers):
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    arrays["W_out"] = policy.action_net.weight.detach().cpu().numpy()
    arrays["b_out"] = policy.action_net.bias.detach().cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
