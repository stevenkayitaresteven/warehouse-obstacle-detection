"""
A stateful simulation session used by the interactive demo.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from src.navigation.controllers import (
    DEFAULT_POLICY, NumpyPolicy, RuleBasedController,
)
from src.safety.shield import SafetyShield
from src.simulation.render import MAP_PAD, SCALE, render_frame
from src.simulation.warehouse_env import (
    ACTIONS, BOX, EnvConfig, Obstacle, PALLET, PERSON, WarehouseEnv,
)

CONTROLLER_CHOICES = [
    ("PPO + Safety Shield (full system)", "ppo_shield"),
    ("PPO only (no shield)", "ppo"),
    ("Rule-based baseline (stop if < 1 m)", "rule"),
]

PERCEPTION_CHOICES = [
    ("Full: detector + depth + uncertainty + tracking", "full"),
    ("Detector + depth fusion + uncertainty", "fusion_uncertainty"),
    ("Detector + depth fusion", "fusion"),
    ("Detector only (existing method)", "detector_only"),
]

SCENARIOS = {
    "Random warehouse": None,
    "Worker steps out in front of the robot": {
        "workers": 0, "forklifts": 0, "pallets": 1, "boxes": 0,
        "step_out": True},
    "Fallen boxes (unknown objects) in the aisles": {
        "workers": 1, "forklifts": 0, "pallets": 1, "boxes": 6},
    "Shiny stretch-wrapped pallets (reflective)": {
        "workers": 1, "forklifts": 0, "pallets": 6, "boxes": 1,
        "reflective": 1.0},
    "Busy shift: many workers + forklifts": {
        "workers": 6, "forklifts": 2, "pallets": 2, "boxes": 2},
}

_policy_cache = {}


def load_policy():
    if "ppo" not in _policy_cache:
        _policy_cache["ppo"] = NumpyPolicy(DEFAULT_POLICY)
    return _policy_cache["ppo"]


class SimSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.env = WarehouseEnv(EnvConfig())
        self.controller = "ppo_shield"
        self.perception = "full"
        self.layout_seed = 7
        self.episode_seed = 7
        self.scenario = "Random warehouse"
        self.counts = None
        self.reflective = None
        self.running = False
        self.reset()

    # ------------------------------------------------------------------
    def _policy(self):
        if self.controller.startswith("ppo"):
            return load_policy()
        return RuleBasedController()

    def reset(self, new_layout=False):
        with self.lock:
            if new_layout:
                self.layout_seed = int(np.random.randint(0, 100_000))
                self.episode_seed = self.layout_seed
            opts = {"layout_seed": self.layout_seed,
                    "perception_mode": self.perception}
            sc = SCENARIOS.get(self.scenario)
            counts = dict(self.counts or {})
            refl = self.reflective
            if sc:
                for k in ("workers", "forklifts", "pallets", "boxes"):
                    counts[k] = sc[k]
                if "reflective" in sc:
                    refl = sc["reflective"]
            if counts:
                opts["counts"] = counts
            if refl is not None:
                opts["reflective_prob"] = refl
            self.obs, _ = self.env.reset(seed=self.episode_seed,
                                         options=opts)
            self.shield = SafetyShield()
            self.shield.enabled = self.controller.endswith("shield")
            self.trail = [(self.env.x, self.env.y)]
            self.ret = 0.0
            self.last_reward = 0.0
            self.decision = None
            self.done = False
            self.outcome = None
            self.step_out_done = False

    # ------------------------------------------------------------------
    def step(self):
        with self.lock:
            if self.done:
                return
            env = self.env
            a = self._policy().act(self.obs, env)
            final, cap, reason = self.shield.filter(env, a)
            if not self.shield.enabled:
                reason = "Shield disabled"
            self.decision = (ACTIONS[a], ACTIONS[final], reason)
            self.obs, r, term, trunc, info = env.step(final)
            self.ret += r
            self.last_reward = r
            self.trail.append((env.x, env.y))
            sc = SCENARIOS.get(self.scenario)
            if sc and sc.get("step_out") and not self.step_out_done and \
                    env.steps >= 18 and env.v > 0.8:
                # a worker steps out from beside the robot's path and walks
                # across the aisle in front of it
                d = 1.3 + env.cfg.robot_radius
                side = 1 if (self.layout_seed % 2) else -1
                nx_, ny_ = -math.sin(env.theta), math.cos(env.theta)
                px = env.x + d * math.cos(env.theta) + side * 0.55 * nx_
                py = env.y + d * math.sin(env.theta) + side * 0.55 * ny_
                speed = 0.7
                horiz = abs(nx_) > abs(ny_)
                worker = Obstacle(
                    PERSON, px, py, radius=0.28, uid=9999, user_placed=True,
                    vx=(-side * speed * math.copysign(1, nx_)
                        if horiz else 0.0),
                    vy=(0.0 if horiz else
                        -side * speed * math.copysign(1, ny_)),
                    axis=0 if horiz else 1,
                    bounds=((0.6, env.cfg.width - 0.6) if horiz
                            else (0.6, env.cfg.height - 0.6)))
                env.dynamics.append(worker)
                self.step_out_done = True
            if term or trunc:
                self.done = True
                if info["reached"]:
                    self.outcome = (f"✅ Goal reached in "
                                    f"{env.steps * env.cfg.dt:.1f} s")
                elif info["collided"]:
                    self.outcome = (f"💥 Collision with "
                                    f"{info['collision_with']}")
                else:
                    self.outcome = "⏱ Time limit reached"

    # ------------------------------------------------------------------
    def frame(self):
        with self.lock:
            name = dict((v, k) for k, v in CONTROLLER_CHOICES)[
                self.controller]
            return render_frame(
                self.env, decision=self.decision, controller=name,
                reward=self.last_reward,
                totals={"return": self.ret,
                        "interventions": self.shield.interventions},
                trail=self.trail,
            )

    def status_md(self):
        env = self.env
        lines = []
        if self.outcome:
            lines.append(f"### {self.outcome}")
        else:
            lines.append("### 🚚 Robot driving…" if self.running
                         else "### ⏸ Paused")
        lines.append(
            f"**Time** {env.steps * env.cfg.dt:.1f} s · "
            f"**Speed** {env.v:.2f} m/s · "
            f"**Uncertainty** {env.percep.uncertainty:.2f} · "
            f"**Shield interventions** {self.shield.interventions} · "
            f"**Near-misses** {env.near_misses} · "
            f"**Layout** #{self.layout_seed}")
        return "\n\n".join(lines)

    def log_rows(self, n=12):
        rows = []
        for rec in self.shield.log[-200:]:
            if rec.reason.startswith("No rule") and rec.proposed == rec.final:
                continue
            rows.append([round(rec.step * self.env.cfg.dt, 1), rec.proposed,
                         rec.final, rec.reason])
        rows = rows[-n:][::-1]
        if not rows:
            rows = [["-", "-", "-", "No overrides yet"]]
        return rows

    # ------------------------------------------------------------------
    def click(self, px, py, mode):
        """Map click (pixel coords in the rendered frame)."""
        env = self.env
        x = (px - MAP_PAD) / SCALE
        y = env.cfg.height - (py - MAP_PAD) / SCALE
        if not (0.3 < x < env.cfg.width - 0.3 and
                0.3 < y < env.cfg.height - 0.3):
            return "Click inside the warehouse map to place obstacles."
        with self.lock:
            if mode.startswith("Remove"):
                ok = env.remove_obstacle_near(x, y)
                msg = "Removed obstacle." if ok else "Nothing to remove here."
            else:
                cat = {"Unknown box": BOX, "Worker": PERSON,
                       "Shiny pallet": PALLET}[mode.split(" (")[0]]
                ok = env.add_obstacle(cat, x, y, reflective=(cat == PALLET))
                msg = (f"Placed {mode.split(' (')[0].lower()} at "
                       f"({x:.1f} m, {y:.1f} m)." if ok
                       else "Too close to the robot.")
            env._perceive()
            self.obs = env._observation()
        return msg
