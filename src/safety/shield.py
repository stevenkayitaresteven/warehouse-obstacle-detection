"""
Rule-based safety shield and decision log (project Section 7.4).

Every action proposed by the RL policy passes through the shield before it
reaches the motors. Rules are checked in priority order:

1. Hard stop        - any obstacle (known or unknown) inside the stopping
                      envelope of the robot's path -> STOP.
2. Person priority  - people get a larger safety margin (stop / slow).
3. Uncertainty      - high corridor uncertainty caps speed or forces STOP.
4. Otherwise        - the RL action is allowed.

Turning is still allowed when the shield stops the robot: the turn is
executed *in place* (speed cap 0), so the robot can never deadlock in
front of a static obstacle and never drives forward into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.simulation.warehouse_env import (
    ACTIONS, GO, SLOW, STOP, TURN_LEFT, TURN_RIGHT, PERSON,
)


@dataclass
class ShieldConfig:
    base_stop_dist: float = 0.35     # clearance kept in front of the robot
    reaction_time: float = 0.25      # s
    person_stop_dist: float = 1.2
    person_slow_dist: float = 2.5
    person_slow_speed: float = 0.6   # m/s
    unc_slow: float = 0.35
    unc_stop: float = 0.75
    path_half_width_margin: float = 0.12


@dataclass
class DecisionRecord:
    step: int
    proposed: str
    final: str
    reason: str
    speed_cap: float | None


@dataclass
class SafetyShield:
    config: ShieldConfig = field(default_factory=ShieldConfig)
    log: list = field(default_factory=list)
    interventions: int = 0
    enabled: bool = True
    memory: int = 10
    recovery_after: int = 5      # blocked steps before a recovery turn

    def __post_init__(self):
        self._memory = []
        self._blocked = 0

    def reset(self):
        self.log = []
        self.interventions = 0
        self._memory = []
        self._blocked = 0

    # ------------------------------------------------------------------
    def _hit_points(self, env):
        """Obstacle points in the robot frame (x forward, y left).

        Current fused returns plus a short memory (last ``memory`` frames)
        of earlier returns, so obstacles that left the camera's field of
        view (e.g. beside the robot while it turns) are still respected.
        """
        c = env.cfg
        d = env.fused_rays
        m = d < c.max_range - 0.05
        a = env.theta + env.ray_angles[m]
        world = np.stack([env.x + d[m] * np.cos(a),
                          env.y + d[m] * np.sin(a)], axis=1)
        self._memory.append(world)
        if len(self._memory) > self.memory:
            self._memory.pop(0)
        pts = np.concatenate(self._memory, axis=0)
        if len(pts) == 0:
            return pts
        dx, dy = pts[:, 0] - env.x, pts[:, 1] - env.y
        ct, st = math.cos(env.theta), math.sin(env.theta)
        return np.stack([ct * dx + st * dy, -st * dx + ct * dy], axis=1)

    def _path_obstacle_distance(self, env, v=None, omega=0.0):
        """Free travel distance along the predicted arc of the robot.

        The robot (a disc) is swept along the arc implied by (v, omega);
        the first arc length at which the disc (plus a margin) gets closer
        to an obstacle point than allowed is returned. Motions that move
        the robot *away* from a close obstacle are always allowed.
        """
        c = env.cfg
        pts = self._hit_points(env)
        if len(pts) == 0:
            return math.inf
        half_w = c.robot_radius + self.config.path_half_width_margin
        v = max(env.v if v is None else v, 0.3)
        s = np.linspace(0.0, 4.0, 41)          # arc length samples (m)
        kappa = omega / v
        if abs(kappa) < 1e-3:
            px, py, = s, np.zeros_like(s)
        else:
            px = np.sin(kappa * s) / kappa
            py = (1 - np.cos(kappa * s)) / kappa
        dx = pts[None, :, 0] - px[:, None]
        dy = pts[None, :, 1] - py[:, None]
        dist = np.hypot(dx, dy)                 # (S, P)
        thresh = np.minimum(half_w, dist[0] - 0.01)
        hit = (dist[1:] < thresh[None, :]).any(axis=1)
        if not hit.any():
            return math.inf
        return float(s[1:][np.argmax(hit)])

    def _person_distance(self, env):
        """Nearest person anywhere in the camera FOV."""
        m = env.ray_labels == PERSON
        if not m.any():
            return math.inf
        return float(env.fused_rays[m].min()) - env.cfg.robot_radius

    def stopping_distance(self, env, v=None):
        v = env.v if v is None else v
        c = env.cfg
        return (self.config.base_stop_dist + v * self.config.reaction_time
                + v * v / (2 * c.brake))

    # ------------------------------------------------------------------
    def filter(self, env, action):
        """Return (final_action, speed_cap, reason). Also sets env cap."""
        action = int(action)
        cfg = self.config
        c = env.cfg
        final, cap, reason = action, None, "No rule triggered"

        if self.enabled:
            # Speed the robot would have after executing the action
            v_target = {GO: c.v_max, SLOW: c.v_max * c.slow_factor,
                        STOP: 0.0, TURN_LEFT: c.v_max * c.turn_speed_factor,
                        TURN_RIGHT: c.v_max * c.turn_speed_factor}[action]
            v_next = min(env.v + c.accel * c.dt, v_target) \
                if v_target > env.v else v_target
            omega = {TURN_LEFT: c.turn_rate,
                     TURN_RIGHT: -c.turn_rate}.get(action, 0.0)
            d_path = self._path_obstacle_distance(env, max(v_next, env.v),
                                                  omega)
            stop_d = self.stopping_distance(env, max(v_next, env.v))
            d_person = self._person_distance(env)
            unc = env.percep.uncertainty

            # Rule 1: hard stop
            if d_path < stop_d and action not in (STOP,):
                cap = 0.0
                final = action if action in (TURN_LEFT, TURN_RIGHT) else STOP
                reason = (f"Person in path at {d_path:.1f} m"
                          if d_person < cfg.person_stop_dist + 0.3
                          else f"Obstacle in path at {d_path:.1f} m")
                if final == STOP:
                    self._blocked += 1
                    if self._blocked > self.recovery_after:
                        # liveness: rotate in place toward the freer side
                        left = env.fused_rays[env.ray_angles > 0.15].mean()
                        right = env.fused_rays[env.ray_angles < -0.15].mean()
                        final = TURN_LEFT if left >= right else TURN_RIGHT
                        reason = (f"Blocked {self._blocked * c.dt:.1f} s: "
                                  f"recovery turn in place")
            # Rule 2: person priority
            elif d_person < cfg.person_stop_dist and action != STOP:
                cap = 0.0
                final = action if action in (TURN_LEFT, TURN_RIGHT) else STOP
                reason = f"Person at {d_person:.1f} m"
            elif d_person < cfg.person_slow_dist and action == GO:
                cap = cfg.person_slow_speed
                final = SLOW
                reason = f"Person nearby ({d_person:.1f} m): slow"
            # Rule 3: uncertainty override
            elif unc >= cfg.unc_stop and action != STOP:
                cap = 0.0
                final = action if action in (TURN_LEFT, TURN_RIGHT) else STOP
                reason = f"High depth uncertainty ({unc:.2f}): re-check"
            elif unc >= cfg.unc_slow and action == GO:
                cap = c.v_max * (1.0 - unc)
                final = SLOW
                reason = f"Uncertain depth ({unc:.2f}): speed capped"

        if not reason.startswith(("Obstacle", "Person in path", "Blocked")):
            self._blocked = 0
        env.speed_cap = cap
        if final != action or cap is not None:
            self.interventions += 1
        self.log.append(DecisionRecord(env.steps, ACTIONS[action],
                                       ACTIONS[final], reason, cap))
        return final, cap, reason
