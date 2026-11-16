"""
Multi-object tracking on the ground plane (the "video processing" part).

Detections are noisy from frame to frame: a worker is missed in one frame,
a box jumps, a label flips. Tracking them over time fixes that. Because
the camera sits on a robot that drives and turns (up to 80 degrees per
second), image-space tracking would lose every object in a turn. Each
detection is therefore projected onto the warehouse floor with the robot's
pose, and objects are tracked there, in world coordinates:

* state [x, y, vx, vy], constant-velocity Kalman filter;
* measurement noise built from range and bearing accuracy (range is much
  noisier than bearing, and gets worse with distance);
* two-stage association in the spirit of ByteTrack: confident detections
  are matched first, weak ones (down to conf 0.15) may only *extend*
  an existing track, never start one;
* a track is reported once it was seen twice (or once with high
  confidence) and survives a few missed frames ("coasting").

The filter also gives every obstacle a velocity, i.e. the closing speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection3D:
    x: float                 # world position of the front surface (m)
    y: float
    rng: float               # distance from the robot (m)
    bearing: float           # robot frame, left positive (rad)
    cls: int
    conf: float
    box: tuple               # image box x1, y1, x2, y2
    half_width: float        # physical half width (m)
    height: float = 0.0      # physical height (m), 0 if the box is cut off
    range_sigma: float = 0.2
    depth_ok: bool = True    # depth image agreed with the box geometry


@dataclass
class Track:
    tid: int
    state: np.ndarray        # [x, y, vx, vy]
    cov: np.ndarray
    scores: np.ndarray       # running class scores
    conf: float
    box: tuple
    half_width: float
    height: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    matched: bool = True     # matched to a detection in the latest frame
    depth_ok: bool = True
    history: list = field(default_factory=list)

    @property
    def cls(self):
        return int(np.argmax(self.scores))

    @property
    def speed(self):
        return float(math.hypot(self.state[2], self.state[3]))


class GroundTracker:
    """Constant-velocity Kalman tracker in world (floor) coordinates."""

    def __init__(self, dt=0.2, n_classes=3, accel_sigma=1.6,
                 conf_new=0.40, conf_instant=0.62, min_hits=2,
                 max_misses=4, bearing_sigma=math.radians(1.6)):
        self.dt = dt
        self.n_classes = n_classes
        self.conf_new, self.conf_instant = conf_new, conf_instant
        self.min_hits, self.max_misses = min_hits, max_misses
        self.bearing_sigma = bearing_sigma
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = dt
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.Q = accel_sigma ** 2 * g @ g.T
        self.Hm = np.zeros((2, 4))
        self.Hm[0, 0] = self.Hm[1, 1] = 1.0
        self.tracks: list[Track] = []
        self._next = 1

    def reset(self):
        self.tracks = []
        self._next = 1

    # ------------------------------------------------------------------
    def _meas_cov(self, det, robot_theta):
        """Range / bearing noise rotated into world axes."""
        a = robot_theta + det.bearing
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        sr = det.range_sigma
        st = max(det.rng * self.bearing_sigma, 0.03)
        return R @ np.diag([sr * sr, st * st]) @ R.T

    def _gate(self, det):
        return 0.75 + 0.14 * det.rng

    def _associate(self, tracks, dets):
        """Greedy nearest-neighbour matching with a distance gate."""
        pairs = []
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                dist = math.hypot(t.state[0] - d.x, t.state[1] - d.y)
                if d.cls != t.cls:
                    dist += 0.35             # prefer the same class
                if dist < self._gate(d) + 0.3 * t.misses:
                    pairs.append((dist, i, j))
        pairs.sort()
        used_t, used_d, out = set(), set(), []
        for dist, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            out.append((i, j))
        return out, used_t, used_d

    def _update(self, t, d, robot_theta):
        R = self._meas_cov(d, robot_theta)
        z = np.array([d.x, d.y])
        y = z - self.Hm @ t.state
        S = self.Hm @ t.cov @ self.Hm.T + R
        K = t.cov @ self.Hm.T @ np.linalg.inv(S)
        t.state = t.state + K @ y
        t.cov = (np.eye(4) - K @ self.Hm) @ t.cov
        t.scores *= 0.75
        t.scores[d.cls] += 0.25 * d.conf + 0.1
        t.conf = 0.6 * t.conf + 0.4 * d.conf
        t.box, t.half_width = d.box, 0.6 * t.half_width + 0.4 * d.half_width
        if d.height > 0:
            t.height = d.height if t.height == 0 else \
                0.6 * t.height + 0.4 * d.height
        t.depth_ok = d.depth_ok
        t.hits += 1
        t.misses = 0
        t.matched = True
        if t.hits >= self.min_hits or d.conf >= self.conf_instant:
            t.confirmed = True

    # ------------------------------------------------------------------
    def step(self, dets, robot_theta, conf_low=0.15):
        """Advance one frame. Returns the list of live tracks."""
        for t in self.tracks:                              # predict
            t.state = self.F @ t.state
            t.cov = self.F @ t.cov @ self.F.T + self.Q
            t.age += 1
            t.matched = False
        strong = [d for d in dets if d.conf >= self.conf_new]
        weak = [d for d in dets if conf_low <= d.conf < self.conf_new]

        pairs, used_t, used_d = self._associate(self.tracks, strong)
        for i, j in pairs:
            self._update(self.tracks[i], strong[j], robot_theta)
        # weak detections can only extend tracks that already exist
        rest = [t for k, t in enumerate(self.tracks) if k not in used_t]
        pairs2, _, _ = self._associate(rest, weak)
        for i, j in pairs2:
            self._update(rest[i], weak[j], robot_theta)
        # unmatched strong detections start new tracks
        for j, d in enumerate(strong):
            if j in used_d:
                continue
            sc = np.zeros(self.n_classes)
            sc[d.cls] = d.conf
            t = Track(self._next, np.array([d.x, d.y, 0.0, 0.0]),
                      np.diag([0.3 ** 2, 0.3 ** 2, 1.5 ** 2, 1.5 ** 2]),
                      sc, d.conf, d.box, d.half_width, d.height,
                      confirmed=d.conf >= self.conf_instant,
                      depth_ok=d.depth_ok)
            self._next += 1
            self.tracks.append(t)
        # age out
        alive = []
        for t in self.tracks:
            if not t.matched:
                t.misses += 1
            limit = self.max_misses if t.confirmed else 1
            if t.misses <= limit:
                alive.append(t)
        self.tracks = alive
        return self.tracks
