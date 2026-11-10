"""
Warehouse navigation environment (Gymnasium) for the camera-based
obstacle-avoidance robot.

The environment is a 2D top-down warehouse with rack rows, aisles,
cross-aisles, and obstacles that are NOT on the robot's map:

* workers (``person``)            - moving, safety-critical
* forklifts (``forklift``)        - moving along aisles
* pallets (``pallet``)            - static, some are stretch-wrapped
                                    (reflective -> unreliable depth)
* fallen boxes (``unknown``)      - static, NOT a detector class: only the
                                    depth branch can see them

The robot carries a simulated forward-facing camera. Instead of feeding
raw pixels to the policy, the environment emulates the perception stack
from the project design (Sections 6-7 of the project document):

1. Detector branch   - labels known classes with some miss probability.
2. Depth branch      - per-ray distance for ANY surface, with noise,
                       false "shadow" returns and mirror-like failures on
                       reflective surfaces.
3. Fusion            - detector boxes give robust distances; depth finds
                       unknown obstacles.
4. Uncertainty       - temporal consistency + detector/depth disagreement
                       + specular highlights -> one score in [0, 1].
5. Tracking          - median over the last frames removes one-frame
                       flicker; closing speed per sector.

The resulting compact *obstacle map* is the RL state. This mirrors the
sim-to-real argument in the project: the policy reads an obstacle map
that looks the same whether it came from simulation or a real camera.

Global path: the static rack layout is known (as in real AMR fleets), so a
grid distance field to the goal provides a "carrot" waypoint. The RL policy
is responsible for local obstacle avoidance and speed control.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

try:  # gymnasium is light-weight, but keep the core usable without it
    import gymnasium as gym
    from gymnasium import spaces
    _GymBase = gym.Env
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _GymBase = object


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

ACTIONS = ["GO", "SLOW", "STOP", "TURN_LEFT", "TURN_RIGHT"]
GO, SLOW, STOP, TURN_LEFT, TURN_RIGHT = range(5)

# Object categories
RACK, WALL, PALLET, BOX, PERSON, FORKLIFT = range(6)
CATEGORY_NAMES = {
    RACK: "rack",
    WALL: "wall",
    PALLET: "pallet",
    BOX: "unknown",
    PERSON: "person",
    FORKLIFT: "forklift",
    -1: "free",
}
# Categories the YOLO-style detector has been trained on
DETECTABLE = {PALLET, PERSON, FORKLIFT}
# Categories that are part of the static map (always "known")
MAP_CATEGORIES = {RACK, WALL}

PERCEPTION_MODES = {
    # name: (use_depth, use_uncertainty, use_tracking)
    "detector_only": (False, False, False),
    "fusion": (True, False, False),
    "fusion_uncertainty": (True, True, False),
    "full": (True, True, True),
}


@dataclass
class EnvConfig:
    """All tunable parameters of the warehouse simulation."""

    width: float = 24.0
    height: float = 16.0
    dt: float = 0.2
    max_steps: int = 450

    # Robot
    robot_radius: float = 0.35
    v_max: float = 1.5
    accel: float = 1.5          # m/s^2
    brake: float = 3.0          # m/s^2
    turn_rate: float = 1.4      # rad/s
    slow_factor: float = 0.4
    turn_speed_factor: float = 0.3

    # Camera / perception
    fov_deg: float = 120.0
    n_rays: int = 21
    max_range: float = 6.0
    n_sectors: int = 7
    detect_range: float = 5.5

    # Obstacle counts (ranges, sampled per episode = domain randomization)
    n_workers: tuple = (1, 4)
    n_forklifts: tuple = (0, 2)
    n_pallets: tuple = (1, 4)
    n_boxes: tuple = (1, 4)
    reflective_prob: float = 0.4

    # Perception noise (ranges for domain randomization)
    depth_noise: tuple = (0.01, 0.05)
    detect_prob: tuple = (0.80, 0.97)
    shadow_prob: tuple = (0.0, 0.02)
    mirror_fail_prob: float = 0.45

    perception_mode: str = "full"

    # Reward weights
    r_goal: float = 10.0
    r_collision: float = -10.0
    r_progress: float = 1.0
    r_time: float = -0.01
    r_near_miss: float = -0.08
    r_person_near: float = -0.15
    r_harsh_brake: float = -0.05
    near_miss_dist: float = 0.35
    person_margin: float = 0.9

    goal_radius: float = 0.6
    grid_res: float = 0.25

    # Fixed layout (None -> random). Useful for the demo.
    fixed_layout_seed: int | None = None


@dataclass
class Obstacle:
    category: int
    x: float
    y: float
    hw: float = 0.0           # half width (x) for boxes
    hh: float = 0.0           # half height (y) for boxes
    radius: float = 0.0       # for circles (persons)
    vx: float = 0.0
    vy: float = 0.0
    reflective: bool = False
    bounds: tuple = (0.0, 0.0)  # patrol limits along motion axis
    axis: int = 0               # 0: moves along x, 1: along y
    uid: int = 0
    user_placed: bool = False
    wait: int = 0

    @property
    def is_circle(self):
        return self.radius > 0.0


@dataclass
class PerceptionState:
    history: list = field(default_factory=list)  # past raw depth readings
    prev_fused: np.ndarray | None = None
    uncertainty: float = 0.0
    sector_uncertainty: np.ndarray | None = None


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def _ray_aabb(ox, oy, dx, dy, bx, by, hw, hh):
    """Vectorised slab test. Rays (R,) vs boxes (B,) -> distances (R, B)."""
    dx = dx[:, None]
    dy = dy[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_dx = np.where(np.abs(dx) < 1e-9, 1e9, 1.0 / dx)
        inv_dy = np.where(np.abs(dy) < 1e-9, 1e9, 1.0 / dy)
    tx1 = (bx - hw - ox) * inv_dx
    tx2 = (bx + hw - ox) * inv_dx
    ty1 = (by - hh - oy) * inv_dy
    ty2 = (by + hh - oy) * inv_dy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= np.maximum(tmin, 0.0))
    t = np.where(tmin > 0, tmin, tmax)
    return np.where(hit & (t >= 0), t, np.inf)


def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    """Rays (R,) vs circles (C,) -> distances (R, C)."""
    fx = ox - cx  # (C,)
    fy = oy - cy
    dx = dx[:, None]
    dy = dy[:, None]
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = -b - sq
    t = np.where(t < 0, -b + sq, t)
    return np.where((disc >= 0) & (t >= 0), t, np.inf)


def _circle_aabb_dist(px, py, bx, by, hw, hh):
    """Distance from point to boxes' surfaces (negative inside -> 0)."""
    qx = np.maximum(np.abs(px - bx) - hw, 0.0)
    qy = np.maximum(np.abs(py - by) - hh, 0.0)
    return np.sqrt(qx * qx + qy * qy)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

class WarehouseEnv(_GymBase):
    """Gymnasium environment: camera-guided warehouse robot navigation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode=None,
                 seed: int | None = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        c = self.cfg

        self.ray_angles = np.deg2rad(
            np.linspace(-c.fov_deg / 2, c.fov_deg / 2, c.n_rays)
        )
        self.obs_dim = c.n_rays + 3 * c.n_sectors + 5
        if spaces is not None:
            self.action_space = spaces.Discrete(len(ACTIONS))
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
            )
        self.np_random = np.random.default_rng(seed)
        self.speed_cap = None   # set by the safety shield
        self._uid = 0
        self.statics: list[Obstacle] = []
        self.dynamics: list[Obstacle] = []
        self.racks: list[Obstacle] = []

    # ------------------------------------------------------------------
    # Layout generation
    # ------------------------------------------------------------------
    def _next_uid(self):
        self._uid += 1
        return self._uid

    def _build_layout(self, rng):
        c = self.cfg
        racks = []
        W, H = c.width, c.height
        t = 0.6  # wall half thickness outside the world
        walls = [
            Obstacle(WALL, W / 2, -t, W / 2 + t, t),
            Obstacle(WALL, W / 2, H + t, W / 2 + t, t),
            Obstacle(WALL, -t, H / 2, t, H / 2 + t),
            Obstacle(WALL, W + t, H / 2, t, H / 2 + t),
        ]
        n_rows = int(rng.integers(2, 4))   # 2 or 3 rack rows
        thick = float(rng.uniform(0.9, 1.2))
        aisle = (H - n_rows * thick) / (n_rows + 1)
        x_start = float(rng.uniform(4.0, 5.0))
        x_end = float(rng.uniform(19.0, 20.0))
        self.aisles = []  # (y_low, y_high) of each aisle
        y = 0.0
        row_centers = []
        for r in range(n_rows):
            y_low = y
            y += aisle
            self.aisles.append((y_low, y))
            yc = y + thick / 2
            row_centers.append(yc)
            y += thick
        self.aisles.append((y, H))

        self.cross_aisles = []
        for yc in row_centers:
            n_gaps = int(rng.integers(1, 3))
            gap_w = float(rng.uniform(2.0, 2.8))
            cands = np.sort(rng.uniform(x_start + 3, x_end - 3, n_gaps))
            # ensure gaps don't overlap
            gaps = []
            for g in cands:
                if not gaps or g - gaps[-1] > gap_w + 2.0:
                    gaps.append(float(g))
            segs = []
            cur = x_start
            for g in gaps:
                segs.append((cur, g - gap_w / 2))
                self.cross_aisles.append((g - gap_w / 2, g + gap_w / 2, yc))
                cur = g + gap_w / 2
            segs.append((cur, x_end))
            for (a, b) in segs:
                if b - a > 0.5:
                    racks.append(Obstacle(RACK, (a + b) / 2, yc,
                                          (b - a) / 2, thick / 2))
        self.x_start, self.x_end = x_start, x_end
        self.racks = racks
        self.walls = walls

    # ------------------------------------------------------------------
    def _free_point(self, rng, x_range, y_range=None, clearance=0.7,
                    tries=200):
        for _ in range(tries):
            x = float(rng.uniform(*x_range))
            if y_range is None:
                a = self.aisles[int(rng.integers(len(self.aisles)))]
                y = float(rng.uniform(a[0] + 0.8, a[1] - 0.8))
            else:
                y = float(rng.uniform(*y_range))
            if self._clearance_at(x, y) > clearance:
                return x, y
        return None

    def _all_boxes(self, include_dynamic=True):
        objs = self.racks + self.walls + [o for o in self.statics
                                          if not o.is_circle]
        if include_dynamic:
            objs = objs + [o for o in self.dynamics if not o.is_circle]
        return objs

    def _all_circles(self, include_dynamic=True):
        objs = [o for o in self.statics if o.is_circle]
        if include_dynamic:
            objs = objs + [o for o in self.dynamics if o.is_circle]
        return objs

    def _clearance_at(self, x, y, include_dynamic=True, exclude=None):
        """Distance from a point to the nearest obstacle surface."""
        best = np.inf
        boxes = [o for o in self._all_boxes(include_dynamic)
                 if o is not exclude]
        if boxes:
            bx = np.array([o.x for o in boxes])
            by = np.array([o.y for o in boxes])
            hw = np.array([o.hw for o in boxes])
            hh = np.array([o.hh for o in boxes])
            best = min(best, float(np.min(
                _circle_aabb_dist(x, y, bx, by, hw, hh))))
        circles = [o for o in self._all_circles(include_dynamic)
                   if o is not exclude]
        if circles:
            cx = np.array([o.x for o in circles])
            cy = np.array([o.y for o in circles])
            r = np.array([o.radius for o in circles])
            best = min(best, float(np.min(
                np.hypot(x - cx, y - cy) - r)))
        return best

    def _aisle_of(self, y):
        for i, (a, b) in enumerate(self.aisles):
            if a <= y <= b:
                return i
        return None

    def _place_static(self, rng, category):
        c = self.cfg
        for _ in range(100):
            a_idx = int(rng.integers(len(self.aisles)))
            y0, y1 = self.aisles[a_idx]
            x = float(rng.uniform(self.x_start + 0.5, self.x_end - 0.5))
            if category == PALLET:
                if rng.random() < 0.5:
                    hw, hh = 0.6, 0.4
                else:
                    hw, hh = 0.4, 0.6
            else:
                hw = hh = float(rng.uniform(0.2, 0.35))
            # Keep a passable gap (>= 1.3 m) on at least one side
            min_gap = 2 * c.robot_radius + 0.6
            if (y1 - y0) - 2 * hh < min_gap + 0.1:
                continue
            side = rng.random()
            if side < 0.35:   # against the lower rack
                y = y0 + hh + 0.05
            elif side < 0.7:  # against the upper rack
                y = y1 - hh - 0.05
            else:             # somewhere in the middle
                y = float(rng.uniform(y0 + hh, y1 - hh))
            gap = max((y - hh) - y0, y1 - (y + hh))
            if gap < min_gap:
                continue
            # avoid start/goal areas and other statics
            if math.hypot(x - self.start[0], y - self.start[1]) < 2.0:
                continue
            if math.hypot(x - self.goal[0], y - self.goal[1]) < 2.0:
                continue
            if self._clearance_at(x, y, include_dynamic=False) < \
                    max(hw, hh) + 0.9:
                # don't stack obstacles to block aisles
                if self._clearance_at(x, y, include_dynamic=False) < \
                        max(hw, hh) + 0.1:
                    continue
            o = Obstacle(category, x, y, hw, hh,
                         reflective=(category == PALLET and
                                     rng.random() < c.reflective_prob),
                         uid=self._next_uid())
            return o
        return None

    def _spawn_worker(self, rng):
        c = self.cfg
        speed = float(rng.uniform(0.4, 1.1))
        if self.cross_aisles and rng.random() < 0.4:
            # crosses through a cross-aisle (moves along y)
            g0, g1, _ = self.cross_aisles[int(rng.integers(
                len(self.cross_aisles)))]
            x = float(rng.uniform(g0 + 0.4, g1 - 0.4))
            y = float(rng.uniform(1.0, c.height - 1.0))
            o = Obstacle(PERSON, x, y, radius=0.28, vy=speed *
                         (1 if rng.random() < 0.5 else -1), axis=1,
                         bounds=(0.6, c.height - 0.6),
                         uid=self._next_uid())
        else:
            a = self.aisles[int(rng.integers(len(self.aisles)))]
            y = float(rng.uniform(a[0] + 0.5, a[1] - 0.5))
            x = float(rng.uniform(self.x_start, self.x_end))
            lo = float(rng.uniform(1.0, self.x_start + 2))
            hi = float(rng.uniform(self.x_end - 2, c.width - 1.0))
            o = Obstacle(PERSON, x, y, radius=0.28, vx=speed *
                         (1 if rng.random() < 0.5 else -1), axis=0,
                         bounds=(lo, hi), uid=self._next_uid())
        if math.hypot(o.x - self.start[0], o.y - self.start[1]) < 2.5:
            return None
        if self._clearance_at(o.x, o.y) < 0.5:
            return None
        return o

    def _spawn_forklift(self, rng):
        c = self.cfg
        wide = [a for a in self.aisles if a[1] - a[0] > 2.6]
        if not wide:
            return None
        a = wide[int(rng.integers(len(wide)))]
        y = (a[0] + a[1]) / 2 + float(rng.uniform(-0.3, 0.3))
        x = float(rng.uniform(self.x_start + 2, self.x_end - 2))
        speed = float(rng.uniform(0.6, 1.2))
        o = Obstacle(FORKLIFT, x, y, hw=0.9, hh=0.5,
                     vx=speed * (1 if rng.random() < 0.5 else -1), axis=0,
                     bounds=(self.x_start - 1.5, self.x_end + 1.5),
                     uid=self._next_uid())
        if math.hypot(o.x - self.start[0], o.y - self.start[1]) < 3.0:
            return None
        if self._clearance_at(o.x, o.y) < 0.3:
            return None
        return o

    # ------------------------------------------------------------------
    # Distance field for the global path
    # ------------------------------------------------------------------
    def _build_distance_field(self):
        c = self.cfg
        res = c.grid_res
        nx = int(math.ceil(c.width / res))
        ny = int(math.ceil(c.height / res))
        xs = (np.arange(nx) + 0.5) * res
        ys = (np.arange(ny) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)  # (ny, nx)
        occ_clear = np.full(gx.shape, np.inf)
        for o in self.racks + self.walls:
            d = _circle_aabb_dist(gx, gy, o.x, o.y, o.hw, o.hh)
            occ_clear = np.minimum(occ_clear, d)
        inflate = c.robot_radius + 0.15
        blocked = occ_clear < inflate
        # extra cost near racks so the path keeps to the aisle centre
        extra = np.clip(1.2 - occ_clear, 0.0, None) * 2.0

        gi = int(min(max(self.goal[0] / res, 0), nx - 1))
        gj = int(min(max(self.goal[1] / res, 0), ny - 1))
        dist = np.full((ny, nx), np.inf)
        dist[gj, gi] = 0.0
        heap = [(0.0, gj, gi)]
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142),
                (1, 1, 1.4142)]
        while heap:
            d, j, i = heapq.heappop(heap)
            if d > dist[j, i]:
                continue
            for dj, di, w in nbrs:
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and not blocked[jj, ii]:
                    nd = d + w * res * (1.0 + extra[jj, ii])
                    if nd < dist[jj, ii]:
                        dist[jj, ii] = nd
                        heapq.heappush(heap, (nd, jj, ii))
        finite = np.isfinite(dist)
        if finite.any():
            dist[~finite] = dist[finite].max() + 5.0
        self.dist_field = dist
        self.grid_shape = (ny, nx)

    def _field_value(self, x, y):
        res = self.cfg.grid_res
        ny, nx = self.grid_shape
        fx = min(max(x / res - 0.5, 0), nx - 1.001)
        fy = min(max(y / res - 0.5, 0), ny - 1.001)
        i0, j0 = int(fx), int(fy)
        ax, ay = fx - i0, fy - j0
        d = self.dist_field
        return float(
            d[j0, i0] * (1 - ax) * (1 - ay) + d[j0, i0 + 1] * ax * (1 - ay)
            + d[j0 + 1, i0] * (1 - ax) * ay + d[j0 + 1, i0 + 1] * ax * ay
        )

    def global_path(self, x, y, max_len=200):
        """Steepest descent on the distance field from (x, y) to goal."""
        res = self.cfg.grid_res
        ny, nx = self.grid_shape
        i = int(min(max(x / res, 0), nx - 1))
        j = int(min(max(y / res, 0), ny - 1))
        path = [(x, y)]
        d = self.dist_field
        for _ in range(max_len):
            best = (d[j, i], j, i)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < ny and 0 <= ii < nx and d[jj, ii] < best[0]:
                        best = (d[jj, ii], jj, ii)
            if best[1] == j and best[2] == i:
                break
            j, i = best[1], best[2]
            path.append(((i + 0.5) * res, (j + 0.5) * res))
            if d[j, i] <= 0:
                break
        path.append(tuple(self.goal))
        return path

    def carrot(self, lookahead=2.0):
        path = self.global_path(self.x, self.y, max_len=int(
            lookahead / self.cfg.grid_res) + 2)
        acc = 0.0
        px, py = path[0]
        for qx, qy in path[1:]:
            acc += math.hypot(qx - px, qy - py)
            px, py = qx, qy
            if acc >= lookahead:
                break
        return px, py

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        rng = self.np_random
        c = self.cfg
        options = options or {}
        self._uid = 0
        layout_seed = options.get("layout_seed", c.fixed_layout_seed)
        layout_rng = (np.random.default_rng(layout_seed)
                      if layout_seed is not None else rng)

        self.statics, self.dynamics = [], []
        self._build_layout(layout_rng)

        # Start (left staging area) and goal (right staging area)
        self.start = self._free_point(layout_rng, (1.0, 2.8), clearance=0.8)
        self.goal = self._free_point(layout_rng, (c.width - 2.8, c.width - 1.0),
                                     clearance=0.8)
        if layout_rng.random() < 0.5:  # sometimes drive right -> left
            self.start, self.goal = self.goal, self.start
            self.heading0 = math.pi
        else:
            self.heading0 = 0.0
        self.goal = np.array(self.goal, dtype=float)

        # Domain randomisation of the episode
        counts = options.get("counts", {})
        n_p = counts.get("pallets", int(rng.integers(c.n_pallets[0],
                                                     c.n_pallets[1] + 1)))
        n_b = counts.get("boxes", int(rng.integers(c.n_boxes[0],
                                                   c.n_boxes[1] + 1)))
        n_w = counts.get("workers", int(rng.integers(c.n_workers[0],
                                                     c.n_workers[1] + 1)))
        n_f = counts.get("forklifts", int(rng.integers(c.n_forklifts[0],
                                                       c.n_forklifts[1] + 1)))
        if "reflective_prob" in options:
            refl_backup = c.reflective_prob
            c.reflective_prob = options["reflective_prob"]
        for _ in range(n_p):
            o = self._place_static(rng, PALLET)
            if o:
                self.statics.append(o)
        for _ in range(n_b):
            o = self._place_static(rng, BOX)
            if o:
                self.statics.append(o)
        if "reflective_prob" in options:
            c.reflective_prob = refl_backup
        for _ in range(n_w):
            for _try in range(20):
                o = self._spawn_worker(rng)
                if o:
                    self.dynamics.append(o)
                    break
        for _ in range(n_f):
            for _try in range(20):
                o = self._spawn_forklift(rng)
                if o:
                    self.dynamics.append(o)
                    break

        self.depth_noise = float(rng.uniform(*c.depth_noise))
        self.detect_prob = float(rng.uniform(*c.detect_prob))
        self.shadow_prob = float(rng.uniform(*c.shadow_prob))
        self.perception_mode = options.get("perception_mode",
                                           c.perception_mode)

        self.x, self.y = self.start
        self.theta = self.heading0 + float(rng.uniform(-0.4, 0.4))
        self.v = 0.0
        self.omega = 0.0
        self.steps = 0
        self.speed_cap = None
        self.collided = False
        self.collision_with = None
        self.reached = False
        self.near_misses = 0
        self._was_near = False
        self.min_clearance = np.inf
        self.path_length = 0.0
        self.last_action = STOP

        self._build_distance_field()
        self.prev_potential = self._field_value(self.x, self.y)
        self.percep = PerceptionState()
        self.sec_dist = None
        self.percep.sector_uncertainty = np.zeros(c.n_sectors)
        self._perceive()
        return self._observation(), self._info()

    # ------------------------------------------------------------------
    def add_obstacle(self, category, x, y, reflective=False):
        """Insert a user-placed obstacle (used by the interactive demo)."""
        if category == PERSON:
            o = Obstacle(PERSON, x, y, radius=0.28, uid=self._next_uid(),
                         user_placed=True)
        elif category == PALLET:
            o = Obstacle(PALLET, x, y, 0.6, 0.4, reflective=reflective,
                         uid=self._next_uid(), user_placed=True)
        else:
            o = Obstacle(BOX, x, y, 0.3, 0.3, uid=self._next_uid(),
                         user_placed=True)
        if math.hypot(x - self.x, y - self.y) < self.cfg.robot_radius + 0.6:
            return False
        self.statics.append(o)
        return True

    def remove_obstacle_near(self, x, y, radius=0.8):
        best, bd = None, radius
        for o in self.statics:
            d = math.hypot(o.x - x, o.y - y)
            if d < bd:
                best, bd = o, d
        if best is not None:
            self.statics.remove(best)
            return True
        return False

    # ------------------------------------------------------------------
    def _move_dynamics(self):
        c = self.cfg
        dt = c.dt
        for o in self.dynamics:
            nx_, ny_ = o.x + o.vx * dt, o.y + o.vy * dt
            # bounce at patrol bounds
            if o.axis == 0 and not (o.bounds[0] <= nx_ <= o.bounds[1]):
                o.vx = -o.vx
                continue
            if o.axis == 1 and not (o.bounds[0] <= ny_ <= o.bounds[1]):
                o.vy = -o.vy
                continue
            # yield to the robot (people and forklift drivers see it)
            if o.is_circle:
                d_new = math.hypot(nx_ - self.x, ny_ - self.y) - o.radius
            else:
                d_new = float(_circle_aabb_dist(self.x, self.y, nx_, ny_,
                                                o.hw, o.hh))
            if d_new < c.robot_radius + 0.25:
                o.wait += 1
                if o.wait > 6:      # waited ~1.2 s: turn around
                    o.vx, o.vy, o.wait = -o.vx, -o.vy, 0
                continue
            o.wait = 0
            # workers should not walk through racks / statics
            ox, oy = o.x, o.y
            o.x, o.y = nx_, ny_
            if o.is_circle:
                cl = self._clearance_at(o.x, o.y, include_dynamic=False,
                                        exclude=o)
                if cl < o.radius:
                    o.x, o.y = ox, oy
                    if o.axis == 0:
                        o.vx = -o.vx
                    else:
                        o.vy = -o.vy
            else:
                blocked = False
                for s in self.statics:
                    if s.is_circle:
                        continue
                    if (abs(s.x - o.x) < s.hw + o.hw and
                            abs(s.y - o.y) < s.hh + o.hh):
                        blocked = True
                        break
                if blocked:
                    o.x, o.y = ox, oy
                    o.vx = -o.vx

    def _robot_clearance(self):
        """Clearance between robot disc and every obstacle + nearest cat."""
        best, cat = np.inf, -1
        boxes = self._all_boxes()
        if boxes:
            bx = np.array([o.x for o in boxes])
            by = np.array([o.y for o in boxes])
            hw = np.array([o.hw for o in boxes])
            hh = np.array([o.hh for o in boxes])
            d = _circle_aabb_dist(self.x, self.y, bx, by, hw, hh)
            k = int(np.argmin(d))
            best, cat = float(d[k]), boxes[k].category
        circles = self._all_circles()
        person_clear = np.inf
        if circles:
            cx = np.array([o.x for o in circles])
            cy = np.array([o.y for o in circles])
            r = np.array([o.radius for o in circles])
            d = np.hypot(self.x - cx, self.y - cy) - r
            k = int(np.argmin(d))
            person_clear = float(d[k])
            if d[k] < best:
                best, cat = float(d[k]), circles[k].category
        return best - self.cfg.robot_radius, cat, \
            person_clear - self.cfg.robot_radius

    def step(self, action):
        c = self.cfg
        action = int(action)
        self.last_action = action
        # Commanded velocities
        if action == GO:
            v_cmd, w_cmd = c.v_max, 0.0
        elif action == SLOW:
            v_cmd, w_cmd = c.v_max * c.slow_factor, 0.0
        elif action == STOP:
            v_cmd, w_cmd = 0.0, 0.0
        elif action == TURN_LEFT:
            v_cmd, w_cmd = c.v_max * c.turn_speed_factor, c.turn_rate
        else:
            v_cmd, w_cmd = c.v_max * c.turn_speed_factor, -c.turn_rate
        if self.speed_cap is not None:
            v_cmd = min(v_cmd, self.speed_cap)

        v_prev = self.v
        if v_cmd > self.v:
            self.v = min(v_cmd, self.v + c.accel * c.dt)
        else:
            self.v = max(v_cmd, self.v - c.brake * c.dt)
        self.omega = w_cmd

        # Integrate in two sub-steps for collision robustness
        reward = 0.0
        for _ in range(2):
            h = c.dt / 2
            self.theta = wrap_angle(self.theta + self.omega * h)
            dx = self.v * math.cos(self.theta) * h
            dy = self.v * math.sin(self.theta) * h
            self.x += dx
            self.y += dy
            self.path_length += math.hypot(dx, dy)
            clear, cat, _ = self._robot_clearance()
            if clear <= 0.0:
                self.collided = True
                self.collision_with = CATEGORY_NAMES[cat]
                break

        self._move_dynamics()
        self.steps += 1

        clear, cat, person_clear = self._robot_clearance()
        self.min_clearance = min(self.min_clearance, clear)
        pot = self._field_value(self.x, self.y)
        progress = self.prev_potential - pot
        self.prev_potential = pot
        reward += c.r_progress * progress + c.r_time

        near = False
        if clear < c.near_miss_dist and not self.collided:
            reward += c.r_near_miss
            near = True
        if person_clear < c.person_margin and not self.collided:
            reward += c.r_person_near
            near = True
        if near and not self._was_near:
            self.near_misses += 1      # count near-miss *events*
        self._was_near = near
        if v_prev - self.v > 0.5:
            reward += c.r_harsh_brake

        dist_goal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        terminated = False
        if self.collided:
            reward += c.r_collision
            terminated = True
        elif dist_goal < c.goal_radius:
            reward += c.r_goal
            self.reached = True
            terminated = True
        truncated = self.steps >= c.max_steps and not terminated

        self._perceive()
        return self._observation(), float(reward), terminated, truncated, \
            self._info()

    # ------------------------------------------------------------------
    # Perception emulation
    # ------------------------------------------------------------------
    def cast(self, angles, include_unknown=True):
        """Cast rays from the robot. Returns distance & hit object idx."""
        c = self.cfg
        ang = self.theta + angles
        dx, dy = np.cos(ang), np.sin(ang)
        objs = []
        dists = []
        boxes = self._all_boxes()
        if boxes:
            bx = np.array([o.x for o in boxes])
            by = np.array([o.y for o in boxes])
            hw = np.array([o.hw for o in boxes])
            hh = np.array([o.hh for o in boxes])
            dists.append(_ray_aabb(self.x, self.y, dx, dy, bx, by, hw, hh))
            objs += boxes
        circles = self._all_circles()
        if circles:
            cx = np.array([o.x for o in circles])
            cy = np.array([o.y for o in circles])
            r = np.array([o.radius for o in circles])
            dists.append(_ray_circle(self.x, self.y, dx, dy, cx, cy, r))
            objs += circles
        D = np.concatenate(dists, axis=1)
        if not include_unknown:
            mask = np.array([o.category not in (BOX,) for o in objs])
            D = np.where(mask[None, :], D, np.inf)
        k = np.argmin(D, axis=1)
        d = D[np.arange(len(angles)), k]
        hit = [objs[i] if np.isfinite(d[n]) and d[n] < c.max_range else None
               for n, i in enumerate(k)]
        d = np.minimum(d, c.max_range)
        return d, hit

    def _perceive(self):
        """Run the simulated detector + depth + fusion + tracking stack."""
        c = self.cfg
        rng = self.np_random
        use_depth, use_unc, use_track = PERCEPTION_MODES[self.perception_mode]
        true_d, hit = self.cast(self.ray_angles)
        n = c.n_rays

        # ---- Depth branch -------------------------------------------
        depth = true_d * (1.0 + rng.normal(0, self.depth_noise, n))
        specular = np.zeros(n, dtype=bool)
        mirror_fail = np.zeros(n, dtype=bool)
        for i, o in enumerate(hit):
            if o is not None and o.reflective:
                specular[i] = rng.random() < 0.6   # bright highlight seen
                if rng.random() < c.mirror_fail_prob:
                    # depth "sees through" the shiny wrap: far reading
                    depth[i] = min(c.max_range,
                                   true_d[i] + rng.uniform(1.5, 4.0))
                    mirror_fail[i] = True
        shadows = rng.random(n) < self.shadow_prob
        depth[shadows] = np.minimum(depth[shadows],
                                    rng.uniform(0.6, 2.5, shadows.sum()))
        depth = np.clip(depth, 0.05, c.max_range)

        # ---- Detector branch ----------------------------------------
        # One detection per object (like a bounding box). The box gives a
        # robust median depth, so all rays hitting the object share it.
        det_label = np.full(n, -1)
        det_dist = np.full(n, np.inf)
        detected = {}
        for i, o in enumerate(hit):
            if o is None:
                continue
            if o.category in MAP_CATEGORIES:
                # racks/walls are on the static map: always known
                det_label[i] = o.category
                det_dist[i] = true_d[i] * (1 + rng.normal(0, 0.01))
                continue
            if o.category in DETECTABLE and true_d[i] < c.detect_range:
                if o.uid not in detected:
                    detected[o.uid] = rng.random() < self.detect_prob
                if detected[o.uid]:
                    det_label[i] = o.category
                    det_dist[i] = true_d[i] * (1 + rng.normal(0, 0.03))
        self.detections = detected

        # ---- Fusion ---------------------------------------------------
        if use_depth:
            fused = depth.copy()
            has_det = np.isfinite(det_dist)
            # detector box median depth is more robust than a single pixel
            fused[has_det] = np.minimum(det_dist[has_det],
                                        np.where(mirror_fail[has_det],
                                                 det_dist[has_det],
                                                 depth[has_det]))
            labels = det_label.copy()
            # depth-only returns become "unknown obstacle" labels
            unk = (~has_det) & (depth < c.max_range - 0.05)
            labels[unk] = BOX
        else:
            # Detector-only baseline: no distance for anything the detector
            # did not label (unknown boxes are invisible)
            fused = np.where(np.isfinite(det_dist), det_dist, c.max_range)
            labels = det_label.copy()

        # ---- Tracking (temporal median over the last 3 frames) --------
        hist = self.percep.history
        hist.append(fused.copy())
        if len(hist) > 3:
            hist.pop(0)
        raw_fused = fused.copy()
        if use_track and len(hist) >= 3:
            fused = np.median(np.stack(hist), axis=0)

        # ---- Uncertainty ---------------------------------------------
        prev = self.percep.prev_fused
        if prev is None:
            temporal = np.zeros(n)
        else:
            expected = self.v * c.dt * np.cos(self.ray_angles)
            jump = np.abs((prev - expected) - raw_fused)
            temporal = np.clip((jump - 0.3) / 1.5, 0, 1)
        disagree = np.zeros(n)
        has_det = np.isfinite(det_dist)
        disagree[has_det] = np.clip(
            (depth[has_det] - det_dist[has_det] - 0.5) / 1.5, 0, 1)
        spec = specular.astype(float)
        ray_unc = np.clip(0.45 * temporal + 0.4 * disagree + 0.35 * spec,
                          0, 1)
        # proximity weighting: uncertainty matters more when close
        near_w = np.clip(1.4 - raw_fused / c.max_range * 1.2, 0.2, 1.0)
        ray_unc = ray_unc * near_w
        sectors = np.array_split(np.arange(n), c.n_sectors)
        sec_unc = np.array([ray_unc[s].max() for s in sectors])
        if use_unc:
            # temporal smoothing (exponential moving average)
            self.percep.sector_uncertainty = (
                0.6 * self.percep.sector_uncertainty + 0.4 * sec_unc)
        else:
            self.percep.sector_uncertainty = np.zeros(c.n_sectors)
        mid = c.n_sectors // 2
        corridor = self.percep.sector_uncertainty[max(mid - 1, 0):mid + 2]
        self.percep.uncertainty = float(corridor.max())
        self.percep.prev_fused = raw_fused

        # ---- Sector summary (obstacle map) ---------------------------
        sec_dist = np.array([fused[s].min() for s in sectors])
        sec_person = np.zeros(c.n_sectors)
        for k, s in enumerate(sectors):
            j = s[np.argmin(fused[s])]
            if labels[j] == PERSON:
                sec_person[k] = 1.0
            elif labels[j] == FORKLIFT:
                sec_person[k] = 0.5
        prev_sec = getattr(self, "sec_dist", None)
        if prev_sec is None:
            closing = np.zeros(c.n_sectors)
        else:
            closing = (prev_sec - sec_dist) / c.dt
        self.sec_dist = sec_dist
        self.sec_class = sec_person
        self.sec_closing = closing

        self.true_rays = true_d
        self.raw_depth = depth
        self.fused_rays = fused
        self.ray_labels = labels
        self.ray_hits = hit
        self.specular = specular

    def _observation(self):
        c = self.cfg
        rays = self.fused_rays / c.max_range * 2 - 1
        sec_d = self.sec_dist / c.max_range * 2 - 1
        sec_cls = self.sec_class * 2 - 1
        sec_close = np.clip(self.sec_closing / 3.0, -1, 1)
        cx, cy = self.carrot()
        ang = wrap_angle(math.atan2(cy - self.y, cx - self.x) - self.theta)
        dist_goal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        extra = np.array([
            math.sin(ang), math.cos(ang),
            min(dist_goal / 10.0, 1.0) * 2 - 1,
            self.v / c.v_max * 2 - 1,
            self.percep.uncertainty * 2 - 1,
        ])
        obs = np.concatenate([rays, sec_d, sec_cls, sec_close, extra])
        return np.clip(obs, -1, 1).astype(np.float32)

    def _info(self):
        return {
            "reached": self.reached,
            "collided": self.collided,
            "collision_with": self.collision_with,
            "steps": self.steps,
            "near_misses": self.near_misses,
            "uncertainty": self.percep.uncertainty,
            "path_length": self.path_length,
        }

    # ------------------------------------------------------------------
    def render(self):
        from src.simulation.render import render_frame
        return render_frame(self)


def make_env(seed=0, **cfg_kwargs):
    """Factory used by Stable-Baselines3 vectorised environments."""
    def _f():
        env = WarehouseEnv(EnvConfig(**cfg_kwargs), seed=seed)
        return env
    return _f
