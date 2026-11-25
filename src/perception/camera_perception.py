"""
Camera perception for the simulated robot: image + video processing that
turns each RGB-D camera frame into the obstacle map the policy drives on.

Inputs (nothing else): the RGB image, the depth image, the camera
calibration, the robot's pose and the static rack map (the same map the
global planner already uses). The simulator's object list is never read
here; it is used only by the evaluation code to score the output.

Per frame:

1. **Map background subtraction** (depth image). The static map is
   ray-cast from the robot pose into an *expected* depth image. Pixels
   that are clearly closer than expected and 0.15 - 1.7 m above the floor
   are foreground: something stands in the aisle that is not on the map.
2. **Depth image -> range scan.** For each of the policy's 21 bearings the
   nearest return inside the robot's height band is taken (10th percentile
   of the pixels on that bearing), like `depthimage_to_laserscan` on a
   real AMR. Foreground pixels are searched in a wider window around each
   bearing, so an obstacle between two rays is never missed, while racks
   and walls are measured exactly along the ray.
3. **Detector.** The fine-tuned YOLO11n (ONNX) runs on the RGB frame.
4. **Box -> metres.** Each box gets a distance two ways: from the depth
   pixels inside it, and from where its bottom edge touches the floor
   (pinhole ground-plane geometry: z = f * h_cam / (v_bottom - v_horizon)).
   If the two disagree (stretch wrap fools the depth sensor) the nearer
   one is kept. Boxes that do not stand out from the map (a picture of a
   box on a shelf) or have an impossible physical size are rejected.
5. **Tracking** on the floor plane (src/perception/tracking.py): bridges
   missed frames, stabilises class and distance, estimates velocity.
6. **Ray labels.** Rays covered by a tracked object get its class and
   distance; rays that match the map are rack / wall; any other near
   return is an *unknown obstacle* (found by depth only).
7. **Specular highlights** are found in the RGB image (saturated pixels in
   the obstacle band) and feed the uncertainty score.

Depth hole filling: stretch wrap makes the depth sensor "see through" an
object in vertical bands. A band of background-looking depth that has
foreground on both sides in the same image rows is filled with the
neighbouring foreground depth (horizontal morphological closing). Rays
that needed filling are reported as uncertain, so the robot still slows
down near shiny loads, but it no longer sees a gap that is not there.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.perception.onnx_detector import CLASS_NAMES, DEFAULT_ONNX, \
    OnnxDetector
from src.perception.tracking import Detection3D, GroundTracker
from src.simulation.warehouse_env import (
    BOX, FORKLIFT, PALLET, PERSON, RACK, WALL,
)

# detector class id -> simulation category
CLASS_TO_CATEGORY = {0: PERSON, 1: FORKLIFT, 2: PALLET}
# plausible physical height (m) of each detector class
HEIGHT_RANGE = {0: (0.7, 2.4), 1: (1.2, 3.4), 2: (0.2, 2.0)}


@dataclass
class PerceptionConfig:
    conf_low: float = 0.15       # weakest detection that can extend a track
    conf_new: float = 0.40       # weakest detection that can start a track
    conf_instant: float = 0.62   # confident enough to report at once
    use_tracking: bool = True
    use_depth_gate: bool = True  # reject boxes that are flat on the map
    use_size_gate: bool = True
    min_fg_frac: float = 0.10
    floor_margin: float = 0.15   # ignore depth returns lower than this (m)
    max_height: float = 1.7      # ... and higher than this
    ray_window: float = 0.20     # foreground search window per ray
                                 # (fraction of the spacing to the next ray)
    percentile: float = 0.10     # nearest-return percentile in a window
    fg_rel: float = 0.07         # closer than the map by this fraction ...
    fg_abs: float = 0.10         # ... plus this many metres = foreground
    fg_sigma: float = 2.6        # or this many sigmas of the depth noise
    fill_px: int = 45            # widest see-through band that is filled
    max_coast: int = 2           # frames a track is reported without a box
    saturation: int = 250        # RGB level that counts as a specular glint


@dataclass
class PerceptionResult:
    depth: np.ndarray            # (n_rays,) range from the depth image (m)
    det_label: np.ndarray        # (n_rays,) category or -1
    det_dist: np.ndarray         # (n_rays,) range of the labelled thing
    specular: np.ndarray         # (n_rays,) bool
    objects: list = field(default_factory=list)   # tracked, for display
    unknown: list = field(default_factory=list)   # depth-only blobs
    raw: np.ndarray | None = None                  # raw detector boxes
    rejected: list = field(default_factory=list)   # boxes the gates removed
    ms: dict = field(default_factory=dict)


class CameraPerception:
    """RGB-D frame -> per-ray obstacle map (see module docstring)."""

    def __init__(self, camera, detector=None, config=None, n_rays=21,
                 fov_deg=120.0, dt=0.2):
        self.cam = camera
        self.cfg = config or PerceptionConfig()
        self.detector = detector if detector is not None else OnnxDetector(
            DEFAULT_ONNX)
        self.tracker = GroundTracker(dt=dt, conf_new=self.cfg.conf_new,
                                     conf_instant=self.cfg.conf_instant,
                                     max_misses=self.cfg.max_coast)
        self._kernel = np.ones((3, 3), np.uint8)
        self._fill = np.ones((1, self.cfg.fill_px), np.uint8)
        self._setup_rays(n_rays, fov_deg)
        self.last = None

    def _setup_rays(self, n_rays, fov_deg):
        cam = self.cam
        ang = np.deg2rad(np.linspace(-fov_deg / 2, fov_deg / 2, n_rays))
        self.ray_angles = ang
        centre = cam.cx - cam.f * np.tan(ang)          # image column
        edges = np.clip(centre, 0, cam.w - 1)
        self.windows = []
        for i in range(n_rays):
            lo = edges[i + 1] if i + 1 < n_rays else edges[i]
            hi = edges[i - 1] if i > 0 else edges[i]
            # columns decrease with the angle: neighbours i+1 (left) and
            # i-1 (right) bound this ray's share of the image
            a = edges[i] - self.cfg.ray_window * (edges[i] - lo)
            b = edges[i] + self.cfg.ray_window * (hi - edges[i])
            a, b = int(round(min(a, b))), int(round(max(a, b)))
            a = max(min(a, cam.w - 3), 0)
            b = min(max(b, a + 2), cam.w - 1)
            self.windows.append((a, b + 1))
        self.ray_col = np.clip(np.round(centre), 0, cam.w - 1).astype(int)
        self.narrow = [(max(k - 1, 0), min(k + 2, cam.w))
                       for k in self.ray_col]
        self.ray_cos = np.cos(ang)

    def reset(self):
        self.tracker.reset()
        self.last = None

    # ------------------------------------------------------------------
    def _scan(self, depth, mask, windows, min_px=6):
        """Nearest return per ray among the masked pixels of its image
        window (metres along the optical axis), inf if there are none."""
        q = self.cfg.percentile
        out = np.full(len(windows), np.inf)
        for i, (a, b) in enumerate(windows):
            d = depth[:, a:b][mask[:, a:b]]
            if d.size >= min_px:
                k = int(q * d.size)
                out[i] = np.partition(d, k)[k]
        return out

    def _noise_level(self, depth, z_map):
        """Relative depth noise, measured on the floor right in front of
        the robot (robust: median absolute deviation)."""
        H = depth.shape[0]
        r = depth[H - 90:H:3, ::4] / z_map[H - 90:H:3, ::4] - 1.0
        return float(1.4826 * np.median(np.abs(r - np.median(r))))

    def _box_range(self, box, depth, fg, f, cy, cam_h, H):
        """Distance of a detection box: depth pixels vs floor contact."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        ix1, ix2 = int(x1 + 0.15 * w), int(math.ceil(x2 - 0.15 * w))
        iy1, iy2 = int(y1 + 0.1 * (y2 - y1)), int(math.ceil(y2))
        ix2, iy2 = max(ix2, ix1 + 1), max(iy2, iy1 + 1)
        m = fg[iy1:iy2, ix1:ix2]
        frac = float(m.mean()) if m.size else 0.0
        z_d = None
        if m.sum() >= 8:
            vals = depth[iy1:iy2, ix1:ix2][m]
            z_d = float(np.percentile(vals, 25))
        z_g = None
        cut = y2 >= H - 3
        if not cut and y2 > cy + 3:
            z_g = f * cam_h / (y2 - cy)
        return z_d, z_g, frac, cut

    # ------------------------------------------------------------------
    def process(self, env, frame, use_depth=True, use_tracking=None):
        """One frame. `use_depth=False` is the detector-only ablation: the
        depth image is ignored, so distances come from floor contact only
        and anything the detector has no class for is invisible."""
        t0 = time.perf_counter()
        cam, cfg, c = self.cam, self.cfg, env.cfg
        use_tracking = cfg.use_tracking if use_tracking is None \
            else use_tracking
        f, cx, cy, cam_h = cam.f, cam.cx, cam.cy, cam.cam_h
        H, W = frame.depth.shape
        depth = frame.depth

        # --- 1. map prior and foreground ------------------------------
        cast = cam.cast_static(env)
        z_map = cam.static_depth(env, cast)
        height = cam_h - cam.dy[:, None] * depth / f
        band = (height > cfg.floor_margin) & (height < cfg.max_height)
        # foreground threshold adapts to the sensor noise in this frame
        sigma = self._noise_level(depth, z_map)
        rel = max(cfg.fg_rel, cfg.fg_sigma * sigma)
        # (low parts of an object count too: the "closer than the map"
        # test already rules the floor out)
        fg = (height > 0.05) & (height < cfg.max_height) & \
            (depth < z_map * (1.0 - rel) - cfg.fg_abs)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                              self._kernel).astype(bool)
        filled = np.zeros(len(self.windows), bool)
        if not use_depth:
            fg[:] = False
        elif fg.any():
            # fill see-through bands inside foreground objects
            fg8 = fg.astype(np.uint8)
            closed = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, self._fill)
            holes = (closed > 0) & ~fg
            if holes.any():
                near = cv2.erode(np.where(fg, depth, np.float32(1e3)),
                                 self._fill)
                holes &= near < depth          # only where depth reads far
                depth = np.where(holes, near, depth)
                fg = fg | holes
                band = band | holes
                for i, (a, b) in enumerate(self.windows):
                    filled[i] = holes[:, a:b].sum() >= 6

        # --- 2. range scan ---------------------------------------------
        z_fg = self._scan(depth, fg, self.windows)       # unmapped things
        z_ray = np.minimum(self._scan(depth, band, self.narrow, 3), z_fg)
        ray_depth = np.minimum(z_ray / self.ray_cos, c.max_range)
        has_fg = np.isfinite(z_fg)
        map_col = np.minimum(cast["wall_z"], cast["rack_z"])
        r_map = np.minimum(map_col[self.ray_col] / self.ray_cos,
                           c.max_range)
        map_cat = np.where(cast["rack_z"][self.ray_col]
                           < cast["wall_z"][self.ray_col], RACK, WALL)
        if not use_depth:
            ray_depth = r_map.copy()       # the robot only has its map
        t1 = time.perf_counter()

        # --- 3. detector -------------------------------------------------
        raw = self.detector(frame.rgb, cfg.conf_low)
        t2 = time.perf_counter()

        # --- 4. boxes -> metres, gating ---------------------------------
        ct, st = math.cos(env.theta), math.sin(env.theta)
        dets, rejected = [], []
        for x1, y1, x2, y2, conf, cls in raw:
            cls = int(cls)
            z_d, z_g, frac, cut = self._box_range(
                (x1, y1, x2, y2), depth, fg, f, cy, cam_h, H)
            if not use_depth:
                z_d = None
                if z_g is None:           # box cut by the frame: use size
                    z_g = f * 0.5 * sum(HEIGHT_RANGE[cls]) / max(y2 - y1, 1)
            elif cfg.use_depth_gate and frac < cfg.min_fg_frac:
                rejected.append((x1, y1, x2, y2, conf, cls, "flat on map"))
                continue
            depth_ok = True
            if z_d is not None and z_g is not None:
                if abs(z_d - z_g) <= 0.2 * z_g + 0.3:
                    z, sigma = z_d, 0.06 + 0.04 * z_d
                else:                     # depth and geometry disagree
                    z, sigma = min(z_d, z_g), 0.15 + 0.08 * min(z_d, z_g)
                    depth_ok = False
            elif z_d is not None:
                z, sigma = z_d, 0.06 + 0.04 * z_d
            elif z_g is not None:
                z, sigma = z_g, 0.10 + 0.10 * z_g
                depth_ok = False
            else:
                rejected.append((x1, y1, x2, y2, conf, cls, "no range"))
                continue
            h_m = (y2 - y1) * z / f
            touches = cut or y1 <= 2 or x1 <= 2 or x2 >= W - 2
            lo, hi = HEIGHT_RANGE[cls]
            if cfg.use_size_gate and (h_m > hi or (h_m < lo and not touches)):
                rejected.append((x1, y1, x2, y2, conf, cls, "size"))
                continue
            uc = 0.5 * (x1 + x2)
            lat = z * (cx - uc) / f                 # left positive
            dets.append(Detection3D(
                x=env.x + ct * z - st * lat, y=env.y + st * z + ct * lat,
                rng=math.hypot(z, lat), bearing=math.atan2(lat, z), cls=cls,
                conf=float(conf), box=(float(x1), float(y1), float(x2),
                                       float(y2)),
                half_width=0.5 * (x2 - x1) * z / f,
                height=0.0 if touches else h_m, range_sigma=sigma,
                depth_ok=depth_ok))

        # --- 5. tracking -------------------------------------------------
        if use_tracking:
            tracks = self.tracker.step(dets, env.theta, cfg.conf_low)
            live = [t for t in tracks if t.confirmed]
        else:
            live = None

        # --- 6. objects in the robot frame, ray labels -----------------
        n = len(self.windows)
        det_label = np.full(n, -1)
        det_dist = np.full(n, np.inf)
        objects = []

        def add_object(px, py, cls, conf, box, half_w, h_m, tid, vel,
                       matched, depth_ok):
            dx, dy = px - env.x, py - env.y
            z = ct * dx + st * dy
            lat = -st * dx + ct * dy
            if z < 0.15:
                return
            uc = cx - f * lat / z
            hw_px = max(f * half_w / z, 2.0)
            if uc + hw_px < 0 or uc - hw_px > W:
                return
            a, b = int(max(uc - hw_px, 0)), int(min(uc + hw_px + 1, W))
            if not matched:
                # coasting: keep the label only while the depth image still
                # shows something at the predicted place
                y_b = int(min(cy + f * cam_h / z, H))
                y_t = int(max(cy + f * (cam_h - 1.2) / z, 0))
                near = fg[y_t:y_b, a:b] & (np.abs(depth[y_t:y_b, a:b] - z)
                                           < 0.35 + 0.12 * z)
                if use_depth and (near.size == 0 or near.mean() < 0.12):
                    return
                y2 = cy + f * cam_h / z
                hh = f * (h_m if h_m > 0 else 1.0) / z
                box = (uc - hw_px, y2 - hh, uc + hw_px, y2)
            rng = math.hypot(z, lat)
            objects.append({
                "cls": cls, "name": CLASS_NAMES[cls], "conf": conf,
                "box": tuple(float(v) for v in box), "range": rng, "z": z,
                "bearing": math.atan2(lat, z), "tid": tid, "vel": vel,
                "coasting": not matched, "depth_ok": depth_ok,
                "cols": (a, b)})

        if live is None:
            for d in dets:
                if d.conf >= cfg.conf_new:
                    add_object(d.x, d.y, d.cls, d.conf, d.box, d.half_width,
                               d.height, -1, (0.0, 0.0), True, d.depth_ok)
        else:
            for t in live:
                add_object(t.state[0], t.state[1], t.cls, t.conf, t.box,
                           t.half_width, t.height, t.tid,
                           (float(t.state[2]), float(t.state[3])),
                           t.matched, t.depth_ok)

        for o in sorted(objects, key=lambda o: -o["z"]):     # near wins
            a, b = o["cols"]
            for i, (wa, wb) in enumerate(self.windows):
                # the object must lie on the ray itself (not just near
                # it), otherwise obstacles would look wider than they are
                if not a <= self.ray_col[i] < b:
                    continue
                r = o["z"] / self.ray_cos[i]
                if ray_depth[i] < r - (0.5 + 0.1 * r):
                    continue          # something nearer is in front of it
                det_label[i] = CLASS_TO_CATEGORY[o["cls"]]
                det_dist[i] = r
        # rays that see the static map: no foreground on their bearing, or
        # a return that agrees with the mapped distance (sensor noise on a
        # rack must not become an "unknown obstacle"). Beyond the sensing
        # range nothing is labelled, as in the emulation.
        agrees = ray_depth >= r_map - (0.18 + 0.08 * r_map)
        on_map = (det_label < 0) & (~has_fg | agrees) & \
            (r_map < c.max_range - 0.05)
        det_label[on_map] = map_cat[on_map]
        det_dist[on_map] = r_map[on_map]

        # --- 7. specular highlights in the RGB image ------------------
        sat = frame.rgb.min(axis=2) >= cfg.saturation
        specular = np.zeros(n, bool)
        if sat.any():
            z_near = np.where(np.isfinite(det_dist) & (det_label >= 0)
                              & (det_label != RACK) & (det_label != WALL),
                              det_dist, ray_depth) * self.ray_cos
            for i, (a, b) in enumerate(self.windows):
                z = max(float(z_near[i]), 0.4)
                y_b = int(min(cy + f * cam_h / z, H))
                y_t = int(max(cy + f * (cam_h - 1.6) / z, 0))
                specular[i] = sat[y_t:y_b, a:b].sum() >= 3
        specular |= filled            # depth had to be repaired here

        # --- unknown obstacles: foreground not explained by a track ----
        unknown = self._unknown_blobs(fg, depth, z_map, objects) \
            if use_depth else []
        t3 = time.perf_counter()
        res = PerceptionResult(
            ray_depth, det_label, det_dist, specular, objects, unknown, raw,
            rejected, {"depth": 1000 * (t1 - t0), "detector": 1000 * (t2 - t1),
                       "fusion": 1000 * (t3 - t2)})
        self.last = res
        return res

    def _unknown_blobs(self, fg, depth, z_map, objects, min_area=45):
        """Depth-only obstacles: foreground blobs outside every tracked
        box (half resolution, connected components)."""
        m = (fg & (depth < z_map * 0.86 - 0.15)).astype(np.uint8)
        for o in objects:
            x1, y1, x2, y2 = o["box"]
            px, py = 0.12 * (x2 - x1) + 3, 0.06 * (y2 - y1) + 3
            m[int(max(y1 - py, 0)):int(y2 + py) + 1,
              int(max(x1 - px, 0)):int(x2 + px) + 1] = 0
        small = m[::2, ::2]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(small, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area or h < 5:
                continue
            vals = depth[::2, ::2][lab == i]
            z = float(np.percentile(vals, 20))
            out.append({"box": (2.0 * x, 2.0 * y, 2.0 * (x + w),
                                2.0 * (y + h)), "z": z, "area": int(area)})
        return out


# ----------------------------------------------------------------------
BOX_COL = {"person": (60, 220, 60), "forklift": (255, 190, 30),
           "package": (70, 160, 255), "unknown": (225, 60, 225)}


def draw_overlay(rgb, result, show_raw=False):
    """Draw what the perception stack reports onto a copy of the frame."""
    vis = rgb.copy()
    if show_raw and result.raw is not None:
        for x1, y1, x2, y2, conf, cls in result.raw:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (150, 150, 150), 1)
    for u in result.unknown:
        x1, y1, x2, y2 = [int(v) for v in u["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COL["unknown"], 2)
        _label(vis, f"unknown (depth) {u['z']:.1f}m", x1, y1,
               BOX_COL["unknown"])
    for o in result.objects:
        x1, y1, x2, y2 = [int(v) for v in o["box"]]
        col = BOX_COL[o["name"]]
        if o["coasting"]:
            _dashed(vis, (x1, y1, x2, y2), col)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        tid = f"#{o['tid']} " if o["tid"] >= 0 else ""
        _label(vis, f"{o['name']} {tid}{o['conf']:.2f} {o['range']:.1f}m",
               x1, y1, col)
    return vis


def _label(img, text, x, y, col):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    y0 = max(y - th - 5, 0)
    x = int(min(max(x, 0), img.shape[1] - tw - 4))
    cv2.rectangle(img, (x, y0), (x + tw + 4, y0 + th + 5), col, -1)
    cv2.putText(img, text, (x + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (15, 15, 15), 1, cv2.LINE_AA)


def _dashed(img, box, col, step=8):
    x1, y1, x2, y2 = box
    for x in range(x1, x2, 2 * step):
        cv2.line(img, (x, y1), (min(x + step, x2), y1), col, 2)
        cv2.line(img, (x, y2), (min(x + step, x2), y2), col, 2)
    for y in range(y1, y2, 2 * step):
        cv2.line(img, (x1, y), (x1, min(y + step, y2)), col, 2)
        cv2.line(img, (x2, y), (x2, min(y + step, y2)), col, 2)


# ----------------------------------------------------------------------
def attach_real_camera(env, splits=("val", "test"), style=None, styles=None,
                       detector=None, config=None, meter=True, **cam_kw):
    """Switch `env` to the real-image camera: from now on every step is
    perceived through rendered photo frames and the detector.

    By default the objects are the held-out (val + test) cut-outs."""
    from src.evaluation.sim_detection import DetectionMeter
    from src.simulation.photo_camera import PhotoCamera
    kw = dict(cam_kw)
    if styles:
        kw["styles"] = tuple(styles)
    cam = PhotoCamera(splits=splits, style=style, **kw)
    env.camera = cam
    env.vision = CameraPerception(cam, detector, config,
                                  n_rays=env.cfg.n_rays,
                                  fov_deg=env.cfg.fov_deg, dt=env.cfg.dt)
    env.vision_meter = DetectionMeter() if meter else None
    return env


def detach_real_camera(env):
    env.camera = env.vision = env.vision_meter = None
    env.frame = env.vision_result = None
    return env
