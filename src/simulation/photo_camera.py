"""
Real-image first-person camera for the warehouse simulation.

The schematic renderer draws flat colour slabs. This camera builds every
frame from **real photo material of the detector's training datasets**
(see assets/sim/ATTRIBUTION.md):

* rack faces, walls with dock doors, floor and ceiling are rectified
  regions of real warehouse photos, mapped onto the simulation's geometry
  with a perspective-correct ray caster (pinhole camera, 120 degree FOV);
* workers, forklifts, parcels and unknown objects are real cut-outs,
  drawn as depth-tested billboards standing on the floor at the true
  distance and scale.

Besides the RGB frame the camera returns a metric depth image (z-buffer
with a depth-sensor noise model) and, for evaluation only, the ground-truth
boxes of every visible object. The perception stack
(src/perception/camera_perception.py) only gets the RGB and depth images.

Everything is NumPy + OpenCV; a 640x384 frame takes about 10 ms.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.simulation.warehouse_env import (
    BOX, FORKLIFT, PALLET, PERSON, _ray_aabb, _ray_circle,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "sim"

# Simulation category -> sprite class (= the fine-tuned detector's class
# names; "unknown" objects have no detector class).
SPRITE_CLASS = {PERSON: "person", FORKLIFT: "forklift", PALLET: "package",
                BOX: "unknown"}

# Warehouse "looks": which rectified photos texture racks / floor / walls.
STYLES = {
    "a": {"rack": ["rack_a", "rack_a2"], "end": "rack_b_end",
          "floor": "floor_a", "wall": "wall_c", "ceiling": "ceiling_a"},
    "b": {"rack": ["rack_b"], "end": "rack_b_end", "floor": "floor_b",
          "wall": "wall_b", "ceiling": "ceiling_b"},
    "c": {"rack": ["rack_c"], "end": "rack_b_end", "floor": "floor_c",
          "wall": "wall_c", "ceiling": "ceiling_a"},
}
# Rows of each wall photo that show only plain wall cladding (above the
# dock door). They are stretched into plain panels placed between doors,
# so a wall is "panel, panel, door" instead of door after door.
WALL_PLAIN_ROWS = {"wall_c": (0.25, 0.40), "wall_b": (0.06, 0.34)}
WALL_PLAIN_M = 7.0       # metres of plain wall between two dock doors
FLOOR_UPSCALE = 3        # floor texture super-sampling (adds fine grain)
CEILING_H = 7.2          # m
WALL_H = 7.2             # m (walls reach the ceiling)
SPECULAR_LEVEL = 242     # diffuse surfaces stay below; glints saturate
NO_HIT = 1.0e6           # "no rack on this column" (finite: no NaNs)


def available(assets=ASSETS):
    """True if the real-image asset bank is present."""
    return (Path(assets) / "sprites" / "index.json").exists() and \
        (Path(assets) / "textures" / "index.json").exists()


# ----------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------
@dataclass
class Sprite:
    sid: str
    cls: str
    split: str
    path: Path
    aspect: float            # width / height
    height_m: float
    name: str = ""
    _mips: list | None = None

    def mips(self):
        """RGBA float32 mip pyramid (colour pre-multiplied by alpha)."""
        if self._mips is None:
            rgba = np.asarray(Image.open(self.path).convert("RGBA"),
                              dtype=np.float32)
            a = rgba[..., 3:4] / 255.0
            pm = np.concatenate([rgba[..., :3] * a, a], axis=2)
            mips = [pm]
            while min(mips[-1].shape[:2]) > 24:
                h, w = mips[-1].shape[:2]
                mips.append(cv2.resize(mips[-1], (max(w // 2, 1),
                                                  max(h // 2, 1)),
                                       interpolation=cv2.INTER_AREA))
            self._mips = mips
        return self._mips


def _stable_unit(text):
    """Deterministic pseudo-random number in [0, 1) from a string."""
    return (zlib.crc32(text.encode()) % 10_000) / 10_000.0


class SpriteBank:
    """Real object cut-outs, grouped by class and filtered by split."""

    def __init__(self, splits=("val", "test"), assets=ASSETS):
        root = Path(assets) / "sprites"
        index = json.loads((root / "index.json").read_text())["sprites"]
        self.splits = tuple(splits)
        self.by_class = {c: [] for c in ("person", "forklift", "package",
                                         "unknown")}
        for s in index:
            if s["split"] not in self.splits:
                continue
            aspect = s["w"] / s["h"]
            u = _stable_unit(s["id"])
            if s["class"] == "person":
                # standing people ~1.75 m; crouching / bending are lower
                h = 1.75 if aspect <= 0.40 else \
                    float(np.clip(1.75 * (0.40 / aspect) ** 1.1, 1.0, 1.75))
                h *= 0.95 + 0.10 * u
            elif s["class"] == "forklift":
                h = 2.05 + 0.2 * u
            elif s["class"] == "package":
                h = 1.0                       # fitted to the footprint later
            else:
                h = float(s.get("height_m", 0.6))
            self.by_class[s["class"]].append(Sprite(
                s["id"], s["class"], s["split"], root / s["file"], aspect,
                h, s.get("name", s["class"])))
        for lst in self.by_class.values():
            lst.sort(key=lambda sp: sp.sid)

    def counts(self):
        return {c: len(v) for c, v in self.by_class.items()}

    def pick(self, cls, rng, k=1):
        lst = self.by_class[cls]
        if not lst:
            raise ValueError(f"no '{cls}' sprites for splits {self.splits}")
        idx = rng.choice(len(lst), size=min(k, len(lst)), replace=False)
        return [lst[int(i)] for i in idx]


class TextureSet:
    """Textures of one warehouse style, packed for fast sampling."""

    PAD = 2

    def __init__(self, style, assets=ASSETS):
        root = Path(assets) / "textures"
        meta = {t["name"]: t for t in json.loads(
            (root / "index.json").read_text())["textures"]}
        src = {t["name"]: t for t in json.loads(
            (root / "sources.json").read_text())["textures"]}
        st = STYLES[style]
        self.style = style

        def load(name):
            img = cv2.imread(str(root / meta[name]["file"]))
            if img is None:
                raise FileNotFoundError(root / meta[name]["file"])
            return np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32)

        entries = []
        for n in st["rack"]:
            entries.append(("rack", load(n), meta[n]["width_m"],
                            meta[n]["height_m"], bool(src[n].get("mirror"))))
        n = st["end"]
        entries.append(("end", load(n), meta[n]["width_m"],
                        meta[n]["height_m"], True))
        n = st["wall"]
        door = load(n)
        r0, r1 = WALL_PLAIN_ROWS[n]
        band = door[int(r0 * door.shape[0]):int(r1 * door.shape[0])]
        ppm = door.shape[1] / meta[n]["width_m"]
        plain = cv2.resize(band, (int(WALL_PLAIN_M * ppm), door.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        entries.append(("wall", np.hstack([plain, door]),
                        WALL_PLAIN_M + meta[n]["width_m"], WALL_H, False))

        pad = self.PAD
        width = max(e[1].shape[1] for e in entries) + 2 * pad
        self.atlas = np.zeros((sum(e[1].shape[0] for e in entries), width, 3),
                              np.float32)
        self.rack_ids, self.tex = [], []
        y0 = 0
        for kind, img, wm, hm, mirror in entries:
            h, w = img.shape[:2]
            self.atlas[y0:y0 + h, pad:pad + w] = img
            # wrap (or mirror) padding so bilinear sampling has no seam
            left = img[:, :pad][:, ::-1] if mirror else img[:, -pad:]
            right = img[:, -pad:][:, ::-1] if mirror else img[:, :pad]
            self.atlas[y0:y0 + h, :pad] = left
            self.atlas[y0:y0 + h, pad + w:pad + w + pad] = right
            self.tex.append({"y0": y0, "w": w, "h": h, "width_m": wm,
                             "height_m": hm, "mirror": mirror})
            if kind == "rack":
                self.rack_ids.append(len(self.tex) - 1)
            elif kind == "end":
                self.end_id = len(self.tex) - 1
            else:
                self.wall_id = len(self.tex) - 1
            y0 += h
        self.rack_h = float(self.tex[self.rack_ids[0]]["height_m"])
        fl, ce = load(st["floor"]), load(st["ceiling"])
        # Floor: the photo patch is small, so close to the camera it would
        # be magnified into a blur. Up-sample it and modulate it with its
        # own fine grain repeated at a higher frequency.
        k = FLOOR_UPSCALE
        big = cv2.resize(fl, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        grain = fl.mean(axis=2)
        grain = grain / (cv2.GaussianBlur(grain, (0, 0), 6) + 1e-3)
        grain = np.tile(np.hstack([grain, grain[:, ::-1]]),
                        (k, (k + 1) // 2 + 1))[:big.shape[0], :big.shape[1]]
        fl = np.clip(big * (grain[..., None] ** 1.3), 0, 255)
        self.floor, self.ceiling = fl.astype(np.float32), ce
        self.floor_m = (meta[st["floor"]]["width_m"],
                        meta[st["floor"]]["height_m"])
        self.ceil_m = (meta[st["ceiling"]]["width_m"],
                       meta[st["ceiling"]]["height_m"])
        # per-texture arrays for vectorised lookups
        self.t_y0 = np.array([t["y0"] for t in self.tex], np.float32)
        self.t_w = np.array([t["w"] for t in self.tex], np.float32)
        self.t_h = np.array([t["h"] for t in self.tex], np.float32)
        self.t_wm = np.array([t["width_m"] for t in self.tex], np.float32)
        self.t_mirror = np.array([t["mirror"] for t in self.tex], bool)


# ----------------------------------------------------------------------
@dataclass
class CameraFrame:
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32 metres, sensor output (noisy)
    depth_true: np.ndarray     # noise-free depth, evaluation only
    gt: list = field(default_factory=list)   # ground truth, evaluation only


class PhotoCamera:
    """Forward-looking pinhole camera rendered from real photo material."""

    def __init__(self, width=640, height=384, fov_deg=120.0, cam_height=1.0,
                 horizon=0.44, splits=("val", "test"), style=None,
                 styles=("a", "b", "c"), assets=ASSETS, seed=0,
                 effects=True):
        self.w, self.h = int(width), int(height)
        self.f = (self.w / 2) / math.tan(math.radians(fov_deg) / 2)
        self.cx = (self.w - 1) / 2
        self.cy = horizon * self.h
        self.cam_h = float(cam_height)
        self.assets = Path(assets)
        self.bank = SpriteBank(splits, assets)
        self.fixed_style = style
        self.styles = tuple(styles)
        self.effects = effects
        self._tex_cache = {}
        self.rng = np.random.default_rng(seed)
        cols = np.arange(self.w, dtype=np.float32)
        self.tan_a = (self.cx - cols) / self.f          # left = +angle
        self.col_angle = np.arctan(self.tan_a)
        self.cos_a = np.cos(self.col_angle)
        rows = np.arange(self.h, dtype=np.float32)
        self.dy = rows - self.cy                        # >0 below horizon
        self.hrow = int(np.argmax(self.dy > 0))         # first floor row
        with np.errstate(divide="ignore"):
            zp = np.where(self.dy > 0,
                          self.f * self.cam_h / np.maximum(self.dy, 1e-3),
                          self.f * (CEILING_H - self.cam_h)
                          / np.maximum(-self.dy, 1e-3))
        self.z_plane = np.minimum(zp, 80.0).astype(np.float32)[:, None]
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        r2 = ((xx - self.cx) / self.w) ** 2 + ((yy - self.h / 2)
                                               / self.w) ** 2
        self._vignette = (1.0 - 0.55 * r2).astype(np.float32)
        nb = np.random.default_rng(12345)
        self._noise = nb.normal(0, 1, (6, self.h, self.w)).astype(np.float32)
        self._lowres = nb.normal(0, 1, (6, self.h // 8 + 1,
                                        self.w // 8 + 1)).astype(np.float32)
        self.sprites = {}           # uid -> list[Sprite]
        self.flip = {}
        self.current = {}           # uid -> index of the sprite in use
        self.tex = None
        self.reset(None, seed)

    # ------------------------------------------------------------------
    def reset(self, env=None, seed=0, style=None):
        """New episode: choose the warehouse look and forget sprites."""
        self.rng = np.random.default_rng([int(seed) % (2 ** 31), 77])
        st = style or self.fixed_style or \
            self.styles[int(self.rng.integers(len(self.styles)))]
        if st not in self._tex_cache:
            self._tex_cache[st] = TextureSet(st, self.assets)
        self.tex = self._tex_cache[st]
        self.style = st
        self.sprites, self.flip, self.current = {}, {}, {}
        self._rack_look = {}
        # exposure / white balance of this "camera run"
        self.gain = float(self.rng.uniform(0.9, 1.08))
        self.tint = np.array([self.rng.uniform(0.97, 1.03), 1.0,
                              self.rng.uniform(0.96, 1.04)], np.float32)
        vig = self._vignette if self.effects else np.ones_like(self._vignette)
        self._colour = (vig[..., None] * (self.gain * self.tint)).astype(
            np.float32)
        self.frame_idx = 0

    def intrinsics(self):
        return {"f": self.f, "cx": self.cx, "cy": self.cy,
                "cam_h": self.cam_h, "w": self.w, "h": self.h}

    # ------------------------------------------------------------------
    def _sprites_for(self, o):
        if o.uid not in self.sprites:
            cls = SPRITE_CLASS[o.category]
            rng = np.random.default_rng([abs(int(o.uid)) + 1,
                                         int(self.rng.integers(1 << 30))])
            if cls == "forklift":
                # a narrow (front / rear) and a wide (side) view if we can
                lst = sorted(self.bank.pick(cls, rng, k=6),
                             key=lambda s: s.aspect)
                chosen = [lst[0], lst[len(lst) // 2], lst[-1]]
            else:
                chosen = self.bank.pick(cls, rng, k=1)
            self.sprites[o.uid] = chosen
            self.flip[o.uid] = bool(rng.random() < 0.5)
            self.current[o.uid] = 0
        return self.sprites[o.uid]

    def _rack_tex(self, k, rack):
        key = (round(rack.x, 2), round(rack.y, 2))
        if key not in self._rack_look:
            r = np.random.default_rng([int(abs(rack.x) * 100) + 1,
                                       int(abs(rack.y) * 100) + 1])
            self._rack_look[key] = (
                self.tex.rack_ids[int(r.integers(len(self.tex.rack_ids)))],
                float(r.uniform(0, 20)))
        return self._rack_look[key]

    # ------------------------------------------------------------------
    def cast_static(self, env):
        """Ray-cast the static map (racks, walls) for every image column.

        Returns per-column z-depth of the outer wall and of the nearest
        rack (inf if none) plus texture lookups. Uses only the known map
        and the robot pose (what a localised robot has)."""
        ang = env.theta + self.col_angle
        dx, dy = np.cos(ang), np.sin(ang)

        def cast(objs):
            bx = np.array([o.x for o in objs])
            by = np.array([o.y for o in objs])
            hw = np.array([o.hw for o in objs])
            hh = np.array([o.hh for o in objs])
            D = _ray_aabb(env.x, env.y, dx, dy, bx, by, hw, hh)
            k = np.argmin(D, axis=1)
            d = D[np.arange(self.w), k]
            hx, hy = env.x + d * dx, env.y + d * dy
            lx, ly = hx - bx[k], hy - by[k]
            # which face: the one whose plane the hit point lies on
            on_x = np.abs(np.abs(lx) - hw[k]) < np.abs(np.abs(ly) - hh[k])
            return d, k, lx, ly, hx, hy, on_x, hw[k], hh[k]

        out = {}
        d, k, lx, ly, hx, hy, on_x, hw, hh = cast(env.walls)
        # coordinate along the wall, running left -> right for the viewer
        along = np.where(on_x, np.where(lx > 0, hy, -hy),
                         np.where(ly > 0, -hx, hx))
        out["wall_z"] = (d * self.cos_a).astype(np.float32)
        out["wall_u"] = along.astype(np.float32)
        if env.racks:
            d, k, lx, ly, hx, hy, on_x, hw, hh = cast(env.racks)
            hit = np.isfinite(d)
            tex_id = np.zeros(self.w, np.int32)
            u_m = np.zeros(self.w, np.float32)
            for i in np.unique(k[hit]):
                tid, off = self._rack_tex(int(i), env.racks[int(i)])
                m = hit & (k == i)
                long_face = ~on_x[m]          # racks run along x
                tex_id[m] = np.where(long_face, tid, self.tex.end_id)
                # long face: world x (mirrored when seen from above);
                # end face: stretch the end texture over the rack depth
                u_long = np.where(ly[m] > 0, -hx[m], hx[m]) + off
                u_end = (np.where(lx[m] > 0, ly[m], -ly[m]) + hh[m]) \
                    / (2 * hh[m]) * self.tex.t_wm[self.tex.end_id]
                u_m[m] = np.where(long_face, u_long, np.clip(
                    u_end, 0.01, self.tex.t_wm[self.tex.end_id] - 0.01))
            out["rack_z"] = np.where(hit, d * self.cos_a,
                                     NO_HIT).astype(np.float32)
            out["rack_tex"], out["rack_u"] = tex_id, u_m
        else:
            out["rack_z"] = np.full(self.w, NO_HIT, np.float32)
            out["rack_tex"] = np.zeros(self.w, np.int32)
            out["rack_u"] = np.zeros(self.w, np.float32)
        return out

    def static_depth(self, env, cast=None):
        """Expected depth image of the empty warehouse (map prior)."""
        c = cast or self.cast_static(env)
        z = np.minimum(self._plane_z(), c["wall_z"][None, :])
        rack_z = c["rack_z"][None, :]
        zw = self.cam_h - self.dy[:, None] * rack_z / self.f
        rack = (rack_z < z) & (zw >= 0) & (zw <= self.tex.rack_h)
        return np.where(rack, rack_z, z).astype(np.float32)

    def _plane_z(self):
        """z-depth of floor (below horizon) / ceiling (above) per row."""
        return self.z_plane

    def _atlas_u(self, tex_id, u_m):
        t = self.tex
        wm, w = t.t_wm[tex_id], t.t_w[tex_id]
        frac = (u_m / wm) % 2.0
        tri = np.where(frac > 1.0, 2.0 - frac, frac)
        frac = np.where(t.t_mirror[tex_id], tri, frac % 1.0)
        return (t.PAD + frac * (w - 1)).astype(np.float32)

    # ------------------------------------------------------------------
    def _render_static(self, env):
        t = self.tex
        c = self.cast_static(env)
        H, W, f, hy = self.h, self.w, self.f, self.hrow
        dyc = self.dy[:, None]
        z_plane = self.z_plane                             # (H, 1)
        wall_z = c["wall_z"][None, :]
        rack_z = c["rack_z"][None, :]

        # floor / ceiling: world position of every pixel on its plane
        ct, st = math.cos(env.theta), math.sin(env.theta)
        A = (ct - self.tan_a * st)[None, :]
        B = (st + self.tan_a * ct)[None, :]

        def tri(v, period, n):
            # mirrored tiling: triangle wave with period 2 * period
            q = v * (0.5 / period)
            q -= np.floor(q + 0.5)
            np.abs(q, out=q)
            q *= 2.0 * (n - 1)
            return q

        img = np.empty((H, W, 3), np.float32)
        fh, fw = t.floor.shape[:2]
        zf = z_plane[hy:]
        img[hy:] = cv2.remap(t.floor, tri(env.x + zf * A, t.floor_m[0], fw),
                             tri(env.y + zf * B, t.floor_m[1], fh),
                             cv2.INTER_LINEAR)
        ch, cw = t.ceiling.shape[:2]
        zc = z_plane[:hy]
        img[:hy] = cv2.remap(t.ceiling, tri(env.x + zc * A, t.ceil_m[0], cw),
                             tri(env.y + zc * B, t.ceil_m[1], ch),
                             cv2.INTER_LINEAR)

        # walls and racks (vertical surfaces) from the atlas
        zw_rack = self.cam_h - dyc * rack_z / f            # height on rack
        is_rack = (rack_z < np.minimum(z_plane, wall_z)) & (zw_rack >= 0) \
            & (zw_rack <= t.rack_h)
        is_wall = (wall_z < z_plane) & ~is_rack
        solid = is_rack | is_wall
        zw_wall = self.cam_h - dyc * wall_z / f
        wall_u = self._atlas_u(np.full(W, t.wall_id), c["wall_u"])
        rack_u = self._atlas_u(c["rack_tex"], c["rack_u"])
        rid = c["rack_tex"]
        map_x = np.where(is_rack, rack_u[None, :], wall_u[None, :])
        map_y = np.where(
            is_rack,
            t.t_y0[rid][None, :] + np.clip(1.0 - zw_rack / t.rack_h, 0, 1)
            * (t.t_h[rid][None, :] - 1),
            t.t_y0[t.wall_id] + np.clip(1.0 - zw_wall / WALL_H, 0, 1)
            * (t.t_h[t.wall_id] - 1))
        vert = cv2.remap(t.atlas, map_x, map_y.astype(np.float32),
                         cv2.INTER_LINEAR)
        cv2.copyTo(vert, solid.astype(np.uint8), img)
        z = np.where(is_rack, rack_z, np.minimum(z_plane, wall_z)).astype(
            np.float32)

        # lighting: gentle fall-off with distance, darker where surfaces
        # meet the floor (ambient occlusion), ceiling a bit dimmer
        shade = 0.60 + 0.42 / (1.0 + z * (1.0 / 9.0))
        base_h = np.where(is_rack, zw_rack, zw_wall)
        foot = np.minimum(wall_z, rack_z)                  # (1, W)
        ao = np.where(solid, 0.80 + 0.20 * np.clip(base_h * (1 / 0.6), 0, 1),
                      0.78 + 0.22 * np.clip((foot - z_plane) * (1 / 0.9),
                                            0, 1))
        ao[:hy][~solid[:hy]] = 0.82
        img *= (shade * ao).astype(np.float32)[..., None]
        return img, z, c

    # ------------------------------------------------------------------
    def _extent(self, env, o):
        """Bearing interval of the object's footprint + nearest z-depth."""
        ct, st = math.cos(env.theta), math.sin(env.theta)
        if o.is_circle:
            dx, dy = o.x - env.x, o.y - env.y
            fwd, lat = ct * dx + st * dy, -st * dx + ct * dy
            dist = math.hypot(fwd, lat)
            if dist <= o.radius + 1e-3:
                return None
            if fwd <= 0.05:
                return None
            a0 = math.atan2(lat, fwd)
            half = math.asin(min(o.radius / dist, 1.0))
            lo, hi = a0 - half, a0 + half
            rng_near = dist - o.radius
            z_front = fwd * rng_near / dist      # nearest point, along axis
        else:
            xs = np.array([o.x - o.hw, o.x + o.hw, o.x + o.hw, o.x - o.hw])
            ys = np.array([o.y - o.hh, o.y - o.hh, o.y + o.hh, o.y + o.hh])
            dx, dy = xs - env.x, ys - env.y
            fwd, lat = ct * dx + st * dy, -st * dx + ct * dy
            if fwd.max() <= 0.05:
                return None
            ang = np.arctan2(lat, np.maximum(fwd, 1e-3))
            lo, hi = float(ang.min()), float(ang.max())
            # nearest point of the footprint, measured along the optical axis
            nx = min(max(env.x, o.x - o.hw), o.x + o.hw)
            ny = min(max(env.y, o.y - o.hh), o.y + o.hh)
            rng_near = math.hypot(nx - env.x, ny - env.y)
            z_front = ct * (nx - env.x) + st * (ny - env.y)
        if z_front < 0.12:
            z_front = 0.12
        lim = math.radians(88)
        lo, hi = max(lo, -lim), min(hi, lim)
        if hi <= lo:
            return None
        # image columns (left = +angle)
        u_lo = self.cx - self.f * math.tan(hi)
        u_hi = self.cx - self.f * math.tan(lo)
        return u_lo, u_hi, z_front, rng_near

    def _draw_sprites(self, env, img, zbuf):
        H, W, f = self.h, self.w, self.f
        ids = np.full((H, W), -1, np.int32)
        glint = np.zeros((H, W), np.float32)
        items = []
        for o in list(env.statics) + list(env.dynamics):
            if o.category not in SPRITE_CLASS:
                continue
            ext = self._extent(env, o)
            if ext is None:
                continue
            u_lo, u_hi, z_front, rng_near = ext
            if u_hi < -40 or u_lo > W + 40 or z_front > 30:
                continue
            items.append((z_front, o, u_lo, u_hi, rng_near))
        items.sort(key=lambda t: -t[0])                 # far -> near
        records, mirror = [], []
        for k, (z_front, o, u_lo, u_hi, rng_near) in enumerate(items):
            cand = self._sprites_for(o)
            ext_w = max(u_hi - u_lo, 2.0)
            width_m = ext_w * z_front / f
            # choose the view whose proportions fit the footprint best
            best = self.current[o.uid]
            if len(cand) > 1:
                def err(sp):
                    return abs(math.log(max(sp.aspect * sp.height_m, 1e-3)
                                        / max(width_m, 1e-3)))
                j = min(range(len(cand)), key=lambda i: err(cand[i]))
                if err(cand[j]) < err(cand[best]) - 0.18:   # hysteresis
                    best = j
                self.current[o.uid] = best
            sp = cand[best]
            if sp.cls == "package":
                h_m = float(np.clip(width_m / sp.aspect, 0.55, 1.25))
            elif sp.cls == "unknown":
                h_m = float(min(sp.height_m, 1.25 * width_m / sp.aspect))
            else:
                h_m = sp.height_m
            h_px = f * h_m / z_front
            w_nat = sp.aspect * h_px
            w_px = float(np.clip(ext_w, 0.78 * w_nat, 1.30 * w_nat))
            uc = 0.5 * (u_lo + u_hi)
            x0 = uc - w_px / 2
            y1 = self.cy + f * self.cam_h / z_front          # feet row
            y0 = y1 - h_px
            # visible rectangle in the frame
            ix0, ix1 = int(max(math.floor(x0), 0)), int(min(math.ceil(
                x0 + w_px), W))
            iy0, iy1 = int(max(math.floor(y0), 0)), int(min(math.ceil(y1),
                                                            H))
            if ix1 - ix0 < 2 or iy1 - iy0 < 2:
                continue
            mips = sp.mips()
            sh, sw = mips[0].shape[:2]
            scale = h_px / sh
            lvl = int(np.clip(math.floor(math.log2(max(1.0 / max(
                scale, 1e-6), 1.0))), 0, len(mips) - 1))
            src = mips[lvl]
            sy = src.shape[0] / h_px
            sx = src.shape[1] / w_px
            flip = self.flip[o.uid]
            # dst (x, y) in the visible rect -> source pixel
            ty = (iy0 - y0 + 0.5) * sy - 0.5
            if flip:
                M = np.array([[-sx, 0, (x0 + w_px - ix0 - 0.5) * sx - 0.5],
                              [0, sy, ty]], np.float32)
            else:
                M = np.array([[sx, 0, (ix0 - x0 + 0.5) * sx - 0.5],
                              [0, sy, ty]], np.float32)
            patch = cv2.warpAffine(
                src, M, (ix1 - ix0, iy1 - iy0),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            alpha = patch[..., 3]
            total = float((alpha > 0.5).sum())
            # full silhouette size incl. parts outside the frame
            full_px = total / max(((ix1 - ix0) * (iy1 - iy0))
                                  / max(w_px * h_px, 1.0), 1e-3)
            roi_z = zbuf[iy0:iy1, ix0:ix1]
            vis = (roi_z > z_front) & (alpha > 0.02)
            a = np.where(vis, alpha, 0.0)[..., None]
            roi = img[iy0:iy1, ix0:ix1]
            # contact shadow on the floor under the object
            if self.effects and y1 < H + 20:
                self._contact_shadow(img, zbuf, uc, y1, w_px, z_front)
            shade = 0.62 + 0.42 * math.exp(-z_front / 11.0)
            roi[:] = roi * (1 - a) + patch[..., :3] * shade * (a > 0)
            solid = vis & (alpha > 0.5)
            roi_z[solid] = z_front
            ids[iy0:iy1, ix0:ix1][solid] = len(records)
            groi = glint[iy0:iy1, ix0:ix1]
            groi *= (1 - a[..., 0])            # nearer object hides glints
            if getattr(o, "reflective", False) and solid.any():
                self._glints(roi, groi, solid)
                mirror.append((len(records), ix0, ix1, iy0, iy1))
            records.append({
                "uid": int(o.uid), "category": int(o.category),
                "class_name": sp.cls, "name": sp.name, "sprite": sp.sid,
                "z": float(z_front), "range": float(rng_near),
                "full_box": [float(max(x0, 0)), float(max(y0, 0)),
                             float(min(x0 + w_px, W)), float(min(y1, H))],
                "full_px": full_px, "in_frame_px": total,
                "bottom_cut": bool(y1 > H),
                "reflective": bool(getattr(o, "reflective", False))})
        # visible boxes after all occlusions
        gt = []
        for i, r in enumerate(records):
            x0, y0, x1, y1 = [int(v) for v in r["full_box"]]
            m = ids[y0:y1 + 1, x0:x1 + 1] == i
            n = int(m.sum())
            if n == 0:
                continue
            ys, xs = np.where(m)
            r["bbox"] = [float(x0 + xs.min()), float(y0 + ys.min()),
                         float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)]
            r["visible"] = min(n / max(r["full_px"], 1.0), 1.0)
            gt.append(r)
        return gt, ids, mirror, glint

    def _contact_shadow(self, img, zbuf, uc, y1, w_px, z_front):
        H, W = self.h, self.w
        rx = max(0.62 * w_px, 3.0)
        ry = max(0.16 * w_px, 2.0)
        x0, x1 = int(max(uc - 1.6 * rx, 0)), int(min(uc + 1.6 * rx, W))
        y0, y1i = int(max(y1 - 1.3 * ry, 0)), int(min(y1 + 1.6 * ry, H))
        if x1 - x0 < 2 or y1i - y0 < 2:
            return
        xs = (np.arange(x0, x1, dtype=np.float32) - uc) / rx
        ys = (np.arange(y0, y1i, dtype=np.float32) - y1) / ry
        g = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) * 1.4)
        # only darken floor pixels at the object's distance (not walls
        # in front, not things closer to the camera)
        ok = np.abs(zbuf[y0:y1i, x0:x1] - z_front) < 0.45 * z_front + 0.3
        img[y0:y1i, x0:x1] *= (1.0 - 0.5 * g * ok)[..., None]

    def _glints(self, roi, groi, solid):
        """Specular highlights of stretch wrap: small saturated streaks
        that jump around from frame to frame (written to the glint layer,
        which is composited last so only glints saturate the sensor)."""
        h, w = solid.shape
        rng = self.rng
        ys, xs = np.where(solid)
        for _ in range(int(rng.integers(2, 6))):
            i = int(rng.integers(len(xs)))
            cx, cy = xs[i], ys[i]
            sx = max(1.6, 0.035 * w * rng.uniform(0.6, 1.6))
            sy = max(3.0, 0.16 * h * rng.uniform(0.5, 1.5))
            x0, x1 = int(max(cx - 3 * sx, 0)), int(min(cx + 3 * sx + 1, w))
            y0, y1 = int(max(cy - 3 * sy, 0)), int(min(cy + 3 * sy + 1, h))
            gx = np.exp(-((np.arange(x0, x1) - cx) / sx) ** 2)
            gy = np.exp(-((np.arange(y0, y1) - cy) / sy) ** 2)
            g = np.clip(1.7 * gy[:, None] * gx[None, :], 0, 1) \
                * solid[y0:y1, x0:x1]
            groi[y0:y1, x0:x1] = np.maximum(groi[y0:y1, x0:x1], g)
        # the film also washes the colours out a little
        roi[solid] = roi[solid] * 0.9 + 22.0

    # ------------------------------------------------------------------
    def _sense_depth(self, env, z, ids, mirror, rng):
        """Depth-sensor model: correlated range noise, far readings on
        stretch wrap (the sensor 'sees through' the film), phantom near
        returns from shadows. Same three effects as the ray emulation."""
        H, W = self.h, self.w
        sigma = float(getattr(env, "depth_noise", 0.03))
        k = int(rng.integers(len(self._lowres)))
        low = cv2.resize(np.roll(self._lowres[k], int(rng.integers(40)),
                                 axis=1), (W, H),
                         interpolation=cv2.INTER_LINEAR)
        fine = np.roll(self._noise[int(rng.integers(len(self._noise)))],
                       int(rng.integers(W)), axis=1)
        d = z * (1.0 + sigma * (0.9 * low + 0.35 * fine))
        fail_p = float(env.cfg.mirror_fail_prob)
        band = max(int(self.f * math.tan(math.radians(env.cfg.fov_deg)
                                         / (env.cfg.n_rays - 1))), 4)
        for rec_i, x0, x1, y0, y1 in mirror:
            for bx in range(x0 - (x0 % band), x1, band):
                if rng.random() < fail_p:
                    a, b = max(bx, x0), min(bx + band, x1)
                    m = ids[y0:y1, a:b] == rec_i
                    d[y0:y1, a:b][m] += rng.uniform(1.5, 4.0)
        shadow_p = float(getattr(env, "shadow_prob", 0.0))
        for _ in range(int(rng.binomial(env.cfg.n_rays, shadow_p))):
            r = float(rng.uniform(0.6, 2.5))
            uc = int(rng.integers(band, W - band))
            zc = r * float(self.cos_a[uc])
            yb = int(min(self.cy + self.f * self.cam_h / zc, H))
            yt = int(max(self.cy + self.f * (self.cam_h - 0.6) / zc, 0))
            a, b = uc - band // 2, uc + band // 2 + 1
            d[yt:yb, a:b] = np.minimum(d[yt:yb, a:b], zc)
        return np.clip(d, 0.05, 40.0).astype(np.float32)

    # ------------------------------------------------------------------
    def render(self, env, rng=None):
        """Render the robot's view. `rng` drives the sensor noise."""
        rng = rng if rng is not None else self.rng
        img, z, cast = self._render_static(env)
        gt, ids, mirror, glint = self._draw_sprites(env, img, z)
        depth_true = z.copy()
        depth = self._sense_depth(env, z, ids, mirror, rng)

        img *= self._colour
        if self.effects:
            # motion blur while turning (short exposure)
            k = int(round(abs(getattr(env, "omega", 0.0)) * self.f / 55.0))
            if k >= 2:
                img = cv2.blur(img, (min(k, 9), 1))
                glint = cv2.blur(glint, (min(k, 9), 1))
            n = self._noise[self.frame_idx % len(self._noise)]
            img += 2.2 * np.roll(n, (self.frame_idx * 37) % self.w,
                                 axis=1)[..., None]
            img = cv2.GaussianBlur(img, (3, 3), 0.55)
        # diffuse surfaces never saturate; specular glints do
        np.minimum(img, SPECULAR_LEVEL, out=img)
        if glint.any():
            g = glint[..., None]
            img = img * (1 - g) + 262.0 * g
        self.frame_idx += 1
        rgb = np.clip(img, 0, 255).astype(np.uint8)
        self.last_cast = cast
        return CameraFrame(rgb, depth, depth_true, gt)
