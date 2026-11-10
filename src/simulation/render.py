"""
OpenCV renderer for the warehouse simulation.

Produces one RGB frame composed of:

* a top-down warehouse map (racks, obstacles, robot, camera FOV, fused
  depth rays, global path, goal),
* a simulated first-person camera view with detector boxes,
* the simulated depth map strip (what the depth branch sees),
* a HUD with speed, action, shield decision and uncertainty gauge.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.simulation.warehouse_env import (
    ACTIONS, BOX, CATEGORY_NAMES, FORKLIFT, PALLET, PERSON, RACK, WALL,
)

SCALE = 36            # pixels per metre on the map
MAP_PAD = 10
PANEL_W = 420

# BGR -> we draw in RGB order directly (frames are returned as RGB)
COL = {
    "floor": (236, 233, 226),
    "grid": (222, 218, 210),
    "rack": (88, 76, 66),
    "rack_shelf": (150, 110, 70),
    "pallet": (196, 150, 92),
    "pallet_refl": (150, 205, 235),
    "box": (205, 165, 105),
    "person": (235, 110, 40),
    "forklift": (240, 190, 30),
    "robot": (30, 100, 220),
    "goal": (40, 170, 90),
    "path": (90, 150, 230),
    "text": (30, 30, 35),
    "muted": (110, 110, 120),
    "panel": (248, 248, 250),
    "green": (40, 170, 90),
    "amber": (235, 160, 30),
    "red": (215, 50, 50),
}

CLASS_COL = {
    PALLET: COL["pallet"],
    BOX: COL["box"],
    PERSON: COL["person"],
    FORKLIFT: COL["forklift"],
    RACK: COL["rack"],
    WALL: (120, 120, 120),
}


def _to_px(env, x, y):
    h = env.cfg.height
    return (int(MAP_PAD + x * SCALE), int(MAP_PAD + (h - y) * SCALE))


def _rect(img, env, o, color, thickness=-1):
    p1 = _to_px(env, o.x - o.hw, o.y + o.hh)
    p2 = _to_px(env, o.x + o.hw, o.y - o.hh)
    cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def _text(img, s, org, scale=0.5, color=COL["text"], thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def _unc_color(u):
    if u < 0.35:
        return COL["green"]
    if u < 0.75:
        return COL["amber"]
    return COL["red"]


# ----------------------------------------------------------------------
def draw_map(env, trail=None, show_rays=True):
    c = env.cfg
    W = int(c.width * SCALE) + 2 * MAP_PAD
    H = int(c.height * SCALE) + 2 * MAP_PAD
    img = np.full((H, W, 3), COL["floor"], np.uint8)

    # floor grid (1 m)
    for gx in range(int(c.width) + 1):
        cv2.line(img, _to_px(env, gx, 0), _to_px(env, gx, c.height),
                 COL["grid"], 1)
    for gy in range(int(c.height) + 1):
        cv2.line(img, _to_px(env, 0, gy), _to_px(env, c.width, gy),
                 COL["grid"], 1)
    # staging areas
    for x0, x1 in ((0, env.x_start - 0.3), (env.x_end + 0.3, c.width)):
        p1, p2 = _to_px(env, x0, c.height), _to_px(env, x1, 0)
        overlay = img.copy()
        cv2.rectangle(overlay, p1, p2, (225, 232, 240), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # global path
    path = env.global_path(env.x, env.y, max_len=400)
    pts = np.array([_to_px(env, px, py) for px, py in path], np.int32)
    for k in range(0, len(pts) - 1, 2):
        cv2.line(img, tuple(pts[k]), tuple(pts[k + 1]), COL["path"], 2,
                 cv2.LINE_AA)

    # goal
    gx, gy = _to_px(env, *env.goal)
    cv2.circle(img, (gx, gy), int(c.goal_radius * SCALE), COL["goal"], 2,
               cv2.LINE_AA)
    cv2.drawMarker(img, (gx, gy), COL["goal"], cv2.MARKER_STAR, 18, 2)
    _text(img, "GOAL", (gx - 18, gy - 16), 0.45, COL["goal"], 1)

    # trail
    if trail and len(trail) > 1:
        tp = np.array([_to_px(env, x, y) for x, y in trail], np.int32)
        cv2.polylines(img, [tp], False, (150, 175, 225), 2, cv2.LINE_AA)

    # FOV cone + fused rays
    rx, ry = _to_px(env, env.x, env.y)
    if show_rays:
        overlay = img.copy()
        cone = [(rx, ry)]
        for a in np.linspace(env.ray_angles[0], env.ray_angles[-1], 25):
            cone.append(_to_px(env, env.x + c.max_range * math.cos(
                env.theta + a), env.y + c.max_range * math.sin(
                env.theta + a)))
        cv2.fillPoly(overlay, [np.array(cone, np.int32)], (205, 220, 245))
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        for d, a, lab in zip(env.fused_rays, env.ray_angles, env.ray_labels):
            ex = env.x + d * math.cos(env.theta + a)
            ey = env.y + d * math.sin(env.theta + a)
            t = min(d / 3.0, 1.0)
            col = (int(215 * (1 - t) + 60 * t), int(50 * (1 - t) + 170 * t),
                   int(50 * (1 - t) + 90 * t))
            cv2.line(img, (rx, ry), _to_px(env, ex, ey), col, 1, cv2.LINE_AA)
            if d < c.max_range - 0.05:
                cv2.circle(img, _to_px(env, ex, ey), 3, col, -1, cv2.LINE_AA)

    # racks with shelf stripes
    for o in env.racks:
        _rect(img, env, o, COL["rack"])
        n = int(o.hw * 2 / 1.2)
        for k in range(1, n + 1):
            xk = o.x - o.hw + k * (2 * o.hw / (n + 1))
            cv2.line(img, _to_px(env, xk, o.y - o.hh + 0.08),
                     _to_px(env, xk, o.y + o.hh - 0.08), COL["rack_shelf"], 1)

    # static obstacles
    for o in env.statics:
        if o.category == PALLET:
            col = COL["pallet_refl"] if o.reflective else COL["pallet"]
            _rect(img, env, o, col)
            _rect(img, env, o, (120, 90, 50), 1)
            if o.reflective:
                p = _to_px(env, o.x - o.hw * 0.6, o.y + o.hh * 0.5)
                q = _to_px(env, o.x + o.hw * 0.2, o.y - o.hh * 0.4)
                cv2.line(img, p, q, (255, 255, 255), 2, cv2.LINE_AA)
        elif o.category == BOX:
            _rect(img, env, o, COL["box"])
            _rect(img, env, o, (120, 90, 50), 1)
            px, py = _to_px(env, o.x, o.y)
            _text(img, "?", (px - 5, py + 6), 0.55, (90, 60, 30), 2)
        elif o.category == PERSON:
            px, py = _to_px(env, o.x, o.y)
            cv2.circle(img, (px, py), int(o.radius * SCALE), COL["person"],
                       -1, cv2.LINE_AA)

    # dynamic obstacles
    for o in env.dynamics:
        if o.category == PERSON:
            px, py = _to_px(env, o.x, o.y)
            cv2.circle(img, (px, py), int(o.radius * SCALE), COL["person"],
                       -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), int(o.radius * SCALE * 0.45),
                       (250, 205, 170), -1, cv2.LINE_AA)
            sp = math.hypot(o.vx, o.vy) + 1e-6
            tip = _to_px(env, o.x + o.vx / sp * 0.5, o.y + o.vy / sp * 0.5)
            cv2.arrowedLine(img, (px, py), tip, (160, 70, 20), 2,
                            cv2.LINE_AA, tipLength=0.4)
        elif o.category == FORKLIFT:
            _rect(img, env, o, COL["forklift"])
            _rect(img, env, o, (130, 100, 20), 1)
            d = 1 if o.vx >= 0 else -1
            for off in (-0.25, 0.25):
                p = _to_px(env, o.x + d * o.hw, o.y + off)
                q = _to_px(env, o.x + d * (o.hw + 0.5), o.y + off)
                cv2.line(img, p, q, (90, 90, 90), 3)

    # labels of detected / perceived obstacles
    seen = {}
    for d, lab, hit in zip(env.fused_rays, env.ray_labels, env.ray_hits):
        if hit is None or hit.category in (RACK, WALL):
            continue
        if hit.uid not in seen or d < seen[hit.uid][0]:
            seen[hit.uid] = (d, lab, hit)
    for d, lab, hit in seen.values():
        px, py = _to_px(env, hit.x, hit.y)
        name = CATEGORY_NAMES.get(int(lab), "?") if lab >= 0 else "?"
        s = f"{name} {d:.1f}m"
        _text(img, s, (px - 30, py - int(max(hit.hh, hit.radius) * SCALE)
                       - 6), 0.4, (20, 20, 20), 1)

    # robot
    r = int(c.robot_radius * SCALE)
    col = (215, 50, 50) if env.collided else COL["robot"]
    cv2.circle(img, (rx, ry), r, col, -1, cv2.LINE_AA)
    cv2.circle(img, (rx, ry), r, (15, 50, 120), 2, cv2.LINE_AA)
    hx, hy = _to_px(env, env.x + c.robot_radius * 1.3 * math.cos(env.theta),
                    env.y + c.robot_radius * 1.3 * math.sin(env.theta))
    cv2.line(img, (rx, ry), (hx, hy), (255, 255, 255), 3, cv2.LINE_AA)
    if env.collided:
        _text(img, "COLLISION", (rx - 45, ry - r - 8), 0.6, COL["red"], 2)
    if env.reached:
        _text(img, "GOAL REACHED", (rx - 60, ry - r - 8), 0.6,
              COL["goal"], 2)
    return img


# ----------------------------------------------------------------------
def draw_camera(env, w=PANEL_W, h=190, n_cols=140):
    """Pseudo-3D first-person camera view and depth strip."""
    c = env.cfg
    half = math.radians(c.fov_deg / 2)
    # perspective-correct column angles
    u = np.linspace(-1, 1, n_cols)
    angles = -np.arctan(u * math.tan(half))  # left of image = +angle
    d, hit = env.cast(angles)
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.48)
    img[:horizon] = (205, 210, 220)
    for yy in range(horizon, h):
        t = (yy - horizon) / (h - horizon)
        img[yy] = (int(185 + 40 * t), int(180 + 40 * t), int(170 + 40 * t))
    col_w = w / n_cols
    boxes = {}
    for k in range(n_cols):
        dist = d[k] * math.cos(angles[k])
        o = hit[k]
        if o is None:
            continue
        hgt = {RACK: 2.6, WALL: 3.0, PALLET: 1.2, BOX: 0.5, PERSON: 1.75,
               FORKLIFT: 2.1}[o.category]
        f = h * 0.55
        floor_y = horizon + f * 0.9 / max(dist, 0.2)
        top_y = floor_y - f * hgt / max(dist, 0.2)
        base = CLASS_COL[o.category]
        if o.category == PALLET and o.reflective:
            base = COL["pallet_refl"]
        shade = max(0.35, 1.0 - dist / (c.max_range * 1.3))
        colr = tuple(int(ch * shade) for ch in base)
        x0, x1 = int(k * col_w), int((k + 1) * col_w) + 1
        y0, y1 = int(max(top_y, 0)), int(min(floor_y, h - 1))
        img[y0:y1, x0:x1] = colr
        if o.category == RACK:
            for sh in (0.33, 0.66):
                ys = int(floor_y - (floor_y - top_y) * sh)
                if 0 <= ys < h:
                    img[ys:ys + 2, x0:x1] = (150, 110, 70)
        if o.category == PALLET and o.reflective and k % 7 == 0:
            ys = int((y0 + y1) / 2)
            img[max(ys - 3, 0):ys + 3, x0:x1] = (255, 255, 255)
        if o.category in (PALLET, PERSON, FORKLIFT, BOX):
            b = boxes.setdefault(o.uid, [x0, y0, x1, y1, o, dist])
            b[0], b[1] = min(b[0], x0), min(b[1], y0)
            b[2], b[3] = max(b[2], x1), max(b[3], y1)
            b[5] = min(b[5], dist)
    # detector boxes (only for detected known classes) + depth-only flags
    for x0, y0, x1, y1, o, dist in boxes.values():
        if o.category == BOX:
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 40, 200), 1)
            _text(img, f"unknown(depth) {dist:.1f}m", (x0, max(y0 - 4, 14)),
                  0.38, (200, 40, 200))
        elif env.detections.get(o.uid):
            cv2.rectangle(img, (x0, y0), (x1, y1), (20, 200, 60), 2)
            _text(img, f"{CATEGORY_NAMES[o.category]} {dist:.1f}m",
                  (x0, max(y0 - 4, 14)), 0.4, (20, 140, 40))
    _text(img, "Simulated camera + detector", (6, h - 8), 0.45, (40, 40, 40))
    return img


def draw_depth_strip(env, w=PANEL_W, h=46):
    c = env.cfg
    img = np.zeros((h, w, 3), np.uint8)
    n = len(env.raw_depth)
    cw = w / n
    for i in range(n):
        k = n - 1 - i  # left of image = +angle
        t = 1.0 - min(env.raw_depth[k] / c.max_range, 1.0)
        col = cv2.applyColorMap(np.uint8([[int(t * 255)]]),
                                cv2.COLORMAP_INFERNO)[0, 0][::-1]
        img[:, int(i * cw):int((i + 1) * cw) + 1] = col
        if env.specular[k]:
            cv2.circle(img, (int((i + 0.5) * cw), h // 2), 4,
                       (255, 255, 255), -1)
    _text(img, "Depth branch (near = bright)", (6, 14), 0.42,
          (230, 230, 230))
    return img


def draw_hud(env, w=PANEL_W, h=None, decision=None, controller="",
             reward=0.0, totals=None):
    c = env.cfg
    h = h or 300
    img = np.full((h, w, 3), COL["panel"], np.uint8)
    y = 24
    _text(img, controller, (10, y), 0.55, COL["text"], 2)
    y += 26
    _text(img, f"Speed  {env.v:4.2f} m/s", (10, y), 0.5)
    bar = int((w - 170) * env.v / c.v_max)
    cv2.rectangle(img, (160, y - 12), (160 + bar, y - 2), COL["robot"], -1)
    cv2.rectangle(img, (160, y - 12), (w - 10, y - 2), (180, 180, 190), 1)
    y += 24
    u = env.percep.uncertainty
    _text(img, f"Uncertainty {u:4.2f}", (10, y), 0.5)
    bar = int((w - 170) * min(u, 1))
    cv2.rectangle(img, (160, y - 12), (160 + bar, y - 2), _unc_color(u), -1)
    cv2.rectangle(img, (160, y - 12), (w - 10, y - 2), (180, 180, 190), 1)
    y += 28
    if decision is not None:
        prop, final, reason = decision
        _text(img, f"RL proposes: {prop}", (10, y), 0.5)
        y += 22
        col = COL["text"] if prop == final and reason.startswith("No") \
            else COL["red"]
        _text(img, f"Final action: {final}", (10, y), 0.55, col, 2)
        y += 22
        _text(img, f"Shield: {reason}"[:52], (10, y), 0.45,
              COL["muted"] if reason.startswith("No") else COL["red"])
        y += 26
    else:
        _text(img, f"Action: {ACTIONS[env.last_action]}", (10, y), 0.55,
              COL["text"], 2)
        y += 30
    # sector distance bars (obstacle map)
    _text(img, "Obstacle map (sector distance / class)", (10, y), 0.45,
          COL["muted"])
    y += 8
    ns = len(env.sec_dist)
    bw = (w - 20) / ns
    bh = 60
    for k in range(ns):
        kk = ns - 1 - k
        frac = min(env.sec_dist[kk] / c.max_range, 1.0)
        x0 = int(10 + k * bw + 2)
        x1 = int(10 + (k + 1) * bw - 2)
        top = int(y + bh * (1 - frac))
        cls = env.sec_class[kk]
        col = COL["person"] if cls == 1.0 else (COL["forklift"] if cls == 0.5
                                                else (120, 150, 200))
        if env.sec_dist[kk] < 1.0:
            col = COL["red"]
        cv2.rectangle(img, (x0, y + bh), (x1, top), col, -1)
        _text(img, f"{env.sec_dist[kk]:.1f}", (x0, y + bh + 14), 0.38,
              COL["muted"])
    y += bh + 34
    if totals:
        _text(img, f"Step {env.steps}   reward {reward:+.2f}   "
              f"return {totals.get('return', 0):+.1f}", (10, y), 0.45)
        y += 20
        _text(img, f"Shield interventions {totals.get('interventions', 0)}"
              f"   near-misses {env.near_misses}", (10, y), 0.45)
    return img


def render_frame(env, decision=None, controller="", reward=0.0, totals=None,
                 trail=None):
    m = draw_map(env, trail=trail)
    cam = draw_camera(env)
    strip = draw_depth_strip(env)
    hud_h = max(m.shape[0] - cam.shape[0] - strip.shape[0], 120)
    hud = draw_hud(env, h=hud_h, decision=decision, controller=controller,
                   reward=reward, totals=totals)
    side = np.vstack([cam, strip, hud])
    if side.shape[0] < m.shape[0]:
        pad = np.full((m.shape[0] - side.shape[0], PANEL_W, 3),
                      COL["panel"], np.uint8)
        side = np.vstack([side, pad])
    elif side.shape[0] > m.shape[0]:
        pad = np.full((side.shape[0] - m.shape[0], m.shape[1], 3),
                      COL["floor"], np.uint8)
        m = np.vstack([m, pad])
    sep = np.full((m.shape[0], 4, 3), (200, 200, 205), np.uint8)
    return np.hstack([m, sep, side])
