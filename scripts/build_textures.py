"""
Rectify flat surfaces of real warehouse photos (rack faces, floor, walls,
ceiling) into textures for the real-image camera simulation.

The source photos are wide, oblique shots from the detector's training set
(Ultralytics construction-ppe). A rack face is a plane, so its image can be
warped to a fronto-parallel texture with a homography. The four corners of
each face are not clicked by hand: the script detects the rack's orange
beams and its uprights, estimates the two vanishing points of the face
(RANSAC over Hough segments) and builds the quad from them, so beams come
out horizontal and uprights vertical.

The floor is rectified metrically from the two horizontal vanishing points
of the rack (they give the focal length and the floor's normal).

Regions are listed in assets/sim/textures/sources.json.

    python scripts/build_textures.py --src data_downloads \
        --out assets/sim/textures [--debug /tmp/tex_debug]

Only needed to REBUILD the textures; the demo reads the committed JPEGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Vanishing points
# ----------------------------------------------------------------------
def _hough(mask, thr, min_len, gap):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=thr,
                           minLineLength=min_len, maxLineGap=gap)
    if segs is None:
        return np.zeros((0, 4))
    return np.asarray(segs, dtype=float).reshape(-1, 4)   # OpenCV 4 and 5


def _fit_vp(segs):
    A = []
    for x1, y1, x2, y2 in segs:
        line = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        A.append(line / np.hypot(line[0], line[1])
                 * np.hypot(x2 - x1, y2 - y1))
    v = np.linalg.svd(np.array(A))[2][-1]
    return v / v[2] if abs(v[2]) > 1e-12 else v


def _ransac_vp(segs, tol_deg, iters=800, seed=0):
    rng = np.random.default_rng(seed)
    lines = np.array([np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])
                      for s in segs])
    lines /= np.hypot(lines[:, 0], lines[:, 1])[:, None]
    mids = (segs[:, :2] + segs[:, 2:]) / 2
    d_seg = segs[:, 2:] - segs[:, :2]
    lens = np.linalg.norm(d_seg, axis=1)
    d_seg = d_seg / lens[:, None]
    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if np.abs(v).max() < 1e-12:
            continue
        d = (v[:2] / v[2] - mids) if abs(v[2]) > 1e-12 \
            else np.tile(v[:2], (len(segs), 1))
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        ang = np.degrees(np.arcsin(np.clip(np.abs(
            d_seg[:, 0] * d[:, 1] - d_seg[:, 1] * d[:, 0]), 0, 1)))
        inl = ang < tol_deg
        if lens[inl].sum() > best_score:
            best_score, best = lens[inl].sum(), inl
    return _fit_vp(segs[best]), best


def vanishing_points(img, roi, debug=None):
    """Vanishing points of a rack block: beams rising to the right
    ("neg" image slope), beams falling to the right ("pos") and uprights
    ("vert")."""
    x0, y0, x1, y1 = roi
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = (((hsv[..., 0] <= 13) | (hsv[..., 0] >= 172))
              & (hsv[..., 1] > 105) & (hsv[..., 2] > 80))
    m = np.zeros(img.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = orange[y0:y1, x0:x1] * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    beams = _hough(m, 80, 140, 25)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160)
    e = np.zeros_like(edges)
    e[y0:y1, x0:x1] = edges[y0:y1, x0:x1]
    up = _hough(e, 90, 200, 12)
    if len(up):
        a = np.degrees(np.arctan2(up[:, 2] - up[:, 0], up[:, 3] - up[:, 1]))
        up = up[(np.abs(a) < 9) | (np.abs(np.abs(a) - 180) < 9)]
    out, vis = {}, (img.copy() if debug else None)
    slope = (beams[:, 3] - beams[:, 1]) / (beams[:, 2] - beams[:, 0] + 1e-9)
    for name, fam, tol, col in (("pos", beams[slope > 0.02], 3.0,
                                 (0, 255, 0)),
                                ("neg", beams[slope < -0.02], 3.0,
                                 (255, 0, 255)),
                                ("vert", up, 1.5, (255, 255, 0))):
        if len(fam) < 4:
            continue
        vp, inl = _ransac_vp(fam, tol)
        out[name] = vp
        if vis is not None:
            for s in fam[inl]:
                cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])),
                         col, 2)
    if "vert" not in out:           # uprights parallel in the image
        out["vert"] = np.array([0.0, 1.0, 0.0])
    return out, vis


# ----------------------------------------------------------------------
# Geometry helpers (homogeneous 2D)
# ----------------------------------------------------------------------
def _h(p):
    p = np.asarray(p, float)
    return p if len(p) == 3 else np.array([p[0], p[1], 1.0])


def _line(p, q):
    return np.cross(_h(p), _h(q))


def _meet(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def _at_x(line, x):
    return np.array([x, -(line[0] * x + line[2]) / line[1]])


def _at_y(line, y):
    return np.array([-(line[1] * y + line[2]) / line[0], y])


def face_quad(vps, family, floor_pt, x_left, x_right, top_y, near):
    """Corners [TL, TR, BR, BL] of a rack face between two uprights.

    floor_pt : a point on the face's floor line
    x_left / x_right : image x of the left / right edge at the floor
    top_y : image y of the top edge where it meets the *near* edge
    near  : "left" or "right" (the edge closer to the camera)
    """
    vp_h, vp_v = vps[family], vps["vert"]
    floor = _line(floor_pt, vp_h)
    bl, br = _at_x(floor, x_left), _at_x(floor, x_right)
    left, right = _line(bl, vp_v), _line(br, vp_v)
    top = _line(_at_y(left if near == "left" else right, top_y), vp_h)
    return np.array([_meet(top, left), _meet(top, right), br, bl],
                    np.float32)


def _dst(size):
    w, h = size
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    np.float32)


def warp_quad(img, quad, size):
    w, h = size
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), _dst(size))
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_AREA
                               if w * h < 0.5 * cv2.contourArea(quad)
                               else cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _back(H, pts):
    p = cv2.perspectiveTransform(np.array(pts, np.float32)[None],
                                 np.linalg.inv(H))[0]
    return p


def level_face(img, quad, size, near, iters=3):
    """Refine a rack-face quad on the face's own lines.

    The vanishing points come from the whole rack block; this second pass
    looks only at the rectified face: the orange beams must be horizontal
    and the uprights vertical. The edge next to the camera (best resolved)
    stays fixed and the far edge is moved.
    """
    w, h = size
    quad = quad.astype(np.float32).copy()
    for _ in range(iters):
        H = cv2.getPerspectiveTransform(quad, _dst(size))
        tex = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        hsv = cv2.cvtColor(tex, cv2.COLOR_BGR2HSV)
        m = (((hsv[..., 0] <= 13) | (hsv[..., 0] >= 172))
             & (hsv[..., 1] > 105) & (hsv[..., 2] > 80)).astype(np.uint8)
        m = cv2.morphologyEx(m * 255, cv2.MORPH_CLOSE,
                             np.ones((3, 9), np.uint8))
        seg = _hough(m, 40, int(0.22 * w), 20)
        moved = 0.0
        if len(seg):
            sl = (seg[:, 3] - seg[:, 1]) / (seg[:, 2] - seg[:, 0] + 1e-9)
            seg, sl = seg[np.abs(sl) < 0.2], sl[np.abs(sl) < 0.2]
        if len(seg) >= 4:
            y_l = seg[:, 1] - sl * seg[:, 0]
            y_r = seg[:, 1] + sl * (w - 1 - seg[:, 0])
            wt = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
            if near == "left":       # y_r = a * y_l + b ; move right edge
                a, b = np.polyfit(y_l, y_r, 1, w=wt)
                new = _back(H, [[w - 1, b], [w - 1, a * (h - 1) + b]])
                moved = max(moved, abs(b), abs(a * (h - 1) + b - (h - 1)))
                quad[1], quad[2] = new[0], new[1]
            else:
                a, b = np.polyfit(y_r, y_l, 1, w=wt)
                new = _back(H, [[0, b], [0, a * (h - 1) + b]])
                moved = max(moved, abs(b), abs(a * (h - 1) + b - (h - 1)))
                quad[0], quad[3] = new[0], new[1]
        if moved < 1.0:
            break
    return quad


# ----------------------------------------------------------------------
# Floor: metric rectification from two orthogonal vanishing points
# ----------------------------------------------------------------------
def floor_rectifier(vps, shape):
    """Returns f(points_px) -> ground coordinates (arbitrary scale)."""
    h, w = shape[:2]
    c = np.array([w / 2, h / 2])
    v1, v2 = vps["pos"][:2], vps["neg"][:2]
    f2 = -np.dot(v1 - c, v2 - c)
    f = float(np.sqrt(f2)) if f2 > 0 else float(max(h, w))
    Kinv = np.array([[1 / f, 0, -c[0] / f], [0, 1 / f, -c[1] / f],
                     [0, 0, 1]])
    d1 = Kinv @ _h(v1)
    d1 /= np.linalg.norm(d1)
    d2 = Kinv @ _h(v2)
    d2 -= d1 * (d1 @ d2)             # re-orthogonalise
    d2 /= np.linalg.norm(d2)
    n = np.cross(d1, d2)
    if n[1] < 0:                     # floor normal points down in the image
        n = -n

    def to_ground(pts):
        rays = (Kinv @ np.c_[np.asarray(pts, float),
                             np.ones(len(pts))].T).T
        t = 1.0 / (rays @ n)
        return np.stack([(rays @ d1) * t, (rays @ d2) * t], axis=1)
    return to_ground, f


def rectify_floor(img, vps, region, scale_ref, px_per_m, max_aspect=1.6):
    """Top-down floor texture: (texture, (width_m, height_m), focal_px).

    region    : polygon (image px) containing only floor
    scale_ref : ((x1, y1), (x2, y2), metres) known floor distance
    """
    to_ground, f = floor_rectifier(vps, img.shape)
    (p1, p2, metres) = scale_ref
    g = to_ground([p1, p2])
    s = metres / np.linalg.norm(g[0] - g[1])           # ground units -> m
    poly = np.array(region, np.float32)
    gp = to_ground(poly) * s
    lo = gp.min(axis=0)
    # homography image -> texture from 4+ correspondences
    dst = ((gp - lo) * px_per_m).astype(np.float32)
    H, _ = cv2.findHomography(poly, dst)
    full = cv2.warpPerspective(img, H, (int(dst[:, 0].max()) + 1,
                                        int(dst[:, 1].max()) + 1),
                               flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(
        cv2.fillPoly(np.zeros(img.shape[:2], np.uint8),
                     [poly.astype(np.int32)], 255), H,
        (full.shape[1], full.shape[0]))
    # largest axis-aligned rectangle fully inside the floor region
    # (maximal rectangle in a binary matrix, histogram + stack)
    ok = mask > 250
    heights = np.zeros(ok.shape[1], int)
    best = (0, 0, 0, 0, 0)                      # area, x0, y0, x1, y1
    for y in range(ok.shape[0]):
        heights = np.where(ok[y], heights + 1, 0)
        stack = []
        for x in range(ok.shape[1] + 1):
            cur = heights[x] if x < ok.shape[1] else 0
            start = x
            while stack and stack[-1][1] >= cur:
                start, hh = stack.pop()
                # keep it roughly square so it tiles in both directions
                ww = min(x - start, int(max_aspect * hh))
                if ww * hh > best[0]:
                    best = (ww * hh, start, y - hh + 1, start + ww, y + 1)
            stack.append((start, cur))
    _, x0, y0, x1, y1 = best
    tex = full[y0:y1, x0:x1]
    return tex, ((x1 - x0) / px_per_m, (y1 - y0) / px_per_m), f


def flatten_light(tex, sigma_frac=0.25, keep=0.35):
    """Remove most of the baked-in large-scale lighting so the texture
    tiles; the renderer adds its own lighting."""
    t = tex.astype(np.float32)
    s = max(3, int(sigma_frac * min(tex.shape[:2])) | 1)
    low = cv2.GaussianBlur(t, (0, 0), s / 3)
    mean = t.reshape(-1, 3).mean(axis=0)
    out = t - (1 - keep) * (low - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
def find(src, name):
    for p in Path(src).rglob(name):
        return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="assets/sim/textures")
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    cfg = json.loads((out / "sources.json").read_text())
    dbg = Path(a.debug) if a.debug else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
    index = {"note": "Rectified from real warehouse photos of the "
                     "training set. See assets/sim/ATTRIBUTION.md.",
             "textures": []}
    cache = {}
    for t in cfg["textures"]:
        img_path = find(a.src, t["image"])
        key = (t["image"], tuple(t.get("vp_roi", [])))
        if key not in cache:
            img = cv2.imread(str(img_path))
            vps, vis = (vanishing_points(img, t["vp_roi"], dbg)
                        if "vp_roi" in t else ({}, None))
            cache[key] = (img, vps, vis)
        img, vps, vis = cache[key]
        if "vp_v" in t:                 # uprights measured by hand
            vps = dict(vps, vert=np.array([*t["vp_v"], 1.0]))
        if "patch" in t:                # cover a person standing in front
            img = img.copy()
            for x0, y0, x1, y1, sx, sy in t["patch"]:
                img[y0:y1, x0:x1] = img[sy:sy + y1 - y0, sx:sx + x1 - x0]
        ppm = t.get("px_per_m", 72)
        rec = {"name": t["name"], "kind": t["kind"], "style": t["style"],
               "file": f"{t['name']}.jpg", "source": t["image"]}
        if t["kind"] in ("rack", "rack_end", "wall"):
            if "quad" in t:
                quad = np.array(t["quad"], np.float32)
            else:
                quad = face_quad(vps, t["family"], t["floor_pt"],
                                 t["x_left"], t["x_right"], t["top_y"],
                                 t["near"])
            size = (int(round(t["width_m"] * ppm)),
                    int(round(t["height_m"] * ppm)))
            if t["kind"] == "rack" and t.get("level", True):
                quad = level_face(img, quad, size, t.get("near", "left"))
            tex = warp_quad(img, quad, size)
            if t.get("flip"):
                tex = tex[:, ::-1]
            rec.update(width_m=t["width_m"], height_m=t["height_m"],
                       quad=[[round(float(v), 1) for v in p] for p in quad])
            if vis is not None:
                cv2.polylines(vis, [quad.astype(np.int32)], True,
                              (0, 0, 255), 3)
        elif t["kind"] in ("floor", "ceiling"):
            if "rect" in t:                     # plain crop, no rectification
                x0, y0, x1, y1 = t["rect"]
                tex = cv2.resize(
                    img[y0:y1, x0:x1],
                    (int(t["size_m"][0] * ppm), int(t["size_m"][1] * ppm)),
                    interpolation=cv2.INTER_AREA)
            else:
                tex, size_m, f = rectify_floor(
                    img, vps, t["region"],
                    (t["scale_ref"][0], t["scale_ref"][1], t["scale_ref"][2]),
                    ppm, t.get("max_aspect", 1.6))
                t = dict(t, size_m=[round(v, 2) for v in size_m])
                print(f"  {t['name']}: focal {f:.0f}px, "
                      f"{size_m[0]:.1f} x {size_m[1]:.1f} m of floor")
                if vis is not None:
                    cv2.polylines(vis, [np.array(t["region"], np.int32)],
                                  True, (0, 255, 255), 3)
            tex = flatten_light(tex, keep=t.get("keep_light", 0.35))
            rec.update(width_m=t["size_m"][0], height_m=t["size_m"][1])
        else:
            raise ValueError(t["kind"])
        gain = t.get("gain", 1.0)
        if gain != 1.0:
            tex = np.clip(tex.astype(np.float32) * gain, 0, 255).astype(
                np.uint8)
        cv2.imwrite(str(out / rec["file"]), tex,
                    [cv2.IMWRITE_JPEG_QUALITY, t.get("quality", 86)])
        rec["w"], rec["h"] = int(tex.shape[1]), int(tex.shape[0])
        index["textures"].append(rec)
        print(f"{rec['name']:14s} {rec['w']}x{rec['h']}  "
              f"{(out / rec['file']).stat().st_size / 1024:.0f} KB")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    if dbg:
        for (name, _), (_, _, vis) in cache.items():
            if vis is not None:
                cv2.imwrite(str(dbg / f"vp_{Path(name).stem[-8:]}.jpg"),
                            cv2.resize(vis, None, fx=0.6, fy=0.6))


if __name__ == "__main__":
    main()
