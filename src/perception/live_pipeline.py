"""
Real-image perception pipeline for the demo (optional heavy dependencies):

YOLO11n detection + Depth Anything V2-Small relative depth + fusion
(project Section 7.1) + a simple specular-highlight uncertainty cue
(Section 7.2).

Depth Anything V2 returns *relative inverse depth* (larger = closer). The
values are not metres; the demo reports them as relative closeness.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.perception.obstacle_fusion import (
    create_driving_corridor_mask,
    create_uncovered_candidate_mask,
    fuse_detections_with_depth,
)

ROOT = Path(__file__).resolve().parents[2]


def _load(cache):
    if "detector" not in cache:
        from ultralytics import YOLO
        custom = ROOT / "weights" / "yolo" / "best.pt"
        cache["detector_name"] = str(custom) if custom.exists() \
            else "yolo11n.pt (COCO-pretrained)"
        cache["detector"] = YOLO(str(custom) if custom.exists()
                                 else "yolo11n.pt")
    if "depth" not in cache:
        try:
            from src.perception.depth_estimator import RelativeDepthEstimator
            cache["depth"] = RelativeDepthEstimator()
            cache["depth_error"] = None
        except Exception as exc:  # no internet / transformers missing
            cache["depth"] = None
            cache["depth_error"] = str(exc).splitlines()[0][:200]
    return cache["detector"], cache["depth"]


def detect(model, image, conf=0.3):
    res = model.predict(source=image, conf=conf, verbose=False)
    dets = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            k = int(b.cls.item())
            dets.append({"class_id": k, "class_name": r.names[k],
                         "confidence": float(b.conf.item()),
                         "bbox": tuple(float(v) for v in b.xyxy[0].tolist())})
    return dets


def unknown_obstacles(depth, corridor, detections, rel_margin=0.18,
                      min_area_frac=0.004):
    """Corridor regions closer than the expected floor, not covered by
    any detector box -> "unknown obstacle" candidates."""
    h, w = depth.shape
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-6)
    cand = create_uncovered_candidate_mask(corridor, detections) > 0
    corr = corridor > 0
    # Expected floor closeness per image row: lower envelope from the
    # bottom of the image upward (floor gets farther as rows go up).
    row_med = np.full(h, np.nan)
    for y in range(h):
        m = corr[y]
        if m.any():
            row_med[y] = np.median(d[y, m])
    floor = np.full(h, np.nan)
    cur = np.inf
    for y in range(h - 1, -1, -1):
        if np.isfinite(row_med[y]):
            cur = min(cur, row_med[y])
            floor[y] = cur
    floor = np.nan_to_num(floor, nan=1.0)
    above = d > (floor[:, None] + rel_margin)
    mask = (cand & above).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * h * w:
            continue
        region = d[lab == i]
        out.append({"class_name": "unknown", "confidence": None,
                    "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
                    "relative_closeness": float(np.median(region))})
    return out, mask


def specular_uncertainty(image, corridor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sat = (gray > 245) & (corridor > 0)
    frac = sat.sum() / max((corridor > 0).sum(), 1)
    return float(np.clip(frac * 20, 0, 1))


def analyse(image, cache):
    image = np.asarray(image)[:, :, :3].copy()
    det_model, depth_model = _load(cache)
    dets = detect(det_model, image)
    h, w = image.shape[:2]
    if depth_model is None:
        # Detector-only fallback: fake a flat depth map so the drawing code
        # works; unknown-obstacle detection is disabled.
        depth = np.zeros((h, w), np.float32)
    else:
        depth = depth_model.estimate(image)
    dnorm = (depth - depth.min()) / (np.ptp(depth) + 1e-6)
    fused = fuse_detections_with_depth(dets, dnorm)
    corridor = create_driving_corridor_mask((h, w))
    if depth_model is None:
        # Ground-plane cue: the lower a box's bottom edge in the image,
        # the closer the object (flat floor, forward-looking camera).
        unknown = []
        for o in fused:
            o["relative_depth"] = float(o["bbox"][3]) / h
    else:
        unknown, _ = unknown_obstacles(depth, corridor, dets)
    unc = specular_uncertainty(image, corridor)

    vis = image.copy()
    overlay = vis.copy()
    contours, _ = cv2.findContours(corridor, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (60, 140, 255), -1)
    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
    cv2.drawContours(vis, contours, -1, (60, 140, 255), 2)
    for o in fused:
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 200, 60), 2)
        txt = f"{o['class_name']} {o['confidence']:.2f} " \
              f"close={o['relative_depth'] or 0:.2f}"
        cv2.putText(vis, txt, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 200, 60), 2)
    for o in unknown:
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (220, 40, 220), 2)
        cv2.putText(vis, f"UNKNOWN close={o['relative_closeness']:.2f}",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 40, 220), 2)
    depth_col = cv2.applyColorMap((dnorm * 255).astype(np.uint8),
                                  cv2.COLORMAP_INFERNO)[:, :, ::-1]
    out = np.hstack([vis, depth_col])

    # Decision preview, same priority order as the safety shield
    close_obj = [o for o in fused if (o["relative_depth"] or 0) > 0.75]
    person_close = [o for o in close_obj if o["class_name"] == "person"]
    unk_close = [o for o in unknown if o["relative_closeness"] > 0.75]
    if person_close:
        action = "STOP — person close in front"
    elif close_obj or unk_close:
        action = "STOP — obstacle close in the driving corridor"
    elif unc > 0.6:
        action = "STOP & re-check — high uncertainty (reflections)"
    elif unc > 0.3 or unknown:
        action = "SLOW — uncertain / unknown object ahead"
    else:
        action = "GO"
    md = [f"**Detector:** {cache['detector_name']} — {len(dets)} "
          f"detection(s)",
          f"**Unknown obstacles (depth only):** {len(unknown)}",
          f"**Specular-highlight uncertainty:** {unc:.2f}",
          f"**Shield-style decision:** {action}", "",
          "_Depth is relative (0 = far, 1 = near), not metres._"]
    if depth_model is None:
        md.insert(1, "⚠️ **Depth model unavailable** "
                     f"({cache.get('depth_error')}). Showing detector-only "
                     "output; connect to the internet once so Depth "
                     "Anything V2 can download from Hugging Face.")
    return out, "\n\n".join(md)
