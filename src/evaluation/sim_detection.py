"""
Score a detector on frames of the real-image camera simulation.

Ground truth comes from the renderer (which object is where, how much of
it is visible, how far away). Objects are split into

* **required**: a detector class, at least 60 % visible, at least 20 px
  tall and within the robot's operating range. Missing one is a false
  negative.
* **ignored**: heavily occluded, tiny or far objects, and forklift
  drivers. A detection on them is neither rewarded nor punished.

A detection is a true positive if it has the right class and IoU >= 0.5
with a required object; any other detection that is not on an ignored
object is a false positive (including a wrong class on a real object).
"""

from __future__ import annotations

import numpy as np

from src.perception.onnx_detector import box_iou

CLASSES = ("person", "forklift", "package")


def split_gt(gt, max_range=6.0, min_visible=0.6, min_height=20):
    req, ign = [], []
    for g in gt:
        if g["class_name"] not in CLASSES:
            continue
        x0, y0, x1, y1 = g["bbox"]
        ok = (g["visible"] >= min_visible and (y1 - y0) >= min_height
              and g["range"] <= max_range and "inner_of" not in g)
        (req if ok else ign).append(g)
    return req, ign


def match_frame(dets, gt, iou_thr=0.5, **kw):
    """dets: (N, 6) x1 y1 x2 y2 conf cls.

    Returns a list of (conf, cls, is_tp) for the detections that count,
    the list of required objects with a `hit` flag, and range errors."""
    req, ign = split_gt(gt, **kw)
    out = []
    hit = [False] * len(req)
    rb = np.array([g["bbox"] for g in req], np.float32).reshape(-1, 4)
    ib = np.array([g["bbox"] for g in ign], np.float32).reshape(-1, 4)
    # forklift boxes: a "person" inside one is the driver -> ignore
    fb = np.array([g["bbox"] for g in gt if g["class_name"] == "forklift"],
                  np.float32).reshape(-1, 4)
    order = np.argsort(-dets[:, 4]) if len(dets) else []
    for i in order:
        box, conf, cls = dets[i, :4], float(dets[i, 4]), int(dets[i, 5])
        iou_r = box_iou(box[None], rb)[0] if len(rb) else np.zeros(0)
        best, j = 0.0, -1
        for k in np.argsort(-iou_r):
            if hit[k] or CLASSES.index(req[k]["class_name"]) != cls:
                continue
            best, j = float(iou_r[k]), int(k)
            break
        if j >= 0 and best >= iou_thr:
            hit[j] = True
            out.append((conf, cls, True))
            continue
        if len(ib) and box_iou(box[None], ib)[0].max() >= 0.3:
            continue
        if cls == 0 and len(fb):
            x0 = np.maximum(box[0], fb[:, 0])
            y0 = np.maximum(box[1], fb[:, 1])
            x1 = np.minimum(box[2], fb[:, 2])
            y1 = np.minimum(box[3], fb[:, 3])
            inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
            if (inter / area).max() >= 0.7:
                continue
        out.append((conf, cls, False))
    return out, req, hit


def summarise(records, n_gt, conf=0.25):
    """records: list of (conf, cls, tp); n_gt: {cls: count}."""
    res = {}
    rec = np.array(records, dtype=float).reshape(-1, 3)
    for c, name in enumerate(CLASSES):
        r = rec[rec[:, 1] == c]
        r = r[np.argsort(-r[:, 0])]
        n = n_gt.get(c, 0)
        tp_c = np.cumsum(r[:, 2])
        fp_c = np.cumsum(1 - r[:, 2])
        recall = tp_c / max(n, 1)
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        # AP50: area under the monotone precision envelope
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[1.0], precision, [0.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
        k = int((r[:, 0] >= conf).sum())
        tp = float(tp_c[k - 1]) if k else 0.0
        fp = float(fp_c[k - 1]) if k else 0.0
        p = tp / max(tp + fp, 1e-9) if k else 1.0
        rc = tp / max(n, 1)
        res[name] = {"objects": int(n), "precision": p, "recall": rc,
                     "f1": 2 * p * rc / max(p + rc, 1e-9), "AP50": ap,
                     "tp": int(tp), "fp": int(fp), "fn": int(n - tp)}
    tp = sum(v["tp"] for v in res.values())
    fp = sum(v["fp"] for v in res.values())
    n = sum(v["objects"] for v in res.values())
    p, rc = tp / max(tp + fp, 1e-9), tp / max(n, 1)
    res["all"] = {"objects": int(n), "precision": p, "recall": rc,
                  "f1": 2 * p * rc / max(p + rc, 1e-9),
                  "mAP50": float(np.mean([res[c]["AP50"] for c in CLASSES])),
                  "tp": int(tp), "fp": int(fp), "fn": int(n - tp)}
    return res


def evaluate_frames(detect, frames, conf=0.25, low_conf=0.05, **kw):
    """`detect(rgb, conf)` -> (N, 6); `frames` yields (rgb, gt)."""
    records, n_gt = [], {}
    for rgb, gt in frames:
        out, req, _ = match_frame(detect(rgb, low_conf), gt, **kw)
        records += out
        for g in req:
            c = CLASSES.index(g["class_name"])
            n_gt[c] = n_gt.get(c, 0) + 1
    return summarise(records, n_gt, conf)


class DetectionMeter:
    """Running score of the camera perception against ground truth.

    Two outputs are scored on every frame the robot sees:

    * `raw`   - the detector's boxes at conf >= 0.25 (image only);
    * `stack` - what the full perception stack reports after depth
                fusion, gating and tracking.

    Plus the distance error of correct detections and how many unknown
    objects (no detector class) the depth branch flagged.
    """

    def __init__(self, conf=0.25, max_range=6.0):
        self.conf, self.max_range = conf, max_range
        self.reset()

    def reset(self):
        self.frames = 0
        self.rec = {"raw": [], "stack": []}
        self.n_gt = {}
        self.range_err = []
        self.unknown = [0, 0]              # found, total
        self.ms = []

    def update(self, frame, res):
        self.frames += 1
        gt = frame.gt
        if res.raw is not None:
            raw = res.raw[res.raw[:, 4] >= self.conf]
            out, req, _ = match_frame(raw, gt, max_range=self.max_range)
            self.rec["raw"] += out
            for g in req:
                c = CLASSES.index(g["class_name"])
                self.n_gt[c] = self.n_gt.get(c, 0) + 1
        objs = res.objects
        dets = np.array([[*o["box"], o["conf"], o["cls"]] for o in objs],
                        np.float32).reshape(-1, 6)
        out, req, hit = match_frame(dets, gt, max_range=self.max_range)
        self.rec["stack"] += [(1.0, c, tp) for _, c, tp in out]
        # distance error of the objects that were found
        rb = np.array([g["bbox"] for g in req], np.float32).reshape(-1, 4)
        for o in objs:
            if not len(rb):
                break
            iou = box_iou(np.array(o["box"], np.float32)[None], rb)[0]
            j = int(np.argmax(iou))
            if iou[j] >= 0.5 and CLASSES.index(
                    req[j]["class_name"]) == o["cls"]:
                self.range_err.append(o["range"] - req[j]["range"])
        # unknown objects: must show up as a depth-only blob (or, wrongly
        # but safely, as a tracked object)
        boxes = [u["box"] for u in res.unknown] + [o["box"] for o in objs]
        bb = np.array(boxes, np.float32).reshape(-1, 4)
        for g in gt:
            if g["class_name"] != "unknown" or g["visible"] < 0.6 or \
                    g["range"] > min(self.max_range, 5.5) or \
                    g["bbox"][3] - g["bbox"][1] < 12:
                continue
            self.unknown[1] += 1
            if len(bb) and box_iou(np.array(g["bbox"], np.float32)[None],
                                   bb)[0].max() >= 0.2:
                self.unknown[0] += 1
        self.ms.append(sum(res.ms.values()))

    def summary(self):
        out = {"frames": self.frames}
        for key in ("raw", "stack"):
            out[key] = summarise(self.rec[key], self.n_gt, self.conf
                                 if key == "raw" else 0.5)
        e = np.array(self.range_err)
        out["range_mae_m"] = float(np.abs(e).mean()) if len(e) else None
        out["range_bias_m"] = float(e.mean()) if len(e) else None
        out["unknown_recall"] = (self.unknown[0] / self.unknown[1]
                                 if self.unknown[1] else None)
        out["unknown_objects"] = self.unknown[1]
        out["ms_per_frame"] = float(np.mean(self.ms)) if self.ms else None
        return out

    def short(self):
        """Precision / recall of the stack so far (for the HUD)."""
        rec = np.array(self.rec["stack"], dtype=float).reshape(-1, 3)
        tp = float(rec[:, 2].sum())
        n = sum(self.n_gt.values())
        return (tp / max(len(rec), 1) if len(rec) else 1.0,
                tp / max(n, 1) if n else 1.0, int(n))
