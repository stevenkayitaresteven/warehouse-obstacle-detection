"""
Evaluate the fine-tuned YOLO11n warehouse detector on the held-out TEST split
and compare with the COCO-pretrained YOLO11n (before fine-tuning).

* Fine-tuned: standard Ultralytics val on the test split (mAP50, mAP50-95,
  per-class AP, confusion matrix, PR / F1 curves).
* Before fine-tuning: the COCO model has a "person" class but no forklift or
  package class. Its person predictions are scored against our person
  labels with the same AP code; forklift and package count as 0.
* Speed: CPU inference FPS.

    python -m scripts.eval_yolo --weights results/yolo_warehouse/yolo11n_warehouse3/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "processed" / "warehouse3"
OUT = ROOT / "results" / "yolo_eval"


def load_gt(lbl_dir, img_path, w, h):
    f = lbl_dir / f"{img_path.stem}.txt"
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            p = line.split()
            if p:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                            (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def person_ap(model, pred_cls, imgs, lbl_dir, thr=(0.5,)):
    """AP for the person class (our id 0) with a model's class `pred_cls`."""
    from ultralytics.utils.metrics import compute_ap
    scores, tps, n_gt = [], {t: [] for t in thr}, 0
    for img in imgs:
        r = model.predict(str(img), conf=0.001, iou=0.7, imgsz=416,
                          verbose=False, classes=[pred_cls])[0]
        h, w = r.orig_shape
        gt = np.array([g[1:] for g in load_gt(lbl_dir, img, w, h) if g[0] == 0])
        n_gt += len(gt)
        if r.boxes is None or len(r.boxes) == 0:
            continue
        pb = r.boxes.xyxy.cpu().numpy()
        ps = r.boxes.conf.cpu().numpy()
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        scores += ps.tolist()
        M = iou(pb, gt) if len(gt) else np.zeros((len(pb), 0))
        for t in thr:
            used = set()
            for i in range(len(pb)):
                j = int(np.argmax(M[i])) if M.shape[1] else -1
                ok = j >= 0 and M[i, j] >= t and j not in used
                if ok:
                    used.add(j)
                tps[t].append(1 if ok else 0)
    order = np.argsort(-np.array(scores))
    res = {}
    for t in thr:
        tp = np.array(tps[t])[order]
        ctp = np.cumsum(tp)
        recall = ctp / max(n_gt, 1)
        precision = ctp / np.arange(1, len(tp) + 1)
        res[t] = float(compute_ap(recall, precision)[0]) if len(tp) else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pretrained", default=str(ROOT / "weights" / "yolo" / "yolo11n.pt"))
    a = ap.parse_args()
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    data_yaml = DATA / "data.yaml"

    ft = YOLO(a.weights)
    m = ft.val(data=str(data_yaml), split="test", imgsz=416, batch=16,
               device="cpu", plots=True, project=str(OUT), name="test",
               exist_ok=True, verbose=False)
    names = ft.names
    per_class = {names[i]: {"AP50": float(m.box.ap50[k]), "AP50_95": float(m.box.ap[k]),
                            "precision": float(m.box.p[k]), "recall": float(m.box.r[k])}
                 for k, i in enumerate(m.box.ap_class_index)}
    # Confusion matrix at the deployment threshold (conf 0.25), so the
    # false-alarm counts are not inflated by the 0.001 threshold used for mAP.
    m25 = ft.val(data=str(data_yaml), split="test", imgsz=416, batch=16,
                 device="cpu", conf=0.25, plots=True, project=str(OUT),
                 name="test_conf025", exist_ok=True, verbose=False)
    cm = m25.confusion_matrix.matrix.astype(int).tolist()

    imgs = sorted((DATA / "images" / "test").iterdir())
    lbl = DATA / "labels" / "test"
    pre = YOLO(a.pretrained)
    base_person = person_ap(pre, 0, imgs, lbl)[0.5]
    ft_person = person_ap(ft, 0, imgs, lbl)[0.5]

    # speed (CPU, single image, after warm-up)
    sample = [str(p) for p in imgs[:40]]
    for p in sample[:5]:
        ft.predict(p, imgsz=416, verbose=False)
    t0 = time.perf_counter()
    for p in sample:
        ft.predict(p, imgsz=416, verbose=False)
    fps = len(sample) / (time.perf_counter() - t0)

    rep = {
        "note": "Held-out test split of the 3-class warehouse dataset "
                "(package-seg + construction-ppe). CPU evaluation.",
        "test_images": len(imgs),
        "fine_tuned": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                       "precision": float(m.box.mp), "recall": float(m.box.mr),
                       "per_class": per_class},
        "confusion_matrix": {"conf_threshold": 0.25, "labels": [names[i] for i in range(len(names))] + ["background"],
                             "matrix_pred_rows_true_cols": cm},
        "before_vs_after_AP50": {
            "person": {"coco_pretrained": base_person, "fine_tuned": ft_person},
            "forklift": {"coco_pretrained": 0.0, "fine_tuned": per_class.get("forklift", {}).get("AP50")},
            "package": {"coco_pretrained": 0.0, "fine_tuned": per_class.get("package", {}).get("AP50")},
        },
        "cpu_fps_imgsz416": fps,
    }
    (ROOT / "results" / "metrics" / "yolo_eval.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
