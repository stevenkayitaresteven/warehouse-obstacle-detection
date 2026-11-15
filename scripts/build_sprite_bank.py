"""
Cut real objects out of the detector's training photos and store them as
transparent sprites for the real-image camera simulation.

    person    <- Ultralytics construction-ppe ("Person" boxes)
    forklift  <- Ultralytics construction-ppe ("forklift" boxes)
    package   <- Ultralytics package-seg     (segmentation polygons)
    unknown   <- construction-ppe, objects that have NO detector class
                 (traffic cones, tool cases, pallets ...; hand-picked boxes
                 listed in assets/sim/sprites/unknown_objects.json)

Every sprite keeps the split of the photo it was cut from (train / val /
test of the 3-class dataset built by scripts/build_warehouse_dataset.py).
That is what makes the simulation evaluation honest: composite frames used
for fine-tuning only contain *train* sprites, and the demo and the in-loop
evaluation only use *val / test* sprites the detector has never seen.

Boxes are turned into masks with SAM 2.1 (box prompt); package polygons are
used directly. Selection is automatic (size, not truncated, not occluded,
feet and head visible for people) and then filtered by the hand-checked
reject list in assets/sim/sprites/curation.json.

    python -m scripts.build_warehouse_dataset --src data_downloads --download \
        --out datasets/processed/warehouse3
    python scripts/build_sprite_bank.py --src data_downloads \
        --dataset datasets/processed/warehouse3 --out assets/sim/sprites

Needs: ultralytics (SAM), opencv, Pillow. Only needed to REBUILD the bank;
the demo just reads the committed sprites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
PERSON, FORKLIFT = 5, 10          # class ids in construction-ppe
BOOTS, HELMET = {3, 9}, {0, 6}    # (no_)boots, (no_)helmet
MAX_H = {"person": 360, "forklift": 380, "package": 230, "unknown": 240}
# how many sprites to keep per (class, split); the best-scored ones win
QUOTA = {
    "person": {"train": 90, "val": 30, "test": 30},
    "forklift": {"train": 40, "val": 12, "test": 12},
    "package": {"train": 60, "val": 24, "test": 24},
}


def read_labels(path, w, h):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            v = [float(x) for x in p[1:]]
            if len(v) == 4:
                cx, cy, bw, bh = v
                box = [(cx - bw / 2) * w, (cy - bh / 2) * h,
                       (cx + bw / 2) * w, (cy + bh / 2) * h]
                poly = None
            else:
                poly = np.array(v, np.float32).reshape(-1, 2) * (w, h)
                box = [poly[:, 0].min(), poly[:, 1].min(),
                       poly[:, 0].max(), poly[:, 1].max()]
            rows.append((c, box, poly))
    return rows


def inter_frac(a, b):
    """Fraction of box a covered by box b."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def box_iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1])
                + (b[2] - b[0]) * (b[3] - b[1]) - i)


def truncated(b, w, h, m=4):
    return b[0] < m or b[1] < m or b[2] > w - m or b[3] > h - m


def centre_in(box, region):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


# ----------------------------------------------------------------------
def person_candidates(rows, w, h):
    persons = [b for c, b, _ in rows if c == PERSON]
    boots = [b for c, b, _ in rows if c in BOOTS]
    helmets = [b for c, b, _ in rows if c in HELMET]
    out = []
    for b in persons:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bh < 300 or bw < 90 or truncated(b, w, h):
            continue
        aspect = bh / bw
        if not 1.25 <= aspect <= 4.6:
            continue
        if any(o is not b and inter_frac(b, o) > 0.12 for o in persons):
            continue
        feet = [b[0], b[1] + 0.72 * bh, b[2], b[3] + 0.03 * bh]
        head = [b[0], b[1] - 0.03 * bh, b[2], b[1] + 0.34 * bh]
        if not any(centre_in(o, feet) for o in boots):
            continue
        if not any(centre_in(o, head) for o in helmets):
            continue
        # prefer large, upright people
        score = min(bh, 900) / 900 + 0.3 * min(aspect, 3.0) / 3.0
        out.append((b, score))
    return out


def forklift_candidates(rows, w, h):
    persons = [b for c, b, _ in rows if c == PERSON]
    forks = [b for c, b, _ in rows if c == FORKLIFT]
    out = []
    for b in forks:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bh < 200 or bw < 130 or truncated(b, w, h):
            continue
        if not 0.45 <= bh / bw <= 2.6:
            continue
        if any(inter_frac(b, o) > 0.06 for o in persons):
            continue
        if any(o is not b and inter_frac(b, o) > 0.10 for o in forks):
            continue
        out.append((b, min(bh, 900) / 900))
    return out


def package_candidates(rows, w, h, img_bgr):
    out = []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for c, b, poly in rows:
        if c != 0 or poly is None or len(poly) < 4:
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        if min(bw, bh) < 70 or truncated(b, w, h, 3):
            continue
        if not 0.45 <= bh / bw <= 2.2:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        area = int((mask > 0).sum())
        if area / (bw * bh) < 0.55:
            continue
        px = hsv[mask > 0]
        hue, sat, val = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
        # keep natural parcel colours (cardboard / white / grey plastic);
        # the train split of package-seg is heavily colour-augmented
        natural = (sat < 70 and 60 < val < 235) or \
                  (8 <= hue <= 30 and sat < 150 and val > 70)
        # the conveyor's red scanner line lights many parcels: skip those
        laser = (((px[:, 0] < 7) | (px[:, 0] > 172)) & (px[:, 1] > 140)
                 & (px[:, 2] > 120)).mean()
        if not natural or laser > 0.05:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_32F).var()
        if sharp < 60:                   # motion-blurred parcel
            continue
        out.append((b, min(area, 40000) / 40000 + min(sharp, 600) / 1200,
                    mask))
    return out


# ----------------------------------------------------------------------
def clean_mask(mask):
    """Largest blob (+ attached parts), holes filled, soft 1-px edge."""
    m = (mask > 0).astype(np.uint8)
    k = max(3, int(round(0.004 * max(m.shape))) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    m = np.isin(lab, [i for i in range(1, n)
                      if stats[i, cv2.CC_STAT_AREA] >= 0.03 * areas.max()
                      or i == keep]).astype(np.uint8)
    # fill interior holes
    ff = m.copy()
    pad = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(pad, None, (0, 0), 2)
    holes = (pad[1:-1, 1:-1] == 0)
    m[holes] = 1
    return m


def cut(img_bgr, mask, max_h):
    ys, xs = np.where(mask > 0)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = img_bgr[y1:y2, x1:x2]
    m = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    scale = min(1.0, max_h / crop.shape[0])
    # pull the edge in by ~1 output pixel so no background halo survives
    e = max(1, int(round(0.8 / scale)))
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * e + 1, 2 * e + 1)))
    if scale < 1.0:
        size = (max(2, int(round(crop.shape[1] * scale))),
                max(2, int(round(crop.shape[0] * scale))))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (3, 3), 0.6)
    rgba = np.dstack([crop[:, :, ::-1], m])
    return rgba, [int(x1), int(y1), int(x2), int(y2)]


def save_sprite(rgba, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=92, method=6,
                                       exact=False)


def contact_sheet(items, path, cell=150, cols=12):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 70, np.uint8)
    sheet[::16, :, :] = 80
    for k, (sid, rgba) in enumerate(items):
        h, w = rgba.shape[:2]
        s = min((cell - 16) / h, (cell - 4) / w)
        im = cv2.resize(rgba, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA)
        y0 = (k // cols) * cell + 14
        x0 = (k % cols) * cell + (cell - im.shape[1]) // 2
        a = im[:, :, 3:4].astype(np.float32) / 255
        roi = sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]]
        roi[:] = (im[:, :, 2::-1] * a + roi * (1 - a)).astype(np.uint8)
        cv2.putText(sheet, str(k), ((k % cols) * cell + 3,
                                    (k // cols) * cell + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------------
def split_of(dataset, prefix, stem):
    for s in ("train", "val", "test"):
        for ext in (".jpg", ".jpeg", ".png"):
            if (dataset / "images" / s / f"{prefix}_{stem}{ext}").exists():
                return s
    return None


def find_image(folder, name):
    for sub in ("train", "val", "test"):
        p = folder / "images" / sub / name
        if p.exists():
            return p, folder / "labels" / sub / (Path(name).stem + ".txt")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder with construction-ppe/ and package-seg/")
    ap.add_argument("--dataset", default="datasets/processed/warehouse3")
    ap.add_argument("--out", default="assets/sim/sprites")
    ap.add_argument("--sam", default="sam2.1_b.pt")
    ap.add_argument("--sheets", default=None,
                    help="folder for QA contact sheets (not committed)")
    ap.add_argument("--only", default="person,forklift,package,unknown")
    a = ap.parse_args()
    src, dataset, out = Path(a.src), Path(a.dataset), Path(a.out)
    only = set(a.only.split(","))
    out.mkdir(parents=True, exist_ok=True)
    cur_path = out / "curation.json"
    curation = json.loads(cur_path.read_text()) if cur_path.exists() else {}
    reject = set(curation.get("reject", []))
    meta_fix = curation.get("meta", {})

    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() \
        else {"sprites": []}
    index["sprites"] = [s for s in index["sprites"]
                        if s["class"] not in only]
    sheets = {}

    sam = None
    if only & {"person", "forklift", "unknown"}:
        from ultralytics import SAM
        sam = SAM(a.sam)

    def sam_masks(img, boxes):
        res = sam(img, bboxes=[[float(v) for v in b] for b in boxes],
                  verbose=False)[0]
        return res.masks.data.cpu().numpy().astype(np.uint8)

    def add(cls, split, stem, k, rgba, src_box, extra=None):
        sid = f"{cls}-{split}-{stem[-9:]}-{k}"
        if sid in reject:
            return
        rel = f"{cls}/{sid}.webp"
        save_sprite(rgba, out / rel)
        rec = {"id": sid, "class": cls, "split": split, "file": rel,
               "w": int(rgba.shape[1]), "h": int(rgba.shape[0]),
               "source": stem, "source_box": src_box}
        rec.update(extra or {})
        rec.update(meta_fix.get(sid, {}))
        index["sprites"].append(rec)
        sheets.setdefault((cls, split), []).append((sid, rgba))

    # ---- people and forklifts (construction-ppe) ----------------------
    ppe = src / "construction-ppe" / "construction-ppe"
    if only & {"person", "forklift"}:
        cands = {"person": {}, "forklift": {}}
        files = [p for sub in ("train", "val")
                 for p in sorted((ppe / "images" / sub).iterdir())
                 if p.suffix.lower() in IMG_EXT]
        for p in files:
            split = split_of(dataset, "ppe", p.stem)
            if split is None:
                continue
            with Image.open(p) as im:
                w, h = im.size
            rows = read_labels(ppe / "labels" / p.parent.name /
                               f"{p.stem}.txt", w, h)
            if "person" in only:
                for b, sc in person_candidates(rows, w, h):
                    cands["person"].setdefault(split, []).append((sc, p, b))
            if "forklift" in only:
                for b, sc in forklift_candidates(rows, w, h):
                    cands["forklift"].setdefault(split, []).append((sc, p, b))
        for cls in ("person", "forklift"):
            if cls not in only:
                continue
            for split, lst in cands[cls].items():
                lst.sort(key=lambda t: -t[0])
                # 1.6x the quota: some masks fail the quality checks below
                lst = lst[:int(QUOTA[cls][split] * 1.6)]
                by_img = {}
                for sc, p, b in lst:
                    by_img.setdefault(p, []).append((sc, b))
                kept = 0
                for p, items in by_img.items():
                    if kept >= QUOTA[cls][split]:
                        break
                    img = cv2.imread(str(p))
                    masks = sam_masks(img, [b for _, b in items])
                    for k, ((sc, b), mk) in enumerate(zip(items, masks)):
                        m = clean_mask(mk)
                        if m is None:
                            continue
                        ys, xs = np.where(m > 0)
                        mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                        fill = m.sum() / max((b[2] - b[0]) * (b[3] - b[1]), 1)
                        lo, hi = (0.25, 0.85) if cls == "person" \
                            else (0.28, 0.92)
                        if box_iou(mb, b) < 0.78 or not lo <= fill <= hi:
                            continue
                        rgba, sb = cut(img, m, MAX_H[cls])
                        add(cls, split, p.stem, k, rgba, sb,
                            {"fill": round(float(fill), 3)})
                        kept += 1
                print(f"{cls:9s} {split:5s}: {kept} sprites "
                      f"(from {len(lst)} candidates)", flush=True)

    # ---- packages (package-seg polygons) ------------------------------
    if "package" in only:
        pkg = src / "package-seg"
        for split in ("train", "val", "test"):
            # the processed dataset copies each source split unchanged
            names = sorted(p.name[4:] for p in
                           (dataset / "images" / split).glob("pkg_*"))
            cand = []
            for name in names:
                p = pkg / "images" / split / name
                if not p.exists():
                    continue
                img = cv2.imread(str(p))
                h, w = img.shape[:2]
                rows = read_labels(pkg / "labels" / split /
                                   f"{p.stem}.txt", w, h)
                for b, sc, mask in package_candidates(rows, w, h, img):
                    cand.append((sc, p, b, mask))
            cand.sort(key=lambda t: -t[0])
            kept, per_img = 0, {}
            for sc, p, b, mask in cand:
                if kept >= QUOTA["package"][split]:
                    break
                if per_img.get(p, 0) >= 2:      # variety: max 2 per photo
                    continue
                per_img[p] = per_img.get(p, 0) + 1
                img = cv2.imread(str(p))
                rgba, sb = cut(img, (mask > 0).astype(np.uint8),
                               MAX_H["package"])
                add("package", split, p.stem, per_img[p] - 1, rgba, sb)
                kept += 1
            print(f"package   {split:5s}: {kept} sprites "
                  f"(from {len(cand)} candidates)", flush=True)

    # ---- unknown objects (hand-picked boxes, no detector class) -------
    unk_path = out / "unknown_objects.json"
    if "unknown" in only and unk_path.exists():
        for k, rec in enumerate(json.loads(unk_path.read_text())["objects"]):
            p, _ = find_image(ppe, rec["image"])
            if p is None:
                print("missing", rec["image"])
                continue
            split = split_of(dataset, "ppe", p.stem) or "extra"
            img = cv2.imread(str(p))
            m = clean_mask(sam_masks(img, [rec["box"]])[0])
            if m is None:
                print("no mask for", rec)
                continue
            rgba, sb = cut(img, m, MAX_H["unknown"])
            add("unknown", split, p.stem, k, rgba, sb,
                {"name": rec["name"], "height_m": rec["height_m"]})
        n = sum(s["class"] == "unknown" for s in index["sprites"])
        print(f"unknown        : {n} sprites", flush=True)

    index["sprites"].sort(key=lambda s: s["id"])
    index["note"] = ("Real objects cut out of the detector's training "
                     "datasets. See assets/sim/ATTRIBUTION.md.")
    index_path.write_text(json.dumps(index, indent=1))
    if a.sheets:
        sd = Path(a.sheets)
        sd.mkdir(parents=True, exist_ok=True)
        for (cls, split), items in sheets.items():
            contact_sheet(items, sd / f"{cls}_{split}.jpg")
            (sd / f"{cls}_{split}.txt").write_text(
                "\n".join(f"{k} {sid}" for k, (sid, _) in enumerate(items)))
    counts = {}
    for s in index["sprites"]:
        counts[(s["class"], s["split"])] = counts.get(
            (s["class"], s["split"]), 0) + 1
    print("bank:", {f"{c}/{s}": n for (c, s), n in sorted(counts.items())})


if __name__ == "__main__":
    main()
