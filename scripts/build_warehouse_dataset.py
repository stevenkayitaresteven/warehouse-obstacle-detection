"""
Build the 3-class "warehouse obstacle" detection dataset used to fine-tune
YOLO11n:

    0 person    <- Ultralytics construction-ppe  ("Person", workers in PPE)
    1 forklift  <- Ultralytics construction-ppe  ("forklift")
    2 package   <- Ultralytics package-seg       (parcels/boxes; polygons
                                                   converted to boxes)

Both source datasets are public and downloaded from the Ultralytics GitHub
release assets. Splits:
    package-seg      : its own train / val / test
    construction-ppe : train -> train, val -> split 50/50 into val / test

    python -m scripts.build_warehouse_dataset --src /path/to/downloads \
        --out datasets/processed/warehouse3 --max-package-train 900

(--src must contain the extracted `package-seg/` and `construction-ppe/`
folders; use --download to fetch them.)
"""

from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "package-seg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip",
    "construction-ppe": "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
}
NAMES = ["person", "forklift", "package"]
PPE_MAP = {5: 0, 10: 1}     # Person -> person, forklift -> forklift
IMG_EXT = {".jpg", ".jpeg", ".png"}


def download(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if (src / name).exists():
            continue
        z = src / f"{name}.zip"
        print("downloading", url)
        urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(src / name if name == "construction-ppe" else src)
        z.unlink()


def poly_to_box(vals):
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def convert(img: Path, lbl: Path, out: Path, split: str, prefix: str, kind: str):
    lines = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if not p:
                continue
            c = int(float(p[0]))
            vals = [float(v) for v in p[1:]]
            if kind == "package":
                if c != 0 or len(vals) < 6:
                    continue
                cx, cy, w, h = poly_to_box(vals) if len(vals) > 4 else vals
                cls = 2
            else:
                if c not in PPE_MAP or len(vals) < 4:
                    continue
                cx, cy, w, h = (vals[:4] if len(vals) == 4 else poly_to_box(vals))
                cls = PPE_MAP[c]
            if w <= 0.002 or h <= 0.002:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    if kind == "ppe" and not lines:
        return False          # keep only PPE images that contain our classes
    name = f"{prefix}_{img.stem}"
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out / "images" / split / f"{name}{img.suffix.lower()}")
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    return True


def images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="datasets/processed/warehouse3")
    ap.add_argument("--max-package-train", type=int, default=900)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    rng = random.Random(a.seed)
    if a.download:
        download(src)
    if out.exists():
        shutil.rmtree(out)

    counts = {}
    pkg = src / "package-seg"
    for split in ("train", "val", "test"):
        imgs = images(pkg / "images" / split)
        if split == "train" and len(imgs) > a.max_package_train:
            imgs = rng.sample(imgs, a.max_package_train)
        n = sum(convert(i, pkg / "labels" / split / f"{i.stem}.txt", out, split,
                        "pkg", "package") for i in imgs)
        counts[f"package/{split}"] = n

    ppe = src / "construction-ppe" / "construction-ppe"
    tr = images(ppe / "images" / "train")
    va = images(ppe / "images" / "val")
    rng.shuffle(va)
    half = len(va) // 2
    for split, imgs in (("train", tr), ("val", va[:half]), ("test", va[half:])):
        n = sum(convert(i, ppe / "labels" / ("train" if split == "train" else "val")
                        / f"{i.stem}.txt", out, split, "ppe", "ppe") for i in imgs)
        counts[f"ppe/{split}"] = n

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    inst = {n: 0 for n in NAMES}
    for f in (out / "labels").rglob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                inst[NAMES[int(line.split()[0])]] += 1
    print("images:", counts)
    print("instances:", inst)
    print("wrote", yaml)


if __name__ == "__main__":
    main()
