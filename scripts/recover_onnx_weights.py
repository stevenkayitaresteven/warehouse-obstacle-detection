"""
Turn an exported YOLO ONNX file back into a trainable Ultralytics checkpoint.

Why: the robot-view fine-tuning of the detector stopped after epoch 2 and
only its ONNX export (weights/yolo/best.onnx) survived; the PyTorch
checkpoint was lost. ONNX export fuses every Conv + BatchNorm pair into a
single convolution with a bias, so the weights cannot be loaded directly.

How:
1. Build the same architecture (YOLO11n, 3 classes) from a reference
   checkpoint and fuse a copy of it, exactly like the exporter did.
2. Copy the ONNX convolution weights into the fused copy, matching the
   convolutions in execution order (shapes are checked one by one).
3. Check that the fused PyTorch model reproduces the ONNX outputs.
4. Un-fuse: each Conv gets the fused weight, and its BatchNorm is set so
   that it reproduces the fused bias *and* has realistic running
   statistics. The statistics are measured on real training images:
   with per-channel mean mu and variance var of the conv output,
       gamma = sqrt(var + eps), beta = mu + b_fused,
       running_mean = mu, running_var = var,
   so BN(y) = y + b_fused in eval mode, and approximately so in train
   mode (batch statistics ~ population statistics). Training can then
   continue from where the fine-tuning stopped.
5. Save an Ultralytics checkpoint and verify it once more against ONNX.

    python scripts/recover_onnx_weights.py --onnx weights/yolo/best.onnx \
        --ref weights/yolo/best.pt --images datasets/processed/simcam/images/train \
        datasets/processed/warehouse3/images/train \
        --out weights/yolo/recovered_epoch2.pt

Needs ultralytics + torch + onnx + onnxruntime.
"""

from __future__ import annotations

import argparse
import copy
import glob
import random

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


def letterbox(img, size=(384, 640)):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size[0], size[1], 3), 114, np.uint8)
    top, left = (size[0] - nh) // 2, (size[1] - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out


def load_batch(paths, size):
    ims = [letterbox(cv2.imread(p)[:, :, ::-1], size) for p in paths]
    x = np.stack(ims).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def onnx_convs(path):
    """(node name, weight, bias) of every Conv node."""
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = []
    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        w = init.get(node.input[1])
        b = init.get(node.input[2]) if len(node.input) > 2 else None
        if w is None:
            raise RuntimeError(f"conv {node.name}: weight is not an initializer")
        out.append((node.name, w, b))
    return out


def module_path(onnx_name):
    """ONNX node name -> PyTorch module path.

    The exporter names nodes after the module path, and nested blocks
    repeat their prefix: '/model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv'
    is module 'model.23.cv3.2.0.0.conv'."""
    parts = onnx_name.strip("/").split("/")[:-1]          # drop op name
    out = []
    for p in parts:
        if out and p.startswith(out[-1] + "."):
            out[-1] = p                                   # nested prefix
        else:
            out.append(p)
    return ".".join(out)


def run_onnx(path, x):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    # the export has a fixed batch size of 1: run the images one by one
    return np.concatenate([s.run(None, {name: x[i:i + 1]})[0]
                           for i in range(len(x))])


def torch_out(model, x):
    with torch.no_grad():
        y = model(torch.from_numpy(x))
    return (y[0] if isinstance(y, (list, tuple)) else y).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="weights/yolo/best.onnx")
    ap.add_argument("--ref", default="weights/yolo/best.pt")
    ap.add_argument("--images", required=True, nargs="+",
                    help="folder(s) of training images for the BN statistics")
    ap.add_argument("--n-stats", type=int, default=256)
    ap.add_argument("--out", default="weights/yolo/recovered_epoch2.pt")
    a = ap.parse_args()
    from ultralytics import YOLO

    meta = onnx.load(a.onnx).metadata_props
    meta = {p.key: p.value for p in meta}
    size = tuple(int(v) for v in meta.get("imgsz", "[384, 640]")
                 .strip("[]").split(","))
    print("ONNX:", meta.get("description"), meta.get("date"), "imgsz", size)

    ref = YOLO(a.ref)
    base = ref.model.float().eval()
    fused = copy.deepcopy(base).fuse(verbose=False).eval()

    paths = sorted(p for d in a.images for p in glob.glob(f"{d}/*.jpg"))
    random.Random(0).shuffle(paths)
    x_chk = load_batch(paths[:4], size)

    # --- 2. copy the ONNX weights into the fused model -----------------
    src = onnx_convs(a.onnx)
    convs = {n: m for n, m in fused.named_modules()
             if isinstance(m, torch.nn.Conv2d)}
    if len(src) != len(convs):
        raise SystemExit(f"conv count differs: onnx {len(src)} vs "
                         f"torch {len(convs)}")
    with torch.no_grad():
        for on, w, b in src:
            tn = module_path(on)
            if tn not in convs:
                raise SystemExit(f"onnx conv {on} -> {tn}: no such module")
            m = convs.pop(tn)
            if tuple(w.shape) != tuple(m.weight.shape):
                raise SystemExit(f"shape mismatch {on} {w.shape} vs {tn} "
                                 f"{tuple(m.weight.shape)}")
            m.weight.copy_(torch.from_numpy(w.copy()))
            if b is not None:
                if m.bias is None:
                    raise SystemExit(f"{tn} has no bias but onnx {on} does")
                m.bias.copy_(torch.from_numpy(b.copy()))
    if convs:
        raise SystemExit(f"torch convs without onnx weights: {list(convs)}")
    print(f"copied {len(src)} convolutions (matched by module name)")

    # --- 3. fused PyTorch == ONNX? ---------------------------------------
    ref_out = run_onnx(a.onnx, x_chk)
    diff = np.abs(torch_out(fused, x_chk) - ref_out)
    print(f"fused torch vs onnx: max |diff| {diff.max():.2e} "
          f"(output range {np.abs(ref_out).max():.1f})")
    if diff.max() > 1e-2:
        raise SystemExit("fused model does not reproduce the ONNX outputs")

    # --- 4. un-fuse with BN statistics measured on training images ------
    fused_mods = dict(fused.named_modules())
    stats = {}
    hooks = []
    for name, mod in base.named_modules():
        if hasattr(mod, "conv") and hasattr(mod, "bn") and \
                isinstance(mod.bn, torch.nn.BatchNorm2d):
            fconv = fused_mods[name + ".conv"]
            stats[name] = [0.0, 0.0, 0]

            def hook(m, i, o, n=name):
                y = o - m.bias.view(1, -1, 1, 1)      # conv output w/o bias
                s = stats[n]
                s[0] = s[0] + y.sum(dim=(0, 2, 3)).double()
                s[1] = s[1] + (y * y).sum(dim=(0, 2, 3)).double()
                s[2] += y.shape[0] * y.shape[2] * y.shape[3]
            hooks.append(fconv.register_forward_hook(hook))
    with torch.no_grad():
        for k in range(0, min(a.n_stats, len(paths)), 16):
            fused(torch.from_numpy(load_batch(paths[k:k + 16], size)))
    for h in hooks:
        h.remove()

    model = copy.deepcopy(base)
    mods = dict(model.named_modules())
    n_bn = n_plain = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if name in stats:
                fconv = fused_mods[name + ".conv"]
                s1, s2, n = stats[name]
                mu = (s1 / n).float()
                var = (s2 / n).float() - mu * mu
                var = torch.clamp(var, min=1e-6)
                mod.conv.weight.copy_(fconv.weight)
                bn = mod.bn
                bn.running_mean.copy_(mu)
                bn.running_var.copy_(var)
                bn.weight.copy_(torch.sqrt(var + bn.eps))
                bn.bias.copy_(mu + fconv.bias)
                n_bn += 1
        # convolutions without BN (detection head outputs, DFL)
        for name, fm in fused.named_modules():
            if not isinstance(fm, torch.nn.Conv2d):
                continue
            parent = name.rsplit(".", 1)[0]
            if name.endswith(".conv") and parent in stats:
                continue
            tm = mods[name]
            tm.weight.copy_(fm.weight)
            if fm.bias is not None:
                tm.bias.copy_(fm.bias)
            n_plain += 1
    print(f"un-fused {n_bn} Conv+BN pairs, copied {n_plain} plain convs")
    model.eval()
    diff = np.abs(torch_out(model, x_chk) - ref_out)
    print(f"un-fused (eval) vs onnx: max |diff| {diff.max():.2e}")

    # --- 5. save as an Ultralytics checkpoint ---------------------------
    ckpt = torch.load(a.ref, map_location="cpu", weights_only=False)
    for k in ("ema", "optimizer", "updates", "train_metrics",
              "train_results", "best_fitness"):
        ckpt[k] = None
    ckpt["epoch"] = -1
    ckpt["model"] = model.float()
    ckpt["train_args"] = dict(ckpt.get("train_args") or {},
                              name="recovered_from_onnx")
    ckpt["recovered_from"] = {"onnx": a.onnx, "onnx_meta": meta}
    torch.save(ckpt, a.out)
    again = YOLO(a.out).model.float().eval()
    diff = np.abs(torch_out(again, x_chk) - ref_out)
    print(f"saved {a.out}; reloaded vs onnx: max |diff| {diff.max():.2e}")


if __name__ == "__main__":
    main()
