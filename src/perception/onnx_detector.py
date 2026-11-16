"""
Fine-tuned YOLO11n warehouse detector, run from an ONNX file.

The simulation's real-image camera calls the detector on every frame, so
it has to be light: no PyTorch, just ``onnxruntime`` (CPU). If
onnxruntime is missing the same file is run with OpenCV's DNN module.

    det = OnnxDetector()                      # weights/yolo/best.onnx
    boxes = det(rgb)                          # (N, 6): x1 y1 x2 y2 conf cls

Pre- and post-processing are written out in NumPy (letterbox, decode,
non-maximum suppression) so the whole detector is ~150 lines you can read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "weights" / "yolo" / "best.onnx"
CLASS_NAMES = ("person", "forklift", "package")


def letterbox(img, size, pad_value=114):
    """Resize keeping the aspect ratio and pad to `size` = (h, w)."""
    h, w = img.shape[:2]
    th, tw = size
    if (h, w) == (th, tw):
        return img, 1.0, 0, 0
    r = min(th / h, tw / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR
                         if r > 1 else cv2.INTER_AREA)
    top, left = (th - nh) // 2, (tw - nw) // 2
    out = np.full((th, tw, 3), pad_value, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def box_iou(a, b):
    """IoU matrix of boxes a (N, 4) and b (M, 4) in xyxy."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def nms(boxes, scores, iou_thr):
    """Greedy non-maximum suppression. Returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        iou = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_thr]
    return keep


class OnnxDetector:
    """YOLO11n (person / forklift / package) from an ONNX export."""

    def __init__(self, path=DEFAULT_ONNX, conf=0.25, iou=0.55,
                 agnostic_iou=0.80, threads=None, backend=None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.conf, self.iou, self.agnostic_iou = conf, iou, agnostic_iou
        self.names = CLASS_NAMES
        self.backend = backend
        self.session = self.net = None
        if backend in (None, "onnxruntime"):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                if threads:
                    opts.intra_op_num_threads = int(threads)
                opts.log_severity_level = 3
                self.session = ort.InferenceSession(
                    str(self.path), opts, providers=["CPUExecutionProvider"])
                inp = self.session.get_inputs()[0]
                self.input_name = inp.name
                self.size = (int(inp.shape[2]), int(inp.shape[3]))
                self.backend = "onnxruntime"
            except ImportError:
                if backend == "onnxruntime":
                    raise
        if self.session is None:                    # OpenCV fallback
            self.net = cv2.dnn.readNetFromONNX(str(self.path))
            self.size = self._onnx_input_size()
            self.backend = "opencv-dnn"

    def _onnx_input_size(self):
        try:
            import onnx
            m = onnx.load(str(self.path))
            d = m.graph.input[0].type.tensor_type.shape.dim
            return int(d[2].dim_value), int(d[3].dim_value)
        except Exception:
            return 384, 640

    # ------------------------------------------------------------------
    def infer(self, rgb):
        """Raw network output for one RGB image: (N, 4 + classes)."""
        img, r, left, top = letterbox(rgb, self.size)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None],
                                 dtype=np.float32) / 255.0
        if self.session is not None:
            out = self.session.run(None, {self.input_name: x})[0]
        else:
            self.net.setInput(x)
            out = self.net.forward()
        return out[0].T, r, left, top

    def __call__(self, rgb, conf=None):
        """Detect. Returns (N, 6) float32: x1, y1, x2, y2, conf, class."""
        conf = self.conf if conf is None else conf
        pred, r, left, top = self.infer(rgb)
        scores = pred[:, 4:]
        cls = scores.argmax(axis=1)
        sc = scores[np.arange(len(cls)), cls]
        m = sc >= conf
        if not m.any():
            return np.zeros((0, 6), np.float32)
        b, sc, cls = pred[m, :4], sc[m], cls[m]
        boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                          b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
        h, w = rgb.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        # per-class NMS (boxes of different classes are shifted apart) ...
        off = cls[:, None].astype(np.float32) * 4096.0
        keep = nms(boxes + off, sc, self.iou)
        boxes, sc, cls = boxes[keep], sc[keep], cls[keep]
        # ... then drop a second label sitting on the very same object
        keep = nms(boxes, sc, self.agnostic_iou)
        out = np.concatenate([boxes[keep], sc[keep, None],
                              cls[keep, None].astype(np.float32)], axis=1)
        return out.astype(np.float32)
