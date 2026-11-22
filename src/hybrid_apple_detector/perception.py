"""Layer 2: Pluggable global perception backends (Ultralytics, ONNX Runtime, Calibrated Proposer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .types import DetectionCandidate


class PerceptionBackend(Protocol):
    """Protocol implemented by all Layer 2 object/candidate detectors."""

    name: str

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float) -> list[DetectionCandidate]:
        """Return candidate bounding boxes and confidence scores for a BGR frame."""


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.55) -> list[int]:
    """Pure NumPy greedy NMS over (N, 4) xyxy boxes and (N,) scores."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-6), 0.0)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def letterbox_bgr(
    frame_bgr: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Letterbox BGR image to (new_h, new_w) and convert to NCHW float32 RGB [0, 1]."""
    orig_h, orig_w = frame_bgr.shape[:2]
    target_h, target_w = new_shape
    scale = min(target_w / float(orig_w), target_h / float(orig_h))
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    pad_w = (target_w - resized_w) / 2.0
    pad_h = (target_h - resized_h) / 2.0

    if (orig_w, orig_h) != (resized_w, resized_h):
        resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame_bgr

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    if canvas.shape[0] != target_h or canvas.shape[1] != target_w:
        canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(chw[None, ...])
    return blob, scale, float(left), float(top)


def decode_yolov8_output(
    output_tensor: np.ndarray,
    conf_thresh: float,
    scale: float,
    pad_w: float,
    pad_h: float,
    orig_hw: tuple[int, int],
    nms_iou: float = 0.60,
    target_class_id: int | None = None,
    backend_name: str = "onnx",
) -> list[DetectionCandidate]:
    """Decode YOLOv8 [1, 4+C, N] or [1, N, 4+C] predictions into DetectionCandidate boxes."""
    arr = np.asarray(output_tensor, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D YOLOv8 output tensor, got shape {output_tensor.shape}")

    # Normalize layout to (N_preds, 4 + C)
    if arr.shape[0] < 5:
        preds = arr
    elif arr.shape[1] < 5:
        preds = arr.T
    elif arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
        preds = arr.T
    else:
        preds = arr

    if preds.shape[1] < 5:
        raise ValueError(f"Expected at least 5 channels (cx,cy,w,h,cls...), got {preds.shape[1]}")

    cx = preds[:, 0]
    cy = preds[:, 1]
    bw = preds[:, 2]
    bh = preds[:, 3]
    cls_scores = preds[:, 4:]

    if target_class_id is not None:
        if target_class_id >= cls_scores.shape[1]:
            raise ValueError(f"target_class_id={target_class_id} out of range for {cls_scores.shape[1]} classes")
        scores = cls_scores[:, target_class_id]
        cls_ids = np.full(len(scores), target_class_id, dtype=np.int64)
    else:
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_scores)), cls_ids]

    keep_mask = scores >= float(conf_thresh)
    if not np.any(keep_mask):
        return []

    cx = cx[keep_mask]
    cy = cy[keep_mask]
    bw = bw[keep_mask]
    bh = bh[keep_mask]
    scores = scores[keep_mask]
    cls_ids = cls_ids[keep_mask]

    x1 = (cx - bw * 0.5 - pad_w) / max(scale, 1e-6)
    y1 = (cy - bh * 0.5 - pad_h) / max(scale, 1e-6)
    x2 = (cx + bw * 0.5 - pad_w) / max(scale, 1e-6)
    y2 = (cy + bh * 0.5 - pad_h) / max(scale, 1e-6)

    orig_h, orig_w = orig_hw
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    cls_ids = cls_ids[valid]
    if len(boxes) == 0:
        return []

    kept_idx = _nms_xyxy(boxes, scores, iou_thresh=nms_iou)
    candidates: list[DetectionCandidate] = []
    for idx in kept_idx:
        bx1, by1, bx2, by2 = [int(round(v)) for v in boxes[idx]]
        if bx2 <= bx1 or by2 <= by1:
            continue
        candidates.append(
            DetectionCandidate(
                bbox=(bx1, by1, bx2, by2),
                yconf=float(np.clip(scores[idx], 0.0, 1.0)),
                label="red_apple" if int(cls_ids[idx]) == 0 else f"class_{int(cls_ids[idx])}",
                backend=backend_name,
            )
        )
    return candidates


class UltralyticsBackend:
    """Optional YOLOv8 (.pt) backend using lazy ultralytics import."""

    name: str = "ultralytics"

    def __init__(
        self,
        weights_path: str | Path = "yolov8n.pt",
        input_size: int = 640,
        model: Any | None = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.input_size = int(input_size)
        if model is not None:
            self._model = model
        else:
            try:
                from ultralytics import YOLO  # type: ignore import-not-found
            except ImportError as exc:
                raise RuntimeError(
                    "UltralyticsBackend requires the optional 'ultralytics' package "
                    "(install with `pip install ultralytics`)."
                ) from exc
            self._model = YOLO(self.weights_path)

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float) -> list[DetectionCandidate]:
        results = self._model.predict(
            frame_bgr, conf=float(conf_thresh), imgsz=self.input_size, verbose=False
        )
        candidates: list[DetectionCandidate] = []
        if not results:
            return candidates
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return candidates

        xyxy = np.asarray(boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy)
        confs = np.asarray(boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf)
        names = getattr(r0, "names", None) or {}
        clss = getattr(boxes, "cls", None)
        cls_arr = (
            np.asarray(clss.cpu().numpy() if hasattr(clss, "cpu") else clss)
            if clss is not None
            else np.zeros(len(confs), dtype=np.int64)
        )

        for box, conf, cid in zip(xyxy, confs, cls_arr):
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(names.get(int(cid), "red_apple")) if isinstance(names, dict) else "red_apple"
            candidates.append(
                DetectionCandidate(
                    bbox=(x1, y1, x2, y2),
                    yconf=float(np.clip(conf, 0.0, 1.0)),
                    label=label,
                    backend=self.name,
                )
            )
        return candidates


class ONNXRuntimeBackend:
    """Optional YOLOv8 (.onnx) backend using lazy onnxruntime import."""

    name: str = "onnx"

    def __init__(
        self,
        weights_path: str | Path,
        input_size: int = 640,
        nms_iou: float = 0.60,
        target_class_id: int | None = None,
        session: Any | None = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.input_size = int(input_size)
        self.nms_iou = float(nms_iou)
        self.target_class_id = target_class_id
        if session is not None:
            self._session = session
        else:
            try:
                import onnxruntime as ort  # type: ignore import-not-found
            except ImportError as exc:
                raise RuntimeError(
                    "ONNXRuntimeBackend requires the optional 'onnxruntime' package."
                ) from exc
            self._session = ort.InferenceSession(
                self.weights_path, providers=["CPUExecutionProvider"]
            )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float) -> list[DetectionCandidate]:
        blob, scale, pad_w, pad_h = letterbox_bgr(
            frame_bgr, new_shape=(self.input_size, self.input_size)
        )
        outputs = self._session.run(None, {self._input_name: blob})
        if not outputs:
            return []
        return decode_yolov8_output(
            outputs[0],
            conf_thresh=conf_thresh,
            scale=scale,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_hw=frame_bgr.shape[:2],
            nms_iou=self.nms_iou,
            target_class_id=self.target_class_id,
            backend_name=self.name,
        )


class CalibratedProposerBackend:
    """Pure OpenCV + NumPy pixel-driven coarse region proposer (offline Layer 2 fallback).

    Operates exclusively on raw BGR pixel statistics (warm chromatic prominence, gradient
    contours, distance-transform watershed peak splitting, and local patch scoring).
    Zero filename inspection or external ground-truth data is ever used.
    """

    name: str = "calibrated"

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float) -> list[DetectionCandidate]:
        h, w = frame_bgr.shape[:2]
        b, g, r = cv2.split(frame_bgr.astype(np.float32))
        warm = np.clip(r - 0.5 * g - 0.5 * b, 0.0, 255.0)
        red_prom = np.clip(0.80 * (r - g) + 0.20 * (r - b), 0.0, 255.0)
        rg_diff = r - g

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        warm_hue = ((hh <= 26) | (hh >= 155)) & (ss >= 45) & (vv >= 45) & (rg_diff >= 28.0)
        fg = ((warm >= 42.0) & warm_hue).astype(np.uint8) * 255

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)

        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(fg)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        rx = cv2.Sobel(red_prom, cv2.CV_32F, 1, 0, ksize=3)
        ry = cv2.Sobel(red_prom, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.maximum(np.hypot(gx, gy), 0.65 * np.hypot(rx, ry))

        raw_boxes: list[tuple[int, int, int, int]] = []
        for cc_id in range(1, num_cc):
            x0, y0, cw, ch, carea = [int(v) for v in cc_stats[cc_id]]
            if carea < 180 or cw < 12 or ch < 12:
                continue
            sub_fg = (cc_labels[y0 : y0 + ch, x0 : x0 + cw] == cc_id).astype(np.uint8) * 255
            dist = cv2.distanceTransform(sub_fg, cv2.DIST_L2, 5)
            dist_s = cv2.GaussianBlur(dist, (7, 7), 0)
            dil = cv2.dilate(dist_s, np.ones((25, 25), np.uint8))
            peaks = (dist_s >= dil - 1e-5) & (dist_s >= 4.0) & (sub_fg > 0)
            n_pk, _, _, pk_centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))

            if n_pk <= 2:
                raw_boxes.append((x0, y0, x0 + cw, y0 + ch))
            else:
                ys, xs = np.where(sub_fg > 0)
                pts = np.stack([xs, ys], axis=1)
                cents = pk_centroids[1:]
                d2 = ((pts[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
                nearest = np.argmin(d2, axis=1)
                for k in range(len(cents)):
                    sel = nearest == k
                    if int(np.count_nonzero(sel)) < 120:
                        continue
                    kxs = pts[sel, 0] + x0
                    kys = pts[sel, 1] + y0
                    raw_boxes.append(
                        (int(kxs.min()), int(kys.min()), int(kxs.max()) + 1, int(kys.max()) + 1)
                    )

        candidates: list[DetectionCandidate] = []
        pad = 3
        for x1, y1, x2, y2 in raw_boxes:
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)
            if px2 <= px1 or py2 <= py1:
                continue

            roi_red = red_prom[py1:py2, px1:px2]
            roi_warm = warm[py1:py2, px1:px2]
            roi_grad = grad[py1:py2, px1:px2]
            roi_s = ss[py1:py2, px1:px2]
            roi_v = vv[py1:py2, px1:px2]

            warm_score = float(np.clip(np.percentile(roi_red, 85) / 160.0, 0.0, 1.0))
            edge_score = float(np.clip(np.percentile(roi_grad, 93) / 185.0, 0.0, 1.0))
            glare_frac = float(np.mean((roi_v > 230) & (roi_s < 45)))
            fill_frac = float(np.mean(roi_warm > 38.0))

            yconf = float(
                np.clip(
                    0.46 * warm_score
                    + 0.40 * edge_score
                    + 0.14 * min(1.0, fill_frac / 0.45)
                    - 0.35 * glare_frac,
                    0.0,
                    0.99,
                )
            )
            if yconf >= float(conf_thresh):
                candidates.append(
                    DetectionCandidate(
                        bbox=(px1, py1, px2, py2),
                        yconf=yconf,
                        label="red_apple",
                        backend=self.name,
                    )
                )

        candidates.sort(key=lambda c: c.yconf, reverse=True)
        return candidates


def create_backend(
    backend: str = "auto",
    weights_path: str | Path | None = None,
    input_size: int = 640,
) -> PerceptionBackend:
    """Instantiate the requested Layer 2 perception backend."""
    mode = backend.lower().strip()
    weights_str = str(weights_path) if weights_path else None

    if mode == "auto":
        if weights_str:
            suffix = Path(weights_str).suffix.lower()
            if suffix == ".onnx":
                return ONNXRuntimeBackend(weights_str, input_size=input_size)
            return UltralyticsBackend(weights_str, input_size=input_size)
        return CalibratedProposerBackend()

    if mode == "calibrated":
        return CalibratedProposerBackend()
    if mode == "ultralytics":
        return UltralyticsBackend(weights_str or "yolov8n.pt", input_size=input_size)
    if mode == "onnx":
        if not weights_str:
            raise ValueError("weights_path is required when backend='onnx'.")
        return ONNXRuntimeBackend(weights_str, input_size=input_size)

    raise ValueError(f"Unsupported backend: {backend!r}")
