"""Layer 4: Hybrid decision fusion and secondary Non-Maximum Suppression (NMS)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import PipelineConfig
from .types import HybridDetection, RefinementResult


def fuse_score(
    yconf: float,
    hsv_conf: float,
    bypass_conf: float = 0.65,
    yolo_weight: float = 0.45,
    hsv_weight: float = 0.55,
) -> tuple[float, bool]:
    """Compute final hybrid score with high-confidence YOLOv8 bypass.

    Returns (score_final, bypassed).
    """
    if yconf > bypass_conf:
        return float(np.clip(yconf, 0.0, 1.0)), True
    weighted = yolo_weight * float(yconf) + hsv_weight * float(hsv_conf)
    return float(np.clip(weighted, 0.0, 1.0)), False


def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union (IoU) for two (x1, y1, x2, y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def nms_hybrid_boxes(
    detections: Sequence[HybridDetection],
    iou_threshold: float = 0.55,
) -> list[HybridDetection]:
    """Sort detections descending by score_final and suppress overlapping boxes > iou_threshold."""
    ordered = sorted(detections, key=lambda d: d.score_final, reverse=True)
    kept: list[HybridDetection] = []
    for det in ordered:
        if all(compute_iou(det.bbox, prev.bbox) <= iou_threshold for prev in kept):
            kept.append(det)
    return kept


def fuse_and_suppress(
    refinements: Sequence[RefinementResult],
    config: PipelineConfig,
) -> list[HybridDetection]:
    """Fuse Layer 2 + Layer 3 outputs, filter by final threshold, and run secondary NMS."""
    accepted: list[HybridDetection] = []
    for ref in refinements:
        cand = ref.candidate
        if not ref.passed:
            if not (config.strict_bypass and cand.yconf > config.bypass_conf):
                continue

        score_final, bypassed = fuse_score(
            yconf=cand.yconf,
            hsv_conf=ref.hsv_conf,
            bypass_conf=config.bypass_conf,
            yolo_weight=config.yolo_weight,
            hsv_weight=config.hsv_weight,
        )
        if score_final < config.final_conf_thresh:
            continue

        accepted.append(
            HybridDetection(
                bbox=cand.bbox,
                score_final=score_final,
                yconf=cand.yconf,
                hsv_conf=ref.hsv_conf,
                red_ratio=ref.red_ratio,
                circularity=ref.circularity,
                solidity=ref.solidity,
                extent=ref.extent,
                aspect_ratio=ref.aspect_ratio,
                bypassed=bypassed,
                label=cand.label,
            )
        )

    return nms_hybrid_boxes(accepted, iou_threshold=config.nms_iou_thresh)
