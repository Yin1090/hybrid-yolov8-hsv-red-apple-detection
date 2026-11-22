"""Tests for Layer 4 hybrid fusion formula, bypass boundary, and secondary NMS."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_apple_detector.config import PipelineConfig
from hybrid_apple_detector.fusion import (
    compute_iou,
    fuse_and_suppress,
    fuse_score,
    nms_hybrid_boxes,
)
from hybrid_apple_detector.types import DetectionCandidate, HybridDetection, RefinementResult


def test_fuse_score_weighted_and_bypass_boundary() -> None:
    # Below or equal to 0.65 uses weighted 0.45 * yconf + 0.55 * hsv_conf
    s1, b1 = fuse_score(yconf=0.40, hsv_conf=0.80)
    assert b1 is False
    assert s1 == pytest.approx(0.45 * 0.40 + 0.55 * 0.80, abs=1e-6)

    s_eq, b_eq = fuse_score(yconf=0.65, hsv_conf=0.20)
    assert b_eq is False
    assert s_eq == pytest.approx(0.45 * 0.65 + 0.55 * 0.20, abs=1e-6)

    # Strictly greater than 0.65 bypasses directly to yconf
    s_hi, b_hi = fuse_score(yconf=0.66, hsv_conf=0.10)
    assert b_hi is True
    assert s_hi == pytest.approx(0.66, abs=1e-6)


def test_fuse_and_suppress_filters_threshold_and_physical_gate() -> None:
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    low_score = RefinementResult(
        candidate=DetectionCandidate((10, 10, 50, 50), yconf=0.20),
        passed=True,
        reject_reason=None,
        red_ratio=0.30,
        circularity=0.80,
        solidity=0.90,
        extent=0.75,
        aspect_ratio=1.0,
        hsv_conf=0.30,  # 0.45*0.2 + 0.55*0.3 = 0.255 < 0.35
        roi_mask=dummy_mask,
    )
    valid_mid = RefinementResult(
        candidate=DetectionCandidate((100, 100, 150, 150), yconf=0.30),
        passed=True,
        reject_reason=None,
        red_ratio=0.40,
        circularity=0.85,
        solidity=0.95,
        extent=0.78,
        aspect_ratio=1.0,
        hsv_conf=0.70,  # 0.45*0.3 + 0.55*0.7 = 0.52 >= 0.35
        roi_mask=dummy_mask,
    )
    failed_high_yconf = RefinementResult(
        candidate=DetectionCandidate((200, 200, 250, 250), yconf=0.78),
        passed=False,
        reject_reason="red_ratio=0.000<=0.08",
        red_ratio=0.0,
        circularity=0.0,
        solidity=0.0,
        extent=0.0,
        aspect_ratio=0.0,
        hsv_conf=0.0,
        roi_mask=dummy_mask,
    )

    dets = fuse_and_suppress([low_score, valid_mid, failed_high_yconf], PipelineConfig())
    assert len(dets) == 1
    assert dets[0].bbox == (100, 100, 150, 150)
    assert dets[0].score_final == pytest.approx(0.52, abs=1e-5)

    # Strict bypass ablation allows failed_high_yconf through
    dets_strict = fuse_and_suppress(
        [low_score, valid_mid, failed_high_yconf],
        PipelineConfig(strict_bypass=True),
    )
    assert len(dets_strict) == 2


def test_nms_hybrid_boxes_iou_055() -> None:
    d_high = HybridDetection((0, 0, 100, 100), 0.90, 0.90, 0.85, 0.5, 0.8, 0.9, 0.75, 1.0, True)
    # Shift by 15 px -> intersection = 85*85=7225, union = 10000+10000-7225=12775, IoU = 0.5655 > 0.55 -> suppressed
    d_overlap = HybridDetection((15, 15, 115, 115), 0.80, 0.80, 0.80, 0.5, 0.8, 0.9, 0.75, 1.0, True)
    # Shift by 20 px -> intersection = 80*80=6400, union = 13600, IoU = 0.4706 <= 0.55 -> kept
    d_distinct = HybridDetection((20, 20, 120, 120), 0.75, 0.75, 0.75, 0.5, 0.8, 0.9, 0.75, 1.0, True)

    assert compute_iou(d_high.bbox, d_overlap.bbox) > 0.55
    assert compute_iou(d_high.bbox, d_distinct.bbox) <= 0.55

    kept = nms_hybrid_boxes([d_overlap, d_distinct, d_high], iou_threshold=0.55)
    assert [d.score_final for d in kept] == [0.90, 0.75]
