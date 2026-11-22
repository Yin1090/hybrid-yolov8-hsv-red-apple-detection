"""Tests for Layer 3 CLAHE(V), split-range HSV, ROI calibration, morphology, and geometry."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hybrid_apple_detector.config import ROI_HALF, AdaptiveHSVProfile, PipelineConfig
from hybrid_apple_detector.refinement import (
    AdaptiveHSVCalibrator,
    build_split_red_mask,
    compute_geometry_metrics,
    detect_in_rect,
    preprocess_roi_hsv,
    refine_candidate,
)
from hybrid_apple_detector.types import DetectionCandidate


def test_preprocess_applies_clahe_only_to_v_channel() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = (15, 20, 90)
    img[:, 50:] = (25, 30, 210)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv_plain = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv_eq = preprocess_roi_hsv(img)

    assert np.array_equal(hsv_eq[..., 0], hsv_plain[..., 0])
    assert np.array_equal(hsv_eq[..., 1], hsv_plain[..., 1])
    assert not np.array_equal(hsv_eq[..., 2], hsv_plain[..., 2])


def test_split_range_hsv_captures_low_and_high_red_rejects_orange_green() -> None:
    # Construct 4 quadrants in BGR: orange-red (H~4), purple-red (H~172), orange/yellow (H~22), green (H~60)
    hsv_canvas = np.zeros((120, 120, 3), dtype=np.uint8)
    hsv_canvas[:60, :60] = (4, 210, 190)      # low red
    hsv_canvas[:60, 60:] = (172, 210, 190)   # high red
    hsv_canvas[60:, :60] = (22, 220, 200)    # orange-yellow distractor
    hsv_canvas[60:, 60:] = (60, 200, 150)    # green foliage

    mask = build_split_red_mask(
        hsv_canvas,
        AdaptiveHSVProfile(),
        median_ksize=5,
        morph_ksize=7,
        open_iter=1,
        close_iter=3,
    )
    assert np.mean(mask[10:50, 10:50] > 0) > 0.90
    assert np.mean(mask[10:50, 70:110] > 0) > 0.90
    assert np.count_nonzero(mask[70:110, 10:50]) == 0
    assert np.count_nonzero(mask[70:110, 70:110]) == 0


def test_adaptive_hsv_calibrator_roi_half_and_reset() -> None:
    assert ROI_HALF == 100
    frame = np.full((400, 400, 3), (60, 60, 60), dtype=np.uint8)
    # Paint a rich red apple patch centered at (200, 200)
    cv2.circle(frame, (200, 200), 65, (25, 25, 205), -1)

    cal = AdaptiveHSVCalibrator()
    prof, rect, logs = cal.calibrate_from_point(frame, 200, 200, roi_half=ROI_HALF)
    assert rect == (100, 100, 300, 300)
    assert prof.sampled_pixels > 0
    assert prof.low_hue_range[0] >= 0 and prof.low_hue_range[1] <= 15
    assert prof.high_hue_range[0] >= 160 and prof.high_hue_range[1] <= 179
    assert 40 <= prof.s_min <= 185
    assert 30 <= prof.v_min <= 175
    assert logs[0].startswith("[ROI CLICK] added ")
    assert logs[1].startswith("[HSV CALC] H=[")

    reset_prof = cal.reset()
    assert reset_prof.low_hue_range == (0, 15)
    assert reset_prof.high_hue_range == (160, 179)
    assert reset_prof.sampled_pixels == 0


def test_morphology_close_iter3_bridges_glare_gap_vs_iter0() -> None:
    # Draw a red circle bisected by a specular white glare band
    frame = np.full((140, 140, 3), (190, 190, 190), dtype=np.uint8)
    cv2.circle(frame, (70, 70), 40, (18, 20, 208), -1)
    cv2.line(frame, (20, 68), (120, 72), (245, 245, 245), 9)

    cand = DetectionCandidate(bbox=(25, 25, 115, 115), yconf=0.72)
    res_close3 = refine_candidate(frame, cand, config=PipelineConfig(close_iter=3))
    res_close0 = refine_candidate(frame, cand, config=PipelineConfig(close_iter=0))

    assert res_close3.passed is True
    assert res_close3.solidity > 0.60
    assert res_close3.extent > 0.40
    # Without morphological closing, the glare band splits the mask in half, dropping extent/circularity/red_ratio
    assert res_close0.passed is False


@pytest.mark.parametrize(
    "shape_kind,expected_reason_prefix",
    [
        ("non_red", "red_ratio"),
        ("ribbon_low_extent", "extent"),
        ("sticker_aspect_ratio", "aspect_ratio"),
        ("concave_ring_low_solidity", "solidity"),
    ],
)
def test_geometric_and_red_ratio_gates_reject_invalid_shapes(
    shape_kind: str, expected_reason_prefix: str
) -> None:
    frame = np.full((160, 160, 3), (80, 80, 80), dtype=np.uint8)
    if shape_kind == "non_red":
        cv2.circle(frame, (80, 80), 40, (20, 165, 230), -1)  # orange
    elif shape_kind == "ribbon_low_extent":
        cv2.line(frame, (25, 25), (135, 135), (20, 22, 210), 22)
    elif shape_kind == "sticker_aspect_ratio":
        cv2.rectangle(frame, (20, 65), (140, 90), (20, 22, 210), -1)
    elif shape_kind == "concave_ring_low_solidity":
        # Thick horseshoe arc: circularity > 0.25, solidity <= 0.60
        cv2.ellipse(frame, (80, 80), (44, 44), 0, 55, 305, (20, 22, 210), 22)

    cand = DetectionCandidate(bbox=(10, 10, 150, 150), yconf=0.80)
    res = refine_candidate(frame, cand, config=PipelineConfig())
    assert res.passed is False
    assert res.reject_reason is not None
    assert res.reject_reason.startswith(expected_reason_prefix)


def test_circularity_gate_rejects_spiky_star_polygon() -> None:
    mask = np.zeros((160, 160), dtype=np.uint8)
    # Create a deeply serrated polygon with high perimeter relative to area (circularity <= 0.25)
    pts = []
    for k in range(24):
        ang = 2.0 * np.pi * k / 24.0
        rad = 60 if k % 2 == 0 else 28
        pts.append([int(80 + rad * np.cos(ang)), int(80 + rad * np.sin(ang))])
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    _, circ, _, _, _, has_cnt = compute_geometry_metrics(mask)
    assert has_cnt
    assert circ <= 0.25


def test_detect_in_rect_translates_coordinates() -> None:
    frame = np.full((300, 300, 3), (60, 110, 50), dtype=np.uint8)
    cv2.circle(frame, (210, 190), 34, (20, 22, 208), -1)
    refs, full_mask = detect_in_rect(frame, rect=(140, 120, 280, 260))
    assert len(refs) == 1
    x1, y1, x2, y2 = refs[0].candidate.bbox
    assert 170 <= x1 <= 185 and 150 <= y1 <= 165
    assert 235 <= x2 <= 250 and 215 <= y2 <= 230
    assert full_mask.shape == (300, 300)
