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


def test_rotation_invariant_aspect_ratio_preserves_axis_boundaries() -> None:
    # Axis-aligned exact boundaries [0.5, 2.0] preserved
    for w, h, expected_pass in [(100, 200, True), (200, 100, True), (100, 202, False), (202, 100, False)]:
        mask = np.full((h, w), 255, dtype=np.uint8)
        _, _, _, _, ar, has_cnt = compute_geometry_metrics(mask)
        assert has_cnt
        assert ((0.5 <= ar <= 2.0) is expected_pass)

    # 45-degree rotated 85x35 (2.43:1) bar rejected via minAreaRect oriented aspect ratio
    frame = np.full((240, 240, 3), (75, 110, 80), dtype=np.uint8)
    pts = cv2.boxPoints(((120, 120), (85, 35), 45.0)).astype(np.int32)
    cv2.fillPoly(frame, [pts], (18, 20, 205))
    refs, _ = detect_in_rect(frame)
    assert len(refs) == 0


def test_refine_and_calibrator_reject_nan_oob_and_empty_rois() -> None:
    frame = np.full((128, 128, 3), (18, 20, 205), dtype=np.uint8)

    nan_res = refine_candidate(frame, DetectionCandidate((float("nan"), 10, 50, 50), yconf=0.80))  # type: ignore[arg-type]
    assert nan_res.passed is False
    assert nan_res.reject_reason == "invalid_bbox"

    for bad_box in [(-50, -50, -10, -10), (128, 128, 178, 178), (20, 20, 20, 20), (60, 60, 20, 20)]:
        bad_res = refine_candidate(frame, DetectionCandidate(bad_box, yconf=0.90))
        assert bad_res.passed is False
        assert bad_res.reject_reason == "invalid_roi"

    cal = AdaptiveHSVCalibrator()
    prof_pt, rect_pt, logs_pt = cal.calibrate_from_point(frame, -200, -200)
    assert rect_pt is None
    assert prof_pt.sampled_pixels == 0
    assert logs_pt == ["[ROI CLICK] ignored out-of-bounds"]

    prof_rc, rect_rc, logs_rc = cal.calibrate_from_rect(frame, (-50, -50, -10, -10))
    assert rect_rc is None
    assert prof_rc.sampled_pixels == 0
    assert logs_rc == ["[ROI CLICK] ignored out-of-bounds"]


def test_admissibility_solidity_boundary_and_thick_c_ribbon_rejection() -> None:
    from hybrid_apple_detector.refinement import _check_admissibility

    cfg = PipelineConfig()
    assert _check_admissibility(0.20, 0.60, 0.60, 0.50, 1.0, True, cfg)[0] is False
    assert _check_admissibility(0.20, 0.60, 0.600001, 0.50, 1.0, True, cfg)[0] is True

    # Thick C_r70_t40 has solidity ~0.64 > 0.60 and passes circularity/extent,
    # yet is rejected for its annular concave void.
    scene = np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)
    cv2.ellipse(scene, (320, 240), (70, 70), 0, 35, 325, (18, 20, 205), thickness=40)
    res = refine_candidate(scene, DetectionCandidate((227, 147, 400, 334), yconf=0.60), config=cfg)
    assert res.solidity > 0.60
    assert res.circularity > 0.25
    assert res.passed is False
    assert res.reject_reason == "concave_void_ribbon"
    refs, _ = detect_in_rect(scene, config=cfg)
    assert len(refs) == 0
