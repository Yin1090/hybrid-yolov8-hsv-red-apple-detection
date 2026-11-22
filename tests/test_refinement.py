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


def test_padded_dt_and_closed_ring_near_closed_arc_clipped_and_uj_rejection() -> None:
    from hybrid_apple_detector.refinement import (
        _padded_distance_transform,
        extract_blob_components,
        is_annular_or_ribbon_void,
    )

    solid_60 = np.full((60, 60), 255, dtype=np.uint8)
    dist_60 = _padded_distance_transform(solid_60)
    assert float(dist_60.max()) == pytest.approx(30.0, abs=1.0)
    assert is_annular_or_ribbon_void(solid_60) is False
    assert len(extract_blob_components(solid_60)) == 1

    cfg = PipelineConfig()
    # Closed ring (r=65, t=36), near-closed C-arc (ang 20..340), clipped C-ribbon, U_t42, J_t42
    shapes = []

    def _bg() -> np.ndarray:
        return np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)

    s_ring = _bg()
    cv2.circle(s_ring, (320, 240), 65, (18, 20, 205), thickness=36)
    shapes.append(("ring", s_ring, (220, 140, 420, 340)))

    s_arc = _bg()
    cv2.ellipse(s_arc, (320, 240), (70, 70), 0, 20, 340, (18, 20, 205), thickness=40)
    shapes.append(("near_closed_c", s_arc, (220, 140, 420, 340)))

    s_clip = _bg()
    cv2.ellipse(s_clip, (20, 240), (70, 70), 0, -145, 145, (18, 20, 205), thickness=40)
    shapes.append(("clipped_c", s_clip, (0, 140, 120, 340)))

    s_u = _bg()
    cv2.polylines(
        s_u,
        [np.array([[250, 150], [250, 300], [370, 300], [370, 150]], dtype=np.int32)],
        False,
        (18, 20, 205),
        42,
    )
    shapes.append(("u_t42", s_u, (220, 120, 400, 330)))

    s_j = _bg()
    cv2.line(s_j, (360, 130), (360, 280), (18, 20, 205), 42)
    cv2.ellipse(s_j, (310, 280), (50, 50), 0, 0, 180, (18, 20, 205), 42)
    shapes.append(("j_t42", s_j, (230, 100, 390, 360)))

    for label, scene, box in shapes:
        res = refine_candidate(scene, DetectionCandidate(box, yconf=0.60), config=cfg)
        assert res.passed is False, f"{label} unexpectedly passed refine_candidate"
        refs, _ = detect_in_rect(scene, config=cfg)
        assert len(refs) == 0, f"{label} unexpectedly passed detect_in_rect"


def test_cluster_splitting_collinear_double_leaf_rotated_3t_4sq_4diam_and_unequal() -> None:
    import math

    cfg = PipelineConfig()

    def _draw_shaded(img: np.ndarray, cx: int, cy: int, r: int) -> None:
        y0, y1 = max(0, cy - r - 2), min(img.shape[0], cy + r + 3)
        x0, x1 = max(0, cx - r - 2), min(img.shape[1], cx + r + 3)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = (xx - cx) / float(r)
        dy = (yy - cy) / float(r)
        d2 = dx * dx + dy * dy
        m = d2 <= 1.0
        nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
        ndotl = np.clip(-0.45 * dx - 0.45 * dy + 0.77 * nz, 0.30, 1.0)
        patch = img[y0:y1, x0:x1].astype(np.float32)
        for c, b in enumerate((18, 20, 205)):
            patch[..., c] = np.where(m, np.clip(b * (0.52 + 0.58 * ndotl), 0.0, 255.0), patch[..., c])
        img[y0:y1, x0:x1] = patch.astype(np.uint8)

    # 3-in-a-row collinear shaded cluster -> 3 (no middle-apple halving)
    im_row = np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)
    for k in (-1, 0, 1):
        _draw_shaded(
            im_row,
            320 + int(round(k * 54 * math.cos(math.radians(30)))),
            240 + int(round(k * 54 * math.sin(math.radians(30)))),
            32,
        )
    refs_row, _ = detect_in_rect(im_row, config=cfg)
    assert len(refs_row) == 3

    # Double-leaf-notched single apple -> 1
    im_leaf = np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)
    cv2.circle(im_leaf, (320, 240), 36, (18, 20, 205), -1)
    cv2.ellipse(im_leaf, (292, 220), (14, 7), 25, 0, 360, (25, 125, 28), -1)
    cv2.ellipse(im_leaf, (348, 222), (14, 7), -25, 0, 360, (25, 125, 28), -1)
    refs_leaf, _ = detect_in_rect(im_leaf, config=cfg)
    assert len(refs_leaf) == 1

    # Rotated 3T_r35_d48 at 30 deg -> 3
    im_3t = np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)
    rad = 48.0 / math.sqrt(3.0)
    for k in range(3):
        th = math.radians(30 + 120 * k - 90)
        cv2.circle(im_3t, (int(round(320 + rad * math.cos(th))), int(round(240 + rad * math.sin(th)))), 35, (18, 20, 205), -1)
    refs_3t, _ = detect_in_rect(im_3t, config=cfg)
    assert len(refs_3t) == 3

    # 4Sq_r35_d50 & 4Diam_r35_d50 -> 4
    for pts in (
        [(295, 215), (345, 215), (295, 265), (345, 265)],
        [(320, 205), (355, 240), (320, 275), (285, 240)],
    ):
        im_4 = np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)
        for cx, cy in pts:
            cv2.circle(im_4, (cx, cy), 35, (18, 20, 205), -1)
        refs_4, _ = detect_in_rect(im_4, config=cfg)
        assert len(refs_4) == 4

    # Unequal 18+38 @ d=48 -> 2
    im_un = np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)
    cv2.circle(im_un, (290, 240), 18, (18, 20, 205), -1)
    cv2.circle(im_un, (338, 240), 38, (18, 20, 205), -1)
    refs_un, _ = detect_in_rect(im_un, config=cfg)
    assert len(refs_un) == 2
