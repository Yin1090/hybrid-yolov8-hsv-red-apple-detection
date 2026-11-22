"""End-to-end pipeline, 3-panel visualization, interactive-equivalent ROI click, and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hybrid_apple_detector import (
    WINDOW_TITLE,
    HybridAppleDetector,
    PipelineConfig,
    render_three_panel,
)
from hybrid_apple_detector.acquisition import load_image
from hybrid_apple_detector.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


def test_e2e_orchard_dense_crate_bagged_and_negative_scenarios() -> None:
    orchard = load_image(SAMPLES_DIR / "orchard_scene.png")
    crate = load_image(SAMPLES_DIR / "dense_crate_scene.png")
    bagged = load_image(SAMPLES_DIR / "bagged_apples_scene.png")
    negative = load_image(SAMPLES_DIR / "negative_foliage_scene.png")

    # Orchard: 12/12
    res_orchard = HybridAppleDetector(PipelineConfig(yolo_conf=0.65)).process_frame(orchard)
    assert len(res_orchard.detections) == 12

    # Dense crate: 56 YOLO candidates -> 46 valid apples in Hybrid @ 0.65; drops to ~20 @ 0.85
    res_crate_65 = HybridAppleDetector(PipelineConfig(yolo_conf=0.65)).process_frame(crate)
    assert len(res_crate_65.candidates) == 56
    assert len(res_crate_65.detections) == 46

    res_crate_85 = HybridAppleDetector(PipelineConfig(yolo_conf=0.85)).process_frame(crate)
    assert len(res_crate_85.detections) == 20

    # Pure YOLO at 0.65 keeps all 56 including 10 false-positive distractors
    res_crate_yolo = HybridAppleDetector(PipelineConfig(mode="yolo", yolo_conf=0.65)).process_frame(crate)
    assert len(res_crate_yolo.detections) == 56

    # Bagged apples: 16/16 @ 0.65, drops to 4 @ 0.85, and drops sharply if close_iter=0
    res_bag_65 = HybridAppleDetector(PipelineConfig(yolo_conf=0.65)).process_frame(bagged)
    assert len(res_bag_65.detections) == 16

    res_bag_85 = HybridAppleDetector(PipelineConfig(yolo_conf=0.85)).process_frame(bagged)
    assert len(res_bag_85.detections) == 4

    res_bag_noclose = HybridAppleDetector(
        PipelineConfig(yolo_conf=0.65, close_iter=0)
    ).process_frame(bagged)
    assert len(res_bag_noclose.detections) < 10

    # Negative foliage: 0/0 across all modes
    for mode in ("hybrid", "yolo", "hsv", "auto"):
        res_neg = HybridAppleDetector(PipelineConfig(mode=mode)).process_frame(negative)
        assert len(res_neg.detections) == 0


def test_all_four_modes_and_roi_click_calibration() -> None:
    bagged = load_image(SAMPLES_DIR / "bagged_apples_scene.png")
    detector = HybridAppleDetector(PipelineConfig(mode="hybrid", yolo_conf=0.65))

    # Click ROI over an apple: updates HSV thresholds and restricts mask panel to ROI,
    # while still counting all 16 full-frame apples.
    logs = detector.click_roi(bagged, 190, 155)
    assert any(line.startswith("[ROI CLICK] added ") for line in logs)
    assert any(line.startswith("[HSV CALC] H=[") for line in logs)

    res = detector.process_frame(bagged)
    assert len(res.detections) == 16
    assert res.active_roi == (90, 55, 290, 255)
    # Pixels outside active_roi in Mask panel are strictly black (0)
    assert np.count_nonzero(res.hsv_mask[:, 310:]) == 0

    for mode in ("yolo", "hybrid"):
        detector.set_mode(mode)
        r = detector.process_frame(bagged)
        assert r.mode == mode
        assert len(r.detections) == 16

    # In hsv and auto modes, active_roi scopes both detection and hsv_mask
    for mode in ("hsv", "auto"):
        detector.set_mode(mode)
        r_roi = detector.process_frame(bagged)
        assert r_roi.mode == mode
        assert 1 <= len(r_roi.detections) < 16
        for det in r_roi.detections:
            bx1, by1, bx2, by2 = det.bbox
            assert bx1 >= 90 and by1 >= 55 and bx2 <= 290 and by2 <= 255
            assert np.count_nonzero(r_roi.hsv_mask[by1:by2, bx1:bx2]) > 0

    detector.reset_roi()
    assert detector.active_roi is None

    for mode in ("yolo", "hsv", "auto", "hybrid"):
        detector.set_mode(mode)
        r_full = detector.process_frame(bagged)
        assert r_full.mode == mode
        assert len(r_full.detections) == 16


def test_three_panel_visualizer_and_headless_cli(tmp_path: Path, capsys) -> None:
    assert WINDOW_TITLE == (
        "HYBRID VIP APPLE | Y:YOLO | H:AUTO | C:HSV | A:HYBRID | S:SAVE | R:RESET"
    )
    orchard = load_image(SAMPLES_DIR / "orchard_scene.png")
    detector = HybridAppleDetector(PipelineConfig(yolo_conf=0.65))
    res = detector.process_frame(orchard)
    panel = render_three_panel(orchard, res)
    assert panel.shape == (orchard.shape[0], orchard.shape[1] * 3, 3)

    # CLI single image with --roi-click and --save-json
    out_single = tmp_path / "single"
    rc = cli_main(
        [
            "--source",
            str(SAMPLES_DIR / "bagged_apples_scene.png"),
            "--mode",
            "hybrid",
            "--yolo-conf",
            "0.65",
            "--roi-click",
            "190,155",
            "--output-dir",
            str(out_single),
            "--save-json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[ROI CLICK] added " in captured
    assert "[YOLO] Found 16" in captured
    assert "[HYBRID] Apple Count: 16" in captured
    assert (out_single / "bagged_apples_scene_result.png").exists()
    payload = json.loads((out_single / "detections.json").read_text(encoding="utf-8"))
    assert payload[0]["apple_count"] == 16

    # CLI directory batch mode
    out_dir = tmp_path / "batch"
    rc_dir = cli_main(
        [
            "--source",
            str(SAMPLES_DIR),
            "--output-dir",
            str(out_dir),
            "--save-json",
        ]
    )
    assert rc_dir == 0
    batch_payload = json.loads((out_dir / "detections.json").read_text(encoding="utf-8"))
    counts = {item["source"]: item["apple_count"] for item in batch_payload}
    assert counts["orchard_scene.png"] == 12
    assert counts["dense_crate_scene.png"] == 46
    assert counts["bagged_apples_scene.png"] == 16
    assert counts["negative_foliage_scene.png"] == 0

    # CLI video mode using synthetic AVI
    video_path = tmp_path / "synthetic_apples.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    assert writer.isOpened()
    for _ in range(2):
        frm = np.full((240, 320, 3), (45, 110, 45), dtype=np.uint8)
        cv2.circle(frm, (160, 120), 34, (20, 22, 208), -1)
        writer.write(frm)
    writer.release()

    out_vid = tmp_path / "video_out"
    rc_vid = cli_main(
        ["--source", str(video_path), "--output-dir", str(out_vid), "--save-json"]
    )
    assert rc_vid == 0
    vid_payload = json.loads((out_vid / "detections.json").read_text(encoding="utf-8"))
    assert len(vid_payload) == 2
    assert all(item["apple_count"] == 1 for item in vid_payload)


def test_oob_click_roi_and_cross_resolution_active_roi() -> None:
    bagged = load_image(SAMPLES_DIR / "bagged_apples_scene.png")
    detector = HybridAppleDetector(PipelineConfig(mode="auto"))

    logs_oob = detector.click_roi(bagged, -200, -200)
    assert logs_oob == ["[ROI CLICK] ignored out-of-bounds"]
    assert detector.active_roi is None
    assert detector.profile.sampled_pixels == 0

    logs_rect_oob = detector.select_roi_rect(bagged, (-50, -50, -10, -10))
    assert logs_rect_oob == ["[ROI CLICK] ignored out-of-bounds"]
    assert detector.active_roi is None
    assert detector.profile.sampled_pixels == 0

    # Valid click on 960x720, then process smaller frames in hsv/auto/hybrid modes
    detector.click_roi(bagged, 800, 600)
    assert detector.active_roi == (700, 500, 900, 700)
    small_frame = np.full((200, 200, 3), (18, 20, 205), dtype=np.uint8)
    for mode in ("hsv", "auto", "hybrid"):
        detector.set_mode(mode)
        r_small = detector.process_frame(small_frame)
        if mode in ("hsv", "auto"):
            assert len(r_small.detections) == 0
            assert np.count_nonzero(r_small.hsv_mask) == 0


def test_cli_validation_and_clean_error_exits(tmp_path: Path, capsys) -> None:
    img_path = SAMPLES_DIR / "orchard_scene.png"
    for bad_args in [
        ["--source", str(img_path), "--yolo-conf", "nan"],
        ["--source", str(img_path), "--yolo-conf", "-0.5"],
        ["--source", str(img_path), "--yolo-conf", "1.5"],
        ["--source", str(img_path), "--bypass-conf", "2.0"],
        ["--source", str(img_path), "--roi-click", "abc"],
    ]:
        with pytest.raises(SystemExit) as exc_info:
            cli_main(bad_args)
        assert exc_info.value.code == 2

    rc_missing = cli_main(["--source", str(tmp_path / "missing.png"), "--output-dir", str(tmp_path / "out1")])
    assert rc_missing == 1
    assert "error:" in capsys.readouterr().err

    rc_onnx = cli_main(["--source", str(img_path), "--backend", "onnx", "--output-dir", str(tmp_path / "out2")])
    assert rc_onnx == 2
    assert "error:" in capsys.readouterr().err


def test_yolo_mode_filters_non_finite_and_degenerate_boxes() -> None:
    from hybrid_apple_detector.types import DetectionCandidate

    class _MockInvalidBackend:
        name = "mock"

        def detect(self, frame_bgr: np.ndarray, conf_thresh: float) -> list[DetectionCandidate]:
            return [
                DetectionCandidate((10, 10, 50, 50), yconf=float("nan")),
                DetectionCandidate((float("nan"), 10, 50, 50), yconf=0.92),  # type: ignore[arg-type]
                DetectionCandidate((20, 20, 20, 60), yconf=0.90),
                DetectionCandidate((80, 80, 40, 120), yconf=0.88),
                DetectionCandidate((100, 100, 150, 150), yconf=0.85),
            ]

    det = HybridAppleDetector(PipelineConfig(mode="yolo"), backend=_MockInvalidBackend())
    res = det.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))
    assert len(res.detections) == 1
    assert res.detections[0].bbox == (100, 100, 150, 150)
    assert res.detections[0].score_final == pytest.approx(0.85)


def test_e2e_voids_rejected_and_extended_clusters_preserved_across_modes() -> None:
    import math

    def _bg() -> np.ndarray:
        return np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)

    # 1. Closed ring, near-closed C-arc, clipped C-ribbon, U_t42, J_t42 -> 0 across all modes
    void_scenes = []
    s_ring = _bg()
    cv2.circle(s_ring, (320, 240), 65, (18, 20, 205), thickness=36)
    void_scenes.append(("Ring_r65_t36", s_ring))

    s_arc = _bg()
    cv2.ellipse(s_arc, (320, 240), (70, 70), 0, 20, 340, (18, 20, 205), thickness=40)
    void_scenes.append(("C_r70_t40_ang20_340", s_arc))

    s_clip = _bg()
    cv2.ellipse(s_clip, (20, 240), (70, 70), 0, -145, 145, (18, 20, 205), thickness=40)
    void_scenes.append(("C_clipped_x0", s_clip))

    s_u = _bg()
    cv2.polylines(
        s_u,
        [np.array([[250, 150], [250, 300], [370, 300], [370, 150]], dtype=np.int32)],
        False,
        (18, 20, 205),
        42,
    )
    void_scenes.append(("U_t42", s_u))

    s_j = _bg()
    cv2.line(s_j, (360, 130), (360, 280), (18, 20, 205), 42)
    cv2.ellipse(s_j, (310, 280), (50, 50), 0, 0, 180, (18, 20, 205), 42)
    void_scenes.append(("J_t42", s_j))

    for name, img in void_scenes:
        for mode in ("hybrid", "hsv", "auto"):
            res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=0.15)).process_frame(img)
            assert len(res.detections) == 0, f"{name} failed in {mode}"

    # 2. Collinear 3Row_D30 -> 3, Double-leaf single apple -> 1, 3T_rot30 -> 3, 4Sq/4Diam -> 4, Unequal -> 2
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

    im_row = _bg()
    for k in (-1, 0, 1):
        _draw_shaded(
            im_row,
            320 + int(round(k * 54 * math.cos(math.radians(30)))),
            240 + int(round(k * 54 * math.sin(math.radians(30)))),
            32,
        )

    im_leaf = _bg()
    cv2.circle(im_leaf, (320, 240), 36, (18, 20, 205), -1)
    cv2.ellipse(im_leaf, (292, 220), (14, 7), 25, 0, 360, (25, 125, 28), -1)
    cv2.ellipse(im_leaf, (348, 222), (14, 7), -25, 0, 360, (25, 125, 28), -1)

    im_3t = _bg()
    rad = 48.0 / math.sqrt(3.0)
    for k in range(3):
        th = math.radians(30 + 120 * k - 90)
        cv2.circle(im_3t, (int(round(320 + rad * math.cos(th))), int(round(240 + rad * math.sin(th)))), 35, (18, 20, 205), -1)

    im_4sq = _bg()
    for cx, cy in [(295, 215), (345, 215), (295, 265), (345, 265)]:
        cv2.circle(im_4sq, (cx, cy), 35, (18, 20, 205), -1)

    im_4diam = _bg()
    for cx, cy in [(320, 205), (355, 240), (320, 275), (285, 240)]:
        cv2.circle(im_4diam, (cx, cy), 35, (18, 20, 205), -1)

    im_un = _bg()
    cv2.circle(im_un, (290, 240), 18, (18, 20, 205), -1)
    cv2.circle(im_un, (338, 240), 38, (18, 20, 205), -1)

    for name, img, expected in [
        ("3Row_D30", im_row, 3),
        ("Apple_double_leaf_small", im_leaf, 1),
        ("3T_rot30_r35_d48", im_3t, 3),
        ("4Sq_r35_d50", im_4sq, 4),
        ("4Diam_r35_d50", im_4diam, 4),
        ("Unequal_18_38_d48", im_un, 2),
    ]:
        for mode in ("hybrid", "hsv", "auto"):
            res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=0.65)).process_frame(img)
            assert len(res.detections) == expected, f"{name} failed in {mode}: got {len(res.detections)}"

