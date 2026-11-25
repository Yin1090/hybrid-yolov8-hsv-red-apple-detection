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


def test_e2e_leaf_occluded_singles_and_thick_rings_across_modes() -> None:
    import math

    def _bg() -> np.ndarray:
        return np.full((480, 640, 3), (75, 110, 80), dtype=np.uint8)

    # Apple_3leaf (14x7, 16x8) -> 1 across hybrid/hsv/auto
    for lx, ly in [(14, 7), (16, 8)]:
        im_3l = _bg()
        cv2.circle(im_3l, (320, 240), 36, (18, 20, 205), -1)
        for deg in (30, 150, 270):
            cx = int(round(320 + 34 * math.cos(math.radians(deg))))
            cy = int(round(240 + 34 * math.sin(math.radians(deg))))
            cv2.ellipse(im_3l, (cx, cy), (lx, ly), deg + 90, 0, 360, (35, 130, 45), -1)
        for mode in ("hybrid", "hsv", "auto"):
            for conf in (0.65, 0.15):
                res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=conf)).process_frame(im_3l)
                assert len(res.detections) == 1, f"Apple_3leaf_{lx}x{ly} failed in {mode}@{conf}: {len(res.detections)}"

    # EllipApple_2leaf (35x30, 38x28, 42x28, 45x26 at 0 & 45 deg) -> 1 across hybrid/hsv/auto
    for rx, ry in [(35, 30), (38, 28), (42, 28), (45, 26)]:
        for ang in (0, 45):
            im_el = _bg()
            cv2.ellipse(im_el, (320, 240), (rx, ry), ang, 0, 360, (18, 20, 205), -1)
            sa, ca = math.sin(math.radians(ang)), math.cos(math.radians(ang))
            for sgn in (-1, 1):
                lx = int(round(320 - sgn * (ry - 2) * sa))
                ly = int(round(240 + sgn * (ry - 2) * ca))
                cv2.ellipse(im_el, (lx, ly), (12, 6), ang, 0, 360, (35, 130, 45), -1)
            for mode in ("hybrid", "hsv", "auto"):
                for conf in (0.65, 0.15):
                    res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=conf)).process_frame(im_el)
                    assert len(res.detections) == 1, f"Ellip_{rx}x{ry}_a{ang}_2leaf failed in {mode}@{conf}: {len(res.detections)}"

    # Thick rings Ring_r70_t58..62, Ring_r60_t48, Ring_r80_t65 -> 0 across hybrid@0.65/0.15, hsv@0.65, auto@0.65
    for r, t in [(70, 58), (70, 60), (70, 62), (60, 48), (80, 65)]:
        im_ring = _bg()
        cv2.circle(im_ring, (320, 240), r, (18, 20, 205), thickness=t)
        for mode, conf in [("hybrid", 0.65), ("hybrid", 0.15), ("hsv", 0.65), ("auto", 0.65)]:
            res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=conf)).process_frame(im_ring)
            assert len(res.detections) == 0, f"Ring_r{r}_t{t} failed in {mode}@{conf}: {len(res.detections)}"


def test_e2e_iteration6_glare_rings_offcenter_ellipses_and_clusters_across_modes() -> None:
    import math

    def _bg(color: tuple[int, int, int] = (75, 110, 80)) -> np.ndarray:
        return np.full((480, 640, 3), color, dtype=np.uint8)

    def _draw_shaded(img: np.ndarray, cx: int, cy: int, r: int, floor: float = 0.0) -> None:
        y0, y1 = max(0, cy - r - 2), min(img.shape[0], cy + r + 3)
        x0, x1 = max(0, cx - r - 2), min(img.shape[1], cx + r + 3)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = (xx - cx) / float(r)
        dy = (yy - cy) / float(r)
        d2 = dx * dx + dy * dy
        m = d2 <= 1.0
        nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
        ndotl = np.clip(-0.45 * dx - 0.45 * dy + 0.77 * nz, floor, 1.0)
        patch = img[y0:y1, x0:x1].astype(np.float32)
        for c_i, b in enumerate((18, 20, 205)):
            patch[..., c_i] = np.where(
                m, np.clip(b * (0.52 + 0.58 * ndotl), 0.0, 255.0), patch[..., c_i]
            )
        img[y0:y1, x0:x1] = patch.astype(np.uint8)

    cases: list[tuple[str, np.ndarray, int]] = []

    # 1. Unequal pairs (20,40,48), (16,38,48) -> 2
    for r1, r2, d in [(20, 40, 48), (16, 38, 48)]:
        im = _bg()
        cv2.circle(im, (320 - d // 2, 240), r1, (18, 20, 205), -1)
        cv2.circle(im, (320 + d // 2, 240), r2, (18, 20, 205), -1)
        cases.append((f"Unequal_{r1}_{r2}_d{d}", im, 2))

    # 2. Deep2H_d44_1leaf(12,6), 2H_r35_d52_1leaf_mid, 2Ellip35x30_d54 -> 2
    im_d2 = _bg()
    cv2.circle(im_d2, (298, 240), 35, (18, 20, 205), -1)
    cv2.circle(im_d2, (342, 240), 35, (18, 20, 205), -1)
    cv2.ellipse(im_d2, (320, 207), (12, 6), 0, 0, 360, (35, 130, 45), -1)
    cases.append(("Deep2H_d44_leaf12x6", im_d2, 2))

    im_mid = _bg()
    cv2.circle(im_mid, (294, 240), 35, (18, 20, 205), -1)
    cv2.circle(im_mid, (346, 240), 35, (18, 20, 205), -1)
    cv2.ellipse(im_mid, (320, 210), (10, 5), 0, 0, 360, (35, 130, 45), -1)
    cases.append(("2H_r35_d52_1leaf_mid", im_mid, 2))

    im_2el = _bg()
    cv2.ellipse(im_2el, (293, 240), (35, 30), 0, 0, 360, (18, 20, 205), -1)
    cv2.ellipse(im_2el, (347, 240), (35, 30), 0, 0, 360, (18, 20, 205), -1)
    cases.append(("2Ellip35x30_d54", im_2el, 2))

    # 3. 3T_rot30_1small_leaf(12,6) -> 3, 4Diam_d46_r35 -> 4, 5Pent(floor=0.0) -> 5
    im_3t = _bg()
    rad = 48.0 / math.sqrt(3.0)
    for k in range(3):
        th = math.radians(30 + 120 * k - 90)
        cv2.circle(im_3t, (int(round(320 + rad * math.cos(th))), int(round(240 + rad * math.sin(th)))), 35, (18, 20, 205), -1)
    th0 = math.radians(30 - 90)
    cv2.ellipse(
        im_3t,
        (int(round(320 + (rad + 33) * math.cos(th0))), int(round(240 + (rad + 33) * math.sin(th0)))),
        (12, 6),
        30,
        0,
        360,
        (35, 130, 45),
        -1,
    )
    cases.append(("3T_rot30_1leaf_12x6", im_3t, 3))

    im_4d = _bg()
    off = int(round(46.0 / math.sqrt(2.0)))
    for cx, cy in [(320, 240 - off), (320 + off, 240), (320, 240 + off), (320 - off, 240)]:
        cv2.circle(im_4d, (cx, cy), 35, (18, 20, 205), -1)
    cases.append(("4Diam_d46_r35", im_4d, 4))

    im_5p = _bg()
    rad5 = 54.0 / (2.0 * math.sin(math.pi / 5.0))
    for k in range(5):
        th = math.radians(72 * k - 90)
        _draw_shaded(im_5p, int(round(320 + rad5 * math.cos(th))), int(round(240 + rad5 * math.sin(th))), 32, floor=0.0)
    cases.append(("5Pent_floor0", im_5p, 5))

    # 4. Off-center EllipApple_2leaf & 1-leaf -> 1
    for rx, ry, sh_off, lx, ly, ang in [
        (45, 26, 12, 12, 6, 0),
        (45, 26, 16, 14, 7, 30),
        (42, 28, 16, 12, 6, 90),
        (38, 28, 0, 8, 4, 0),
    ]:
        im_el = _bg()
        cv2.ellipse(im_el, (320, 240), (rx, ry), ang, 0, 360, (18, 20, 205), -1)
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for sgn in (-1, 1):
            cx = int(round(320 + sh_off * ca - sgn * (ry - 2) * sa))
            cy = int(round(240 + sh_off * sa + sgn * (ry - 2) * ca))
            cv2.ellipse(im_el, (cx, cy), (lx, ly), ang, 0, 360, (35, 130, 45), -1)
        cases.append((f"Ellip_{rx}x{ry}_sh{sh_off}_{lx}x{ly}_a{ang}", im_el, 1))

    im_el1 = _bg()
    cv2.ellipse(im_el1, (320, 240), (45, 26), 0, 0, 360, (18, 20, 205), -1)
    cv2.ellipse(im_el1, (320, 264), (12, 6), 0, 0, 360, (35, 130, 45), -1)
    cases.append(("Ellip1L_45x26", im_el1, 1))

    # 5. Glare singles & clusters -> 1 / 2 / 3
    for r, g in [(36, 10), (45, 10), (55, 12)]:
        im_g = _bg()
        cv2.circle(im_g, (320, 240), r, (18, 20, 205), -1)
        cv2.circle(im_g, (320, 240), g, (245, 245, 245), -1)
        cases.append((f"Apple_r{r}_glare{g}", im_g, 1))

    im_2g = _bg()
    cv2.circle(im_2g, (293, 240), 35, (18, 20, 205), -1)
    cv2.circle(im_2g, (347, 240), 35, (18, 20, 205), -1)
    cv2.circle(im_2g, (293, 240), 10, (245, 245, 245), -1)
    cases.append(("2H_glare10", im_2g, 2))

    im_3g = _bg()
    rad_g = 52.0 / math.sqrt(3.0)
    for k in range(3):
        th = math.radians(120 * k - 90)
        cv2.circle(im_3g, (int(round(320 + rad_g * math.cos(th))), int(round(240 + rad_g * math.sin(th)))), 35, (18, 20, 205), -1)
    cv2.circle(im_3g, (320, int(round(240 - rad_g))), 10, (245, 245, 245), -1)
    cases.append(("3T_glare10", im_3g, 3))

    # 6. Ultra-thick rings on green/dark/white backgrounds -> 0
    for bg_name, bg_c in [("green", (75, 110, 80)), ("dark", (30, 35, 30)), ("white", (245, 245, 245))]:
        for r, t in [(60, 74), (70, 89)]:
            im_r = _bg(bg_c)
            cv2.circle(im_r, (320, 240), r, (18, 20, 205), thickness=t)
            cases.append((f"Ring_r{r}_t{t}_{bg_name}", im_r, 0))

    for name, img, expected in cases:
        for mode, conf in [("hybrid", 0.65), ("hybrid", 0.15), ("hsv", 0.65), ("auto", 0.65)]:
            res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=conf)).process_frame(img)
            assert len(res.detections) == expected, (
                f"{name} failed in {mode}@{conf}: got {len(res.detections)}, expected {expected}"
            )


def test_e2e_iteration7_power_voronoi_ellip_rings_thickc_kidney_and_glare() -> None:
    import math

    def _bg(color: tuple[int, int, int] = (75, 110, 80)) -> np.ndarray:
        return np.full((480, 640, 3), color, dtype=np.uint8)

    def _draw_shaded(img: np.ndarray, cx: int, cy: int, r: int, floor: float = 0.0) -> None:
        y0, y1 = max(0, cy - r - 2), min(img.shape[0], cy + r + 3)
        x0, x1 = max(0, cx - r - 2), min(img.shape[1], cx + r + 3)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = (xx - cx) / float(r)
        dy = (yy - cy) / float(r)
        d2 = dx * dx + dy * dy
        m = d2 <= 1.0
        nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
        ndotl = np.clip(-0.45 * dx - 0.45 * dy + 0.77 * nz, floor, 1.0)
        patch = img[y0:y1, x0:x1].astype(np.float32)
        for c_i, b in enumerate((18, 20, 205)):
            patch[..., c_i] = np.where(
                m, np.clip(b * (0.52 + 0.58 * ndotl), 0.0, 255.0), patch[..., c_i]
            )
        img[y0:y1, x0:x1] = patch.astype(np.uint8)

    cases: list[tuple[str, np.ndarray, int, list[tuple[str, float]]]] = []
    all_modes = [("hybrid", 0.65), ("hybrid", 0.15), ("hsv", 0.65), ("auto", 0.65)]

    # 1. Unequal(16, 38, d=48/49) -> 2
    for d in (48, 49):
        im_u = _bg()
        cv2.circle(im_u, (320 - d // 2, 240), 16, (18, 20, 205), -1)
        cv2.circle(im_u, (320 + (d - d // 2), 240), 38, (18, 20, 205), -1)
        cases.append((f"Unequal_16_38_d{d}", im_u, 2, all_modes))

    # 2. Ellip_42x28_ang0_shift4_leaf8x4 -> 1 (specifically guards auto@0.65)
    im_el = _bg()
    cv2.ellipse(im_el, (320, 240), (42, 28), 0, 0, 360, (18, 20, 205), -1)
    for sgn in (-1, 1):
        cv2.ellipse(im_el, (324, 240 + sgn * 25), (8, 4), 0, 0, 360, (35, 130, 45), -1)
    cases.append(("Ellip_42x28_ang0_shift4_leaf8x4", im_el, 1, all_modes))

    # 3. Ring_r40_t52, Ring_r45_t58, Ring_r50_t62 on white & green backgrounds -> 0
    for bg_name, bg_c in [("white", (245, 245, 245)), ("green", (75, 110, 80))]:
        for r, t in [(40, 52), (45, 58), (50, 62)]:
            im_r = _bg(bg_c)
            cv2.circle(im_r, (320, 240), r, (18, 20, 205), thickness=t)
            cases.append((f"Ring_r{r}_t{t}_{bg_name}", im_r, 0, all_modes))

    # 4. ThickC_60_58_g70, ThickC_70_68_g80, Kidney_80_32, Kidney_80_38 -> 0
    for name, draw_fn in [
        ("ThickC_60_58_g70", lambda im: cv2.ellipse(im, (320, 240), (60, 60), 0, 35, 325, (18, 20, 205), thickness=58)),
        ("ThickC_70_68_g80", lambda im: cv2.ellipse(im, (320, 240), (70, 70), 0, 40, 320, (18, 20, 205), thickness=68)),
        ("Kidney_80_32", lambda im: cv2.ellipse(im, (320, 270), (80, 80), 0, 210, 330, (18, 20, 205), thickness=32)),
        ("Kidney_80_38", lambda im: cv2.ellipse(im, (320, 270), (80, 80), 0, 210, 330, (18, 20, 205), thickness=38)),
    ]:
        im_v = _bg()
        draw_fn(im_v)
        cases.append((name, im_v, 0, all_modes))

    # 5. 5Pent(floor=0.0) on green & white -> 5
    for bg_name, bg_c, mode_list in [
        ("green", (75, 110, 80), all_modes),
        ("white", (245, 245, 245), [("hybrid", 0.65), ("hybrid", 0.15), ("hsv", 0.65)]),
    ]:
        im_5p = _bg(bg_c)
        rad5 = 54.0 / (2.0 * math.sin(math.pi / 5.0))
        for k in range(5):
            th = math.radians(72 * k - 90)
            _draw_shaded(
                im_5p,
                int(round(320 + rad5 * math.cos(th))),
                int(round(240 + rad5 * math.sin(th))),
                32,
                floor=0.0,
            )
        cases.append((f"5Pent_floor0_{bg_name}", im_5p, 5, mode_list))

    # 6. AppleShaded_w245_r45_g10 on white -> 1
    im_sh = _bg((245, 245, 245))
    y0, y1, x0, x1 = 240 - 47, 240 + 48, 320 - 47, 320 + 48
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dx = (xx - 320) / 45.0
    dy = (yy - 240) / 45.0
    d2 = dx * dx + dy * dy
    am = d2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
    shade = np.clip(0.45 + 0.55 * np.clip(-0.25 * dx - 0.25 * dy + 0.90 * nz, 0.25, 1.0), 0.35, 1.0)
    halo = np.exp(-d2 / 0.14)
    patch = im_sh[y0:y1, x0:x1].astype(np.float32)
    for c_i, b in enumerate((18, 20, 205)):
        val = b * shade + halo * (95.0 if c_i < 2 else 50.0)
        patch[..., c_i] = np.where(am, np.clip(val, 0.0, 255.0), patch[..., c_i])
    im_sh[y0:y1, x0:x1] = patch.astype(np.uint8)
    cv2.circle(im_sh, (320, 240), 10, (245, 245, 245), -1)
    cases.append(("AppleShaded_w245_r45_g10", im_sh, 1, all_modes))

    for name, img, expected, mode_list in cases:
        for mode, conf in mode_list:
            res = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=conf)).process_frame(img)
            assert len(res.detections) == expected, (
                f"{name} failed in {mode}@{conf}: got {len(res.detections)}, expected {expected}"
            )


def test_cli_batch_same_stem_collision_and_lazy_video_viewer(tmp_path: Path, monkeypatch) -> None:
    import cv2
    from hybrid_apple_detector.cli import main as cli_main
    from hybrid_apple_detector.visualizer import run_interactive_viewer

    in_dir = tmp_path / "mixed"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sample = np.full((240, 320, 3), (75, 110, 80), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "apple.png"), sample)
    cv2.imwrite(str(in_dir / "apple.jpg"), sample)

    assert cli_main(["--source", str(in_dir), "--output-dir", str(out_dir)]) == 0
    assert (out_dir / "apple_png_result.png").is_file()
    assert (out_dir / "apple_jpg_result.png").is_file()

    # Lazy generator viewer should only consume 1 frame before 'q' quit (never list(frames))
    consumed = 0

    def _lazy_gen():
        nonlocal consumed
        for i in range(10):
            consumed += 1
            yield (f"f_{i}.png", sample)

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _d=0: ord("q"))

    det = HybridAppleDetector(PipelineConfig(mode="hsv"))
    run_interactive_viewer(det, _lazy_gen(), output_dir=out_dir)
    assert consumed == 1




