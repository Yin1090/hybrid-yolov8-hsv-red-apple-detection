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

