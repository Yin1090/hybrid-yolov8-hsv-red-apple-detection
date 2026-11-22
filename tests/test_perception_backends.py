"""Tests for Layer 2 perception backends, ONNX decode, and non-cheating invariance."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hybrid_apple_detector import HybridAppleDetector, PipelineConfig
from hybrid_apple_detector.acquisition import load_image, save_image
from hybrid_apple_detector.perception import (
    CalibratedProposerBackend,
    ONNXRuntimeBackend,
    UltralyticsBackend,
    create_backend,
    decode_yolov8_output,
    letterbox_bgr,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


def test_create_backend_routing_and_errors() -> None:
    b = create_backend("auto")
    assert isinstance(b, CalibratedProposerBackend)

    with pytest.raises(ValueError):
        create_backend("onnx", weights_path=None)

    with pytest.raises(ValueError):
        create_backend("unknown_backend")


def test_onnx_letterbox_and_decode_yolov8_tensor() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blob, scale, pad_w, pad_h = letterbox_bgr(frame, new_shape=(640, 640))
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert scale == pytest.approx(1.0)
    assert pad_w == pytest.approx(0.0)
    assert pad_h == pytest.approx(80.0)

    # Mock YOLOv8 [1, 4+C, N] output tensor with 2 boxes (one above thresh, one duplicate overlap)
    raw = np.zeros((1, 5, 4), dtype=np.float32)
    # Box 0: center (320, 320 in letterbox -> y=240 in 480x640), size 80x80, conf=0.88
    raw[0, :, 0] = [320.0, 320.0, 80.0, 80.0, 0.88]
    # Box 1: heavily overlapping duplicate with lower conf=0.72 -> suppressed by NMS
    raw[0, :, 1] = [322.0, 321.0, 78.0, 78.0, 0.72]
    # Box 2: low conf=0.10 -> filtered by conf_thresh=0.50
    raw[0, :, 2] = [100.0, 100.0, 40.0, 40.0, 0.10]

    cands = decode_yolov8_output(
        raw,
        conf_thresh=0.50,
        scale=scale,
        pad_w=pad_w,
        pad_h=pad_h,
        orig_hw=(480, 640),
    )
    assert len(cands) == 1
    assert cands[0].bbox == (280, 200, 360, 280)
    assert cands[0].yconf == pytest.approx(0.88, abs=1e-4)

    # Also test [1, N, 4+C] transposed layout via ONNXRuntimeBackend with mocked session
    class _FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def run(self, _outputs, feed):
            assert feed["images"].shape == (1, 3, 640, 640)
            return [np.transpose(raw, (0, 2, 1))]

    backend = ONNXRuntimeBackend("dummy.onnx", session=_FakeSession())
    dets = backend.detect(frame, conf_thresh=0.50)
    assert len(dets) == 1
    assert dets[0].backend == "onnx"
    assert dets[0].bbox == (280, 200, 360, 280)


def test_ultralytics_backend_with_mock_model() -> None:
    class _FakeBoxes:
        xyxy = np.array([[50.0, 60.0, 120.0, 130.0]], dtype=np.float32)
        conf = np.array([0.91], dtype=np.float32)
        cls = np.array([0], dtype=np.int64)

        def __len__(self) -> int:
            return 1

    class _FakeModel:
        def predict(self, frame, conf=0.65, imgsz=640, verbose=False):
            return [SimpleNamespace(boxes=_FakeBoxes(), names={0: "red_apple"})]

    backend = UltralyticsBackend("dummy.pt", model=_FakeModel())
    cands = backend.detect(np.zeros((200, 200, 3), dtype=np.uint8), conf_thresh=0.65)
    assert len(cands) == 1
    assert cands[0].bbox == (50, 60, 120, 130)
    assert cands[0].yconf == pytest.approx(0.91, abs=1e-4)
    assert cands[0].backend == "ultralytics"


def test_calibrated_proposer_filename_flip_pad_invariance(tmp_path: Path) -> None:
    """Prove zero hardcoding: renaming, flipping, and padding preserve detection counts."""
    for sample_name, expected in (
        ("orchard_scene.png", 12),
        ("dense_crate_scene.png", 46),
        ("bagged_apples_scene.png", 16),
    ):
        img = load_image(SAMPLES_DIR / sample_name)
        renamed = tmp_path / f"{uuid.uuid4().hex}.png"
        save_image(renamed, img)
        loaded_renamed = load_image(renamed)

        det = HybridAppleDetector(PipelineConfig(mode="hybrid", yolo_conf=0.65))
        res_orig = det.process_frame(img)
        res_ren = det.process_frame(loaded_renamed)
        assert len(res_orig.detections) == expected
        assert len(res_ren.detections) == expected

        flipped = cv2.flip(img, 1)
        padded = cv2.copyMakeBorder(flipped, 24, 24, 32, 32, cv2.BORDER_CONSTANT, value=(90, 90, 90))
        res_aug = det.process_frame(padded)
        assert len(res_aug.detections) == expected


def test_source_package_never_references_ground_truth() -> None:
    pkg_dir = REPO_ROOT / "src" / "hybrid_apple_detector"
    for py_file in pkg_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "ground_truth" not in text


def test_decode_yolov8_output_validates_channels_class_id_and_small_n() -> None:
    # < 5 channels (e.g. [1, 4, 8400]) must raise ValueError
    bad_4ch = np.zeros((1, 4, 8400), dtype=np.float32)
    bad_4ch[0, :, 0] = [320.0, 320.0, 80.0, 80.0]
    with pytest.raises(ValueError, match="at least 5 channels"):
        decode_yolov8_output(bad_4ch, 0.5, 1.0, 0.0, 0.0, (640, 640))

    # Negative target_class_id must raise ValueError
    valid_5ch = np.zeros((1, 5, 10), dtype=np.float32)
    valid_5ch[0, :, 0] = [320.0, 320.0, 80.0, 80.0, 0.9]
    with pytest.raises(ValueError, match="target_class_id=-1"):
        decode_yolov8_output(valid_5ch, 0.5, 1.0, 0.0, 0.0, (640, 640), target_class_id=-1)

    # Small-N multi-class [1, 84, 50] (N=50 < 84) and [1, 50, 84] must decode accurately
    small_n_cf = np.zeros((1, 84, 50), dtype=np.float32)
    small_n_cf[0, :4, 7] = [320.0, 320.0, 100.0, 100.0]
    small_n_cf[0, 4 + 47, 7] = 0.91
    cands_cf = decode_yolov8_output(small_n_cf, 0.5, 1.0, 0.0, 0.0, (640, 640), target_class_id=47)
    assert len(cands_cf) == 1
    assert cands_cf[0].bbox == (270, 270, 370, 370)
    assert cands_cf[0].label == "class_47"

    small_n_cl = np.transpose(small_n_cf, (0, 2, 1))
    cands_cl = decode_yolov8_output(small_n_cl, 0.5, 1.0, 0.0, 0.0, (640, 640), target_class_id=47)
    assert len(cands_cl) == 1
    assert cands_cl[0].bbox == (270, 270, 370, 370)


def test_calibrated_proposer_avoids_ribbon_shattering_splits_touching_and_recalls_dark_apples() -> None:
    # Concave L-ribbon & C-ribbon (standard & thicker variants) -> 0 FPs, touching apples -> 2 detections
    scene = np.full((520, 1000, 3), (75, 110, 80), dtype=np.uint8)
    cv2.line(scene, (90, 90), (90, 200), (18, 20, 205), 22)
    cv2.line(scene, (90, 200), (200, 200), (18, 20, 205), 22)
    cv2.ellipse(scene, (360, 170), (50, 50), 0, 35, 325, (18, 20, 205), 18)
    cv2.line(scene, (540, 70), (540, 200), (18, 20, 205), 28)
    cv2.line(scene, (540, 200), (670, 200), (18, 20, 205), 28)
    cv2.ellipse(scene, (820, 170), (60, 60), 0, 35, 325, (18, 20, 205), 24)
    pts = cv2.boxPoints(((220, 380), (85, 35), 45.0)).astype(np.int32)
    cv2.fillPoly(scene, [pts], (18, 20, 205))

    for mode in ("hybrid", "hsv", "auto"):
        res_dist = HybridAppleDetector(PipelineConfig(mode=mode, yolo_conf=0.65)).process_frame(scene)
        assert len(res_dist.detections) == 0

    touching = np.full((220, 300, 3), (45, 110, 45), dtype=np.uint8)
    cv2.circle(touching, (110, 110), 32, (18, 20, 205), -1)
    cv2.circle(touching, (166, 110), 32, (18, 20, 205), -1)
    res_touch = HybridAppleDetector(PipelineConfig(mode="hybrid", yolo_conf=0.65)).process_frame(touching)
    assert len(res_touch.detections) == 2

    # Dark-shadow red apples (V=40 and V=70) + dim neutral gray noise circle
    for target_v in (40, 70):
        dark = np.full((260, 520, 3), (40, 40, 40), dtype=np.uint8)
        for cx in (80, 170, 260, 350, 440):
            r = 30
            yy, xx = np.ogrid[130 - r : 130 + r + 1, cx - r : cx + r + 1]
            dist2 = ((xx - cx) / r) ** 2 + ((yy - 130) / r) ** 2
            mask = dist2 <= 1.0
            nz = np.sqrt(np.clip(1.0 - dist2, 0.0, 1.0))
            ndotl = np.clip(-0.35 * (xx - cx) / r - 0.35 * (yy - 130) / r + 0.85 * nz, 0.45, 1.0)
            patch = dark[130 - r : 130 + r + 1, cx - r : cx + r + 1].astype(np.float32)
            for ch, val in enumerate((4.0, 4.0, float(target_v))):
                patch[..., ch] = np.where(mask, np.clip(val * ndotl, 0, 255), patch[..., ch])
            dark[130 - r : 130 + r + 1, cx - r : cx + r + 1] = patch.astype(np.uint8)
        cv2.circle(dark, (260, 215), 25, (35, 35, 45), -1)
        res_dark = HybridAppleDetector(PipelineConfig(mode="hybrid", yolo_conf=0.65)).process_frame(dark)
        assert len(res_dark.detections) == 5
