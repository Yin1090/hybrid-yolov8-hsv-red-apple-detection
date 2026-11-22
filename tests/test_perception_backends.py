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
