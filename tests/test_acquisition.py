"""Tests for Layer 1 image, directory batch, and video acquisition."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hybrid_apple_detector.acquisition import (
    VideoFrameSource,
    is_video_source,
    iter_image_directory,
    load_image,
    normalize_bgr,
    save_image,
)


def test_normalize_bgr_handles_gray_bgra_and_resize() -> None:
    gray = np.full((120, 240), 128, dtype=np.uint8)
    bgr = normalize_bgr(gray, max_dim=120)
    assert bgr.shape == (60, 120, 3)
    assert bgr.dtype == np.uint8

    bgra = np.full((50, 80, 4), 200, dtype=np.uint8)
    bgr2 = normalize_bgr(bgra)
    assert bgr2.shape == (50, 80, 3)

    with pytest.raises(ValueError):
        normalize_bgr(np.zeros((10, 10, 5), dtype=np.uint8))


def test_load_and_save_unicode_path(tmp_path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (20, 30, 200)
    unicode_path = tmp_path / "táo_đỏ_mẫu.png"
    save_image(unicode_path, img)
    loaded = load_image(unicode_path)
    assert loaded.shape == (64, 64, 3)
    assert np.array_equal(loaded, img)

    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_iter_image_directory_sorted(tmp_path: Path) -> None:
    for name in ("b.png", "a.png", "ignore.txt"):
        p = tmp_path / name
        if name.endswith(".png"):
            save_image(p, np.zeros((32, 32, 3), dtype=np.uint8))
        else:
            p.write_text("notes", encoding="utf-8")

    items = list(iter_image_directory(tmp_path))
    assert [p.name for p, _ in items] == ["a.png", "b.png"]


def test_video_frame_source_reads_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
    assert writer.isOpened(), "OpenCV VideoWriter failed to open MJPG AVI"
    for i in range(4):
        frame = np.full((120, 160, 3), 20 + i * 20, dtype=np.uint8)
        cv2.circle(frame, (80, 60), 22, (20, 22, 205), -1)
        writer.write(frame)
    writer.release()

    assert is_video_source(video_path)
    assert is_video_source("0")
    assert not is_video_source("image.png")

    with VideoFrameSource(video_path) as cap:
        frames = list(cap)
    assert len(frames) == 4
    assert frames[0][1].shape == (120, 160, 3)

    with pytest.raises(RuntimeError):
        VideoFrameSource(tmp_path / "nonexistent_video.avi")
