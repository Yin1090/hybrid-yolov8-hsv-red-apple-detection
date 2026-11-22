"""Layer 1: Image, directory batch, and video/webcam acquisition and BGR normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")


def normalize_bgr(frame: np.ndarray, max_dim: int | None = None) -> np.ndarray:
    """Ensure a contiguous 3-channel uint8 BGR frame with optional proportional downscaling."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Input frame is empty or invalid.")

    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        bgr = frame
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    if max_dim is not None and max_dim > 0:
        h, w = bgr.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(bgr)


def load_image(path: str | Path, max_dim: int | None = None) -> np.ndarray:
    """Load an image from disk with Unicode-safe fallback on Windows."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if img is None:
        raw = np.fromfile(str(file_path), dtype=np.uint8)
        if raw.size > 0:
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image file: {file_path}")

    return normalize_bgr(img, max_dim=max_dim)


def save_image(path: str | Path, frame_bgr: np.ndarray) -> Path:
    """Save a BGR image to disk with Unicode-safe encoding."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix or ".png"
    ok, buf = cv2.imencode(ext, frame_bgr)
    if not ok:
        raise ValueError(f"Failed to encode image for path: {out_path}")
    buf.tofile(str(out_path))
    return out_path


def iter_image_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    max_dim: int | None = None,
) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield (path, normalized_bgr) for all sorted image files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    ext_set = {e.lower() for e in extensions}
    files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in ext_set
    )
    for img_path in files:
        yield img_path, load_image(img_path, max_dim=max_dim)


class VideoFrameSource:
    """Context-managed video file or webcam stream reader built on cv2.VideoCapture."""

    def __init__(self, source: str | Path | int, max_dim: int | None = None) -> None:
        self.source = int(source) if isinstance(source, int) else str(source)
        self.max_dim = max_dim
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Unable to open video/camera source: {self.source}")

    def is_opened(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, normalize_bgr(frame, max_dim=self.max_dim)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while self.is_opened():
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "VideoFrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def is_video_source(source: str | Path | int) -> bool:
    """Return True if source is an integer camera index or a video file path."""
    if isinstance(source, int):
        return True
    s = str(source).strip()
    if s.isdigit():
        return True
    return Path(s).suffix.lower() in VIDEO_EXTENSIONS
