"""End-to-end 4-layer Coarse-to-Fine Red Apple Detector orchestrator."""

from __future__ import annotations

import time
from typing import Literal

import cv2
import numpy as np

from .acquisition import normalize_bgr
from .config import AdaptiveHSVProfile, PipelineConfig
from .fusion import fuse_and_suppress, nms_hybrid_boxes
from .perception import PerceptionBackend, create_backend
from .refinement import (
    AdaptiveHSVCalibrator,
    detect_in_rect,
    refine_candidate,
    segment_full_frame_hsv,
)
from .types import DetectionCandidate, FrameResult, HybridDetection, RefinementResult


class HybridAppleDetector:
    """Orchestrates Layer 1-4 detection across hybrid, yolo, hsv, and auto modes."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        backend: PerceptionBackend | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.calibrator = AdaptiveHSVCalibrator(self.config.initial_hsv)
        self.backend: PerceptionBackend = backend or create_backend(
            backend=self.config.backend,
            weights_path=self.config.weights_path,
            input_size=self.config.input_size,
        )
        self.active_roi: tuple[int, int, int, int] | None = None
        self._pending_logs: list[str] = []

    @property
    def profile(self) -> AdaptiveHSVProfile:
        return self.calibrator.profile

    def set_mode(self, mode: Literal["hybrid", "yolo", "hsv", "auto"]) -> None:
        normalized = mode.lower().strip()
        if normalized not in ("hybrid", "yolo", "hsv", "auto"):
            raise ValueError(f"Unsupported mode: {mode!r}")
        self.config.mode = normalized  # type: ignore[assignment]

    def click_roi(self, frame_bgr: np.ndarray, cx: int, cy: int) -> list[str]:
        """Calibrate adaptive HSV bounds around (cx, cy) with ROI_HALF radius."""
        frame = normalize_bgr(frame_bgr)
        _, roi_rect, logs = self.calibrator.calibrate_from_point(
            frame,
            cx=cx,
            cy=cy,
            roi_half=self.config.roi_half,
            clahe_clip_limit=self.config.clahe_clip_limit,
            clahe_grid_size=self.config.clahe_grid_size,
        )
        if roi_rect is not None:
            self.active_roi = roi_rect
        return logs

    def select_roi_rect(self, frame_bgr: np.ndarray, rect: tuple[int, int, int, int]) -> list[str]:
        """Calibrate adaptive HSV bounds from an explicit (x1, y1, x2, y2) rectangle."""
        frame = normalize_bgr(frame_bgr)
        _, roi_rect, logs = self.calibrator.calibrate_from_rect(
            frame,
            rect=rect,
            clahe_clip_limit=self.config.clahe_clip_limit,
            clahe_grid_size=self.config.clahe_grid_size,
        )
        if roi_rect is not None:
            self.active_roi = roi_rect
        return logs

    @staticmethod
    def _clamp_roi(
        roi: tuple[int, int, int, int] | None,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int] | None:
        if roi is None:
            return None
        raw_x1, raw_y1, raw_x2, raw_y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if raw_x2 <= raw_x1 or raw_y2 <= raw_y1:
            return None
        rx1 = max(0, min(width, raw_x1))
        ry1 = max(0, min(height, raw_y1))
        rx2 = max(0, min(width, raw_x2))
        ry2 = max(0, min(height, raw_y2))
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        return (rx1, ry1, rx2, ry2)

    def reset_roi(self) -> AdaptiveHSVProfile:
        """Reset active ROI to full frame and restore baseline split-range HSV profile."""
        self.active_roi = None
        return self.calibrator.reset()

    def _auto_calibrate_if_needed(self, frame_bgr: np.ndarray) -> list[str]:
        """Auto-calibrate HSV profile from the full frame when in H:AUTO mode."""
        if self.calibrator.profile.sampled_pixels > 0:
            return []
        h, w = frame_bgr.shape[:2]
        _, _, logs = self.calibrator.calibrate_from_rect(
            frame_bgr,
            (0, 0, w, h),
            clahe_clip_limit=self.config.clahe_clip_limit,
            clahe_grid_size=self.config.clahe_grid_size,
        )
        return logs

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """Run the configured pipeline on a single BGR frame and return FrameResult."""
        t0 = time.perf_counter()
        frame = normalize_bgr(frame_bgr)
        h, w = frame.shape[:2]
        clamped_roi = self._clamp_roi(self.active_roi, h, w)
        mode = self.config.mode
        logs: list[str] = list(self._pending_logs)
        self._pending_logs.clear()

        candidates: list[DetectionCandidate] = []
        refinements: list[RefinementResult] = []
        detections: list[HybridDetection] = []

        if mode == "hybrid":
            candidates = self.backend.detect(frame, self.config.yolo_conf)
            refinements = [
                refine_candidate(frame, cand, self.calibrator.profile, self.config)
                for cand in candidates
            ]
            detections = fuse_and_suppress(refinements, self.config)
            hsv_mask = segment_full_frame_hsv(
                frame, self.calibrator.profile, self.config, active_roi=self.active_roi
            )
            logs.append(f"[YOLO] Found {len(candidates)}")
            logs.append(f"[HYBRID] Apple Count: {len(detections)}")

        elif mode == "yolo":
            candidates = self.backend.detect(frame, self.config.yolo_conf)
            raw_dets = [
                HybridDetection(
                    bbox=c.bbox,
                    score_final=c.yconf,
                    yconf=c.yconf,
                    hsv_conf=0.0,
                    red_ratio=0.0,
                    circularity=0.0,
                    solidity=0.0,
                    extent=0.0,
                    aspect_ratio=0.0,
                    bypassed=True,
                    label=c.label,
                )
                for c in candidates
            ]
            detections = nms_hybrid_boxes(raw_dets, iou_threshold=self.config.nms_iou_thresh)
            hsv_mask = segment_full_frame_hsv(
                frame, self.calibrator.profile, self.config, active_roi=self.active_roi
            )
            logs.append(f"[YOLO] Found {len(candidates)}")
            logs.append(f"[YOLO] Apple Count: {len(detections)}")

        elif mode in ("hsv", "auto"):
            if mode == "auto":
                logs.extend(self._auto_calibrate_if_needed(frame))
            if self.active_roi is not None and clamped_roi is None:
                refinements = []
                hsv_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                refinements, hsv_mask = detect_in_rect(
                    frame,
                    rect=clamped_roi,
                    profile=self.calibrator.profile,
                    config=self.config,
                )
            candidates = [r.candidate for r in refinements]
            raw_dets = [
                HybridDetection(
                    bbox=r.candidate.bbox,
                    score_final=r.hsv_conf,
                    yconf=0.0,
                    hsv_conf=r.hsv_conf,
                    red_ratio=r.red_ratio,
                    circularity=r.circularity,
                    solidity=r.solidity,
                    extent=r.extent,
                    aspect_ratio=r.aspect_ratio,
                    bypassed=False,
                    label="red_apple",
                )
                for r in refinements
                if r.passed and r.hsv_conf >= self.config.final_conf_thresh
            ]
            detections = nms_hybrid_boxes(raw_dets, iou_threshold=self.config.nms_iou_thresh)
            tag = "AUTO" if mode == "auto" else "HSV"
            logs.append(f"[{tag}] Apple Count: {len(detections)}")

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        elapsed_s = max(time.perf_counter() - t0, 1e-6)
        latency_ms = elapsed_s * 1000.0
        fps = 1.0 / elapsed_s

        return FrameResult(
            mode=mode,
            backend=self.backend.name,
            candidates=candidates,
            refinements=refinements,
            detections=detections,
            hsv_mask=hsv_mask,
            fps=fps,
            latency_ms=latency_ms,
            profile=self.calibrator.profile,
            active_roi=clamped_roi,
            logs=logs,
        )
