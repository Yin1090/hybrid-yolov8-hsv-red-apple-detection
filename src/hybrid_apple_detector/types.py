"""Structured result types exchanged across the 4-layer detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .config import AdaptiveHSVProfile


@dataclass(frozen=True)
class DetectionCandidate:
    """Layer 2 candidate bounding box and confidence emitted by a perception backend."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    yconf: float
    label: str = "red_apple"
    backend: str = "calibrated"


@dataclass
class RefinementResult:
    """Layer 3 local ROI HSV + morphological + geometric admissibility output."""

    candidate: DetectionCandidate
    passed: bool
    reject_reason: str | None
    red_ratio: float
    circularity: float
    solidity: float
    extent: float
    aspect_ratio: float
    hsv_conf: float
    roi_mask: np.ndarray


@dataclass(frozen=True)
class HybridDetection:
    """Layer 4 fused and NMS-retained final apple detection."""

    bbox: tuple[int, int, int, int]
    score_final: float
    yconf: float
    hsv_conf: float
    red_ratio: float
    circularity: float
    solidity: float
    extent: float
    aspect_ratio: float
    bypassed: bool
    label: str = "red_apple"


@dataclass
class FrameResult:
    """Full pipeline execution record for one frame."""

    mode: str
    backend: str
    candidates: list[DetectionCandidate]
    refinements: list[RefinementResult]
    detections: list[HybridDetection]
    hsv_mask: np.ndarray
    fps: float
    latency_ms: float
    profile: AdaptiveHSVProfile | None = None
    active_roi: tuple[int, int, int, int] | None = None
    logs: list[str] = field(default_factory=list)
