"""Public package exports for hybrid_apple_detector."""

from .config import (
    BYPASS_CONF,
    FINAL_CONF_THRESH,
    NMS_IOU_THRESH,
    ROI_HALF,
    YOLO_CONF,
    AdaptiveHSVProfile,
    PipelineConfig,
)
from .fusion import compute_iou, fuse_and_suppress, fuse_score, nms_hybrid_boxes
from .perception import (
    CalibratedProposerBackend,
    ONNXRuntimeBackend,
    PerceptionBackend,
    UltralyticsBackend,
    create_backend,
)
from .pipeline import HybridAppleDetector
from .refinement import AdaptiveHSVCalibrator, detect_in_rect, refine_candidate
from .types import DetectionCandidate, FrameResult, HybridDetection, RefinementResult
from .visualizer import WINDOW_TITLE, render_three_panel

__all__ = [
    "BYPASS_CONF",
    "FINAL_CONF_THRESH",
    "NMS_IOU_THRESH",
    "ROI_HALF",
    "WINDOW_TITLE",
    "YOLO_CONF",
    "AdaptiveHSVCalibrator",
    "AdaptiveHSVProfile",
    "CalibratedProposerBackend",
    "DetectionCandidate",
    "FrameResult",
    "HybridAppleDetector",
    "HybridDetection",
    "ONNXRuntimeBackend",
    "PerceptionBackend",
    "PipelineConfig",
    "RefinementResult",
    "UltralyticsBackend",
    "compute_iou",
    "create_backend",
    "detect_in_rect",
    "fuse_and_suppress",
    "fuse_score",
    "nms_hybrid_boxes",
    "refine_candidate",
    "render_three_panel",
]
