"""Configuration dataclasses and domain constants for the 4-layer red apple detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ROI_HALF: int = 100
YOLO_CONF: float = 0.65
BYPASS_CONF: float = 0.65
FINAL_CONF_THRESH: float = 0.35
NMS_IOU_THRESH: float = 0.55


@dataclass
class AdaptiveHSVProfile:
    """Adaptive split-range HSV profile for red apple segmentation in OpenCV HSV space."""

    low_hue_range: tuple[int, int] = (0, 15)
    high_hue_range: tuple[int, int] = (160, 179)
    s_min: int = 70
    s_max: int = 255
    v_min: int = 50
    v_max: int = 255
    s_p20: int = 70
    s_p80: int = 230
    v_p20: int = 50
    v_p80: int = 220
    sampled_pixels: int = 0

    def format_hud(self) -> str:
        """Format HSV range string for the Final Result HUD overlay."""
        return (
            f"HSV: H=[{self.low_hue_range}, {self.high_hue_range}] "
            f"S>{self.s_min} V>{self.v_min}"
        )

    def format_calc_log(self) -> str:
        """Format console calibration log line."""
        return (
            f"[HSV CALC] H=[{self.low_hue_range}, {self.high_hue_range}] "
            f"S={self.s_min} V={self.v_min}"
        )


@dataclass
class PipelineConfig:
    """End-to-end hyperparameters for the 4-layer coarse-to-fine pipeline."""

    mode: Literal["hybrid", "yolo", "hsv", "auto"] = "hybrid"
    backend: Literal["calibrated", "ultralytics", "onnx", "auto"] = "auto"
    weights_path: str | None = None

    # Layer 2 (Global perception / candidate proposal)
    yolo_conf: float = YOLO_CONF
    input_size: int = 640

    # Layer 3 (Local ROI refinement: Gaussian -> CLAHE-V -> Split-Range HSV -> Morphology -> Geometry)
    roi_half: int = ROI_HALF
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    gaussian_ksize: int = 5
    median_ksize: int = 5
    morph_ksize: int = 7
    open_iter: int = 1
    close_iter: int = 3
    min_red_ratio: float = 0.08
    min_circularity: float = 0.25
    min_solidity: float = 0.60
    min_extent: float = 0.40
    aspect_ratio_range: tuple[float, float] = (0.5, 2.0)
    min_contour_area: int = 120

    # Layer 4 (Fusion & secondary NMS)
    bypass_conf: float = BYPASS_CONF
    strict_bypass: bool = False
    final_conf_thresh: float = FINAL_CONF_THRESH
    nms_iou_thresh: float = NMS_IOU_THRESH
    yolo_weight: float = 0.45
    hsv_weight: float = 0.55

    # Initial HSV profile
    initial_hsv: AdaptiveHSVProfile = field(default_factory=AdaptiveHSVProfile)
