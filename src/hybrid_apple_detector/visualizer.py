"""3-panel side-by-side visualizer (Original | Mask (HSV/ROI) | Final Result) & optional GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .acquisition import save_image
from .config import AdaptiveHSVProfile
from .pipeline import HybridAppleDetector
from .types import FrameResult

WINDOW_TITLE: str = (
    "HYBRID VIP APPLE | Y:YOLO | H:AUTO | C:HSV | A:HYBRID | S:SAVE | R:RESET"
)

_COLOR_CYAN = (255, 255, 0)
_COLOR_YELLOW = (0, 255, 255)
_COLOR_GREEN = (0, 255, 0)
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)


def _draw_outlined_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _COLOR_BLACK, thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def render_three_panel(
    frame_bgr: np.ndarray,
    result: FrameResult,
    panel_width: int | None = None,
) -> np.ndarray:
    """Render Original | Mask (HSV/ROI) | Final Result horizontally into a single BGR image."""
    h, w = frame_bgr.shape[:2]
    roi = result.active_roi if result.active_roi is not None else (2, 2, w - 3, h - 3)
    rx1, ry1, rx2, ry2 = roi

    # Panel 1: Original + cyan active ROI border
    left = frame_bgr.copy()
    cv2.rectangle(left, (rx1, ry1), (rx2, ry2), _COLOR_CYAN, 2)
    _draw_outlined_text(left, "Original", (12, 28), _COLOR_WHITE, scale=0.72)

    # Panel 2: Mask (HSV/ROI) + yellow candidate/detection boxes
    mask_u8 = result.hsv_mask
    if mask_u8.ndim == 2:
        middle = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    else:
        middle = mask_u8.copy()
    if result.active_roi is not None:
        cv2.rectangle(middle, (rx1, ry1), (rx2, ry2), _COLOR_CYAN, 2)
    boxes_for_mask = (
        [c.bbox for c in result.candidates]
        if result.candidates
        else [d.bbox for d in result.detections]
    )
    for bx1, by1, bx2, by2 in boxes_for_mask:
        cv2.rectangle(middle, (bx1, by1), (bx2, by2), _COLOR_YELLOW, 2)
    _draw_outlined_text(middle, "Mask (HSV/ROI)", (12, 28), _COLOR_WHITE, scale=0.72)

    # Panel 3: Final Result + green boxes + yellow/cyan HUD overlays
    right = frame_bgr.copy()
    for det in result.detections:
        bx1, by1, bx2, by2 = det.bbox
        cv2.rectangle(right, (bx1, by1), (bx2, by2), _COLOR_GREEN, 2)
        label_y = max(18, by1 - 6)
        _draw_outlined_text(
            right,
            f"APPLE {det.score_final:.2f}",
            (bx1, label_y),
            _COLOR_GREEN,
            scale=0.48,
            thickness=1,
        )

    _draw_outlined_text(right, "Final Result", (12, 28), _COLOR_WHITE, scale=0.72)
    hud_line_1 = f"APPLE = {len(result.detections)} | FPS ~ {result.fps:.1f}"
    _draw_outlined_text(right, hud_line_1, (12, 58), _COLOR_YELLOW, scale=0.62)

    profile = result.profile or AdaptiveHSVProfile()
    hud_line_2 = profile.format_hud()
    _draw_outlined_text(right, hud_line_2, (12, 84), _COLOR_CYAN, scale=0.50, thickness=1)

    if panel_width is not None and panel_width > 0 and panel_width != w:
        scale = panel_width / float(w)
        new_h = max(1, int(round(h * scale)))
        left = cv2.resize(left, (panel_width, new_h), interpolation=cv2.INTER_AREA)
        middle = cv2.resize(middle, (panel_width, new_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (panel_width, new_h), interpolation=cv2.INTER_AREA)

    return np.hstack([left, middle, right])


def run_interactive_viewer(
    detector: HybridAppleDetector,
    frames: Iterable[tuple[str, np.ndarray]],
    output_dir: str | Path = "outputs",
) -> None:
    """Launch interactive OpenCV 3-panel window (strictly only when --show is enabled)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_list = list(frames)
    if not frame_list:
        return

    state: dict[str, int] = {"idx": 0}

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            _, current = frame_list[state["idx"]]
            w = current.shape[1]
            local_x = x % w
            logs = detector.click_roi(current, local_x, y)
            for line in logs:
                print(line)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_TITLE, _on_mouse)

    try:
        while True:
            name, frame = frame_list[state["idx"]]
            result = detector.process_frame(frame)
            canvas = render_three_panel(frame, result)
            cv2.imshow(WINDOW_TITLE, canvas)

            delay = 0 if len(frame_list) == 1 else 30
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("y"), ord("Y")):
                detector.set_mode("yolo")
            elif key in (ord("h"), ord("H")):
                detector.set_mode("auto")
            elif key in (ord("c"), ord("C")):
                detector.set_mode("hsv")
            elif key in (ord("a"), ord("A")):
                detector.set_mode("hybrid")
            elif key in (ord("r"), ord("R")):
                detector.reset_roi()
            elif key in (ord("s"), ord("S")):
                save_path = out_dir / f"{Path(name).stem}_snapshot.png"
                save_image(save_path, canvas)
                print(f"[SAVED] {save_path}")
            elif len(frame_list) > 1:
                state["idx"] = (state["idx"] + 1) % len(frame_list)
    finally:
        cv2.destroyAllWindows()
