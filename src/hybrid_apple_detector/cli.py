"""Command-line interface for Hybrid YOLOv8n + Adaptive HSV Red Apple Detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .acquisition import (
    VideoFrameSource,
    is_video_source,
    iter_image_directory,
    load_image,
    save_image,
)
from .config import BYPASS_CONF, YOLO_CONF, PipelineConfig
from .pipeline import HybridAppleDetector
from .types import FrameResult
from .visualizer import render_three_panel, run_interactive_viewer


def _parse_roi_click(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--roi-click must be 'x,y', got {value!r}")
    return int(parts[0]), int(parts[1])


def _result_to_dict(source_name: str, result: FrameResult) -> dict:
    return {
        "source": source_name,
        "mode": result.mode,
        "backend": result.backend,
        "yolo_candidates": len(result.candidates),
        "apple_count": len(result.detections),
        "fps": round(result.fps, 2),
        "latency_ms": round(result.latency_ms, 3),
        "hsv_profile": {
            "low_hue_range": list(result.profile.low_hue_range) if result.profile else [0, 15],
            "high_hue_range": list(result.profile.high_hue_range) if result.profile else [160, 179],
            "s_min": result.profile.s_min if result.profile else 70,
            "v_min": result.profile.v_min if result.profile else 50,
        },
        "detections": [
            {
                "bbox": list(d.bbox),
                "score_final": round(d.score_final, 4),
                "yconf": round(d.yconf, 4),
                "hsv_conf": round(d.hsv_conf, 4),
                "red_ratio": round(d.red_ratio, 4),
                "circularity": round(d.circularity, 4),
                "solidity": round(d.solidity, 4),
                "extent": round(d.extent, 4),
                "aspect_ratio": round(d.aspect_ratio, 4),
                "bypassed": d.bypassed,
                "label": d.label,
            }
            for d in result.detections
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="red-apple-detect",
        description="Coarse-to-Fine YOLOv8n + Adaptive Split-Range HSV Red Apple Detector",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to input image, directory of images, video file, or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "yolo", "hsv", "auto"),
        default="hybrid",
        help="Detection pipeline mode (default: hybrid).",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=YOLO_CONF,
        help=f"Layer 2 candidate confidence threshold (default: {YOLO_CONF}).",
    )
    parser.add_argument(
        "--bypass-conf",
        type=float,
        default=BYPASS_CONF,
        help=f"Layer 4 bypass confidence threshold (default: {BYPASS_CONF}).",
    )
    parser.add_argument(
        "--strict-bypass",
        action="store_true",
        help="Allow yconf > bypass_conf to skip Layer 3 physical rejection (ablation mode).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional path to YOLOv8 .pt or .onnx weights file.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "calibrated", "ultralytics", "onnx"),
        default="auto",
        help="Layer 2 perception backend (default: auto).",
    )
    parser.add_argument(
        "--roi-click",
        default=None,
        help="Optional 'x,y' coordinate for interactive-equivalent ROI_HALF=100 HSV calibration.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for rendered 3-panel PNG outputs and JSON reports (default: outputs).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save structured detection metrics as JSON files in --output-dir.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive OpenCV 3-panel viewer window (disabled by default for headless runs).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = PipelineConfig(
        mode=args.mode,
        backend=args.backend,
        weights_path=args.weights,
        yolo_conf=args.yolo_conf,
        bypass_conf=args.bypass_conf,
        strict_bypass=args.strict_bypass,
    )
    detector = HybridAppleDetector(config=cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_pt = _parse_roi_click(args.roi_click)

    source_str = str(args.source)
    source_path = Path(source_str)
    all_summaries: list[dict] = []

    if is_video_source(source_str):
        cam_or_path: str | int = int(source_str) if source_str.isdigit() else source_str
        with VideoFrameSource(cam_or_path) as cap:
            frames = list(cap)
        if args.show:
            run_interactive_viewer(
                detector,
                [(f"frame_{idx:05d}.png", frm) for idx, frm in frames],
                output_dir=out_dir,
            )
            return 0
        for idx, frm in frames:
            if idx == 0 and roi_pt is not None:
                for line in detector.click_roi(frm, roi_pt[0], roi_pt[1]):
                    print(line)
            res = detector.process_frame(frm)
            for line in res.logs:
                print(line)
            panel = render_three_panel(frm, res)
            save_image(out_dir / f"frame_{idx:05d}_result.png", panel)
            all_summaries.append(_result_to_dict(f"frame_{idx:05d}", res))
    elif source_path.is_dir():
        items = list(iter_image_directory(source_path))
        if not items:
            raise FileNotFoundError(f"No image files found in directory: {source_path}")
        if args.show:
            run_interactive_viewer(
                detector,
                [(p.name, img) for p, img in items],
                output_dir=out_dir,
            )
            return 0
        for idx, (img_path, img) in enumerate(items):
            if idx == 0 and roi_pt is not None:
                for line in detector.click_roi(img, roi_pt[0], roi_pt[1]):
                    print(line)
            res = detector.process_frame(img)
            for line in res.logs:
                print(line)
            panel = render_three_panel(img, res)
            save_image(out_dir / f"{img_path.stem}_result.png", panel)
            all_summaries.append(_result_to_dict(img_path.name, res))
    else:
        img = load_image(source_path)
        if roi_pt is not None:
            for line in detector.click_roi(img, roi_pt[0], roi_pt[1]):
                print(line)
        if args.show:
            run_interactive_viewer(
                detector,
                [(source_path.name, img)],
                output_dir=out_dir,
            )
            return 0
        res = detector.process_frame(img)
        for line in res.logs:
            print(line)
        panel = render_three_panel(img, res)
        save_image(out_dir / f"{source_path.stem}_result.png", panel)
        all_summaries.append(_result_to_dict(source_path.name, res))

    if args.save_json:
        json_path = out_dir / "detections.json"
        json_path.write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
