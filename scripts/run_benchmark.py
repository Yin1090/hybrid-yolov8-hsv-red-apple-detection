"""Run offline benchmark & ablation comparisons across synthetic data/samples/ scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_apple_detector import HybridAppleDetector, PipelineConfig, compute_iou  # noqa: E402
from hybrid_apple_detector.acquisition import load_image  # noqa: E402


def match_boxes_to_gt(
    pred_boxes: list[tuple[int, int, int, int]],
    gt_boxes: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.50,
) -> tuple[int, int, int]:
    """Greedy 1-to-1 IoU matching returning (tp, fp, fn)."""
    matched_gt: set[int] = set()
    tp = 0
    for pbox in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gi, gbox in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_idx >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def run_benchmark(samples_dir: Path) -> list[dict]:
    gt_path = samples_dir / "ground_truth.json"
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))

    configs = [
        ("hybrid_0.15", PipelineConfig(mode="hybrid", yolo_conf=0.15)),
        ("hybrid_0.65", PipelineConfig(mode="hybrid", yolo_conf=0.65)),
        ("hybrid_0.85", PipelineConfig(mode="hybrid", yolo_conf=0.85)),
        ("yolo_0.65", PipelineConfig(mode="yolo", yolo_conf=0.65)),
        ("hsv_baseline", PipelineConfig(mode="hsv")),
        ("hybrid_no_close", PipelineConfig(mode="hybrid", yolo_conf=0.65, close_iter=0)),
    ]

    records: list[dict] = []
    for filename, meta in gt_data.items():
        img = load_image(samples_dir / filename)
        gt_boxes = [tuple(int(v) for v in b) for b in meta["boxes_xyxy"]]
        for label, cfg in configs:
            if label == "hybrid_no_close" and "bagged" not in filename:
                continue
            detector = HybridAppleDetector(config=cfg)
            res = detector.process_frame(img)
            pred_boxes = [d.bbox for d in res.detections]
            tp, fp, fn = match_boxes_to_gt(pred_boxes, gt_boxes)
            prec = tp / float(tp + fp) if (tp + fp) > 0 else (1.0 if len(gt_boxes) == 0 else 0.0)
            rec = tp / float(tp + fn) if (tp + fn) > 0 else (1.0 if len(pred_boxes) == 0 else 0.0)
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            records.append(
                {
                    "scene": filename,
                    "config": label,
                    "gt": len(gt_boxes),
                    "candidates": len(res.candidates),
                    "detected": len(res.detections),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "fps": round(res.fps, 1),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark ablations on data/samples/.")
    parser.add_argument(
        "--samples-dir",
        default=str(REPO_ROOT / "data" / "samples"),
        help="Path to samples directory containing PNGs and ground_truth.json",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table.")
    args = parser.parse_args()

    records = run_benchmark(Path(args.samples_dir))
    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    header = (
        f"{'Scene':<28} {'Config':<16} {'GT':>3} {'Cand':>4} {'Det':>4} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'FPS':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['scene']:<28} {r['config']:<16} {r['gt']:>3d} {r['candidates']:>4d} "
            f"{r['detected']:>4d} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} {r['fps']:>7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
