"""Analyze YOLOv8n training results.csv and PR/F1 summary without pandas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_results_csv(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {k.strip(): float(v) for k, v in row.items() if k is not None}
            rows.append(parsed)
    return rows


def summarize_training_results(
    csv_path: Path,
    summary_json_path: Path | None = None,
) -> dict:
    rows = load_results_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    def _peak(col: str) -> dict[str, float | int]:
        best = max(rows, key=lambda r: r[col])
        return {"value": round(best[col], 4), "epoch": int(round(best["epoch"]))}

    summary = {
        "epochs": len(rows),
        "columns": list(rows[0].keys()),
        "precision_peak": _peak("metrics/precision(B)"),
        "recall_peak": _peak("metrics/recall(B)"),
        "map50_peak": _peak("metrics/mAP50(B)"),
        "map50_95_peak": _peak("metrics/mAP50-95(B)"),
        "initial_losses": {
            "train_box": round(rows[0]["train/box_loss"], 4),
            "train_cls": round(rows[0]["train/cls_loss"], 4),
            "train_dfl": round(rows[0]["train/dfl_loss"], 4),
        },
        "final_losses": {
            "train_box": round(rows[-1]["train/box_loss"], 4),
            "train_cls": round(rows[-1]["train/cls_loss"], 4),
            "train_dfl": round(rows[-1]["train/dfl_loss"], 4),
            "val_box": round(rows[-1]["val/box_loss"], 4),
            "val_cls": round(rows[-1]["val/cls_loss"], 4),
            "val_dfl": round(rows[-1]["val/dfl_loss"], 4),
        },
    }

    if summary_json_path and summary_json_path.exists():
        pr_f1 = json.loads(summary_json_path.read_text(encoding="utf-8"))
        summary["operating_points"] = {
            "peak_f1": pr_f1["peak_f1"],
            "peak_f1_conf": pr_f1["peak_f1_conf"],
            "ap_at_50": pr_f1["ap_at_50"],
            "recall_at_conf_0": pr_f1["recall_at_conf_0"],
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze YOLOv8n training results.csv.")
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data" / "results.csv"),
        help="Path to results.csv",
    )
    parser.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "data" / "pr_f1_summary.json"),
        help="Path to pr_f1_summary.json",
    )
    args = parser.parse_args()

    summary = summarize_training_results(Path(args.csv), Path(args.summary_json))
    print(f"Epochs: {summary['epochs']} | Columns: {len(summary['columns'])}")
    print(
        f"Peak Precision: {summary['precision_peak']['value']:.4f} @ epoch {summary['precision_peak']['epoch']}"
    )
    print(
        f"Peak Recall:    {summary['recall_peak']['value']:.4f} @ epoch {summary['recall_peak']['epoch']}"
    )
    print(
        f"Peak mAP@0.5:   {summary['map50_peak']['value']:.4f} @ epoch {summary['map50_peak']['epoch']}"
    )
    print(
        f"Peak mAP@50-95: {summary['map50_95_peak']['value']:.4f} @ epoch {summary['map50_95_peak']['epoch']}"
    )
    if "operating_points" in summary:
        op = summary["operating_points"]
        print(
            f"Operating Pts:  F1={op['peak_f1']:.2f} @ conf={op['peak_f1_conf']:.3f} | "
            f"AP@0.5={op['ap_at_50']:.3f} | Recall@0={op['recall_at_conf_0']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
