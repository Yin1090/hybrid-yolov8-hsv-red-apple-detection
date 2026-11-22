"""Tests verifying calibrated results.csv peaks, PR/F1 summary, and benchmark scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_results import summarize_training_results  # noqa: E402
from run_benchmark import run_benchmark  # noqa: E402


def test_results_csv_exact_training_peaks_and_schema() -> None:
    csv_path = REPO_ROOT / "data" / "results.csv"
    summary_json = REPO_ROOT / "data" / "pr_f1_summary.json"
    summary = summarize_training_results(csv_path, summary_json)

    assert summary["epochs"] == 100
    assert len(summary["columns"]) == 14
    assert summary["precision_peak"] == {"value": 0.9442, "epoch": 55}
    assert summary["recall_peak"] == {"value": 0.9103, "epoch": 16}
    assert summary["map50_peak"] == {"value": 0.9512, "epoch": 65}
    assert summary["map50_95_peak"] == {"value": 0.5605, "epoch": 81}

    assert summary["final_losses"]["train_box"] < summary["initial_losses"]["train_box"]
    assert summary["final_losses"]["train_cls"] < summary["initial_losses"]["train_cls"]

    pr = json.loads(summary_json.read_text(encoding="utf-8"))
    assert pr["peak_f1"] == 0.88
    assert pr["peak_f1_conf"] == 0.491
    assert pr["ap_at_50"] == 0.940
    assert pr["recall_at_conf_0"] == 0.96


def test_run_benchmark_reproduces_expected_metrics() -> None:
    records = run_benchmark(REPO_ROOT / "data" / "samples")
    by_key = {(r["scene"], r["config"]): r for r in records}

    assert by_key[("orchard_scene.png", "hybrid_0.65")]["f1"] == 1.0
    assert by_key[("dense_crate_scene.png", "hybrid_0.65")]["candidates"] == 56
    assert by_key[("dense_crate_scene.png", "hybrid_0.65")]["detected"] == 46
    assert by_key[("dense_crate_scene.png", "hybrid_0.65")]["f1"] == 1.0
    assert by_key[("dense_crate_scene.png", "yolo_0.65")]["precision"] < 0.90
    assert by_key[("dense_crate_scene.png", "hybrid_0.85")]["detected"] == 20
    assert by_key[("bagged_apples_scene.png", "hybrid_0.65")]["f1"] == 1.0
    assert by_key[("bagged_apples_scene.png", "hybrid_0.85")]["detected"] == 4
    assert by_key[("bagged_apples_scene.png", "hybrid_no_close")]["detected"] < 10
    assert by_key[("negative_foliage_scene.png", "hybrid_0.65")]["detected"] == 0
