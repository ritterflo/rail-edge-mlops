"""Tests for the operating-point table and the calibration bins.

Two earlier versions of this measurement let the population depend on the model's own
score distribution -- first via a fixed score threshold, then via a fixed top-20. Both
made ECE move for reasons unrelated to calibration. The population is now the image's
ground-truth count, which comes from the data. These tests pin that.
"""

from __future__ import annotations

import torch

from rail_edge_mlops.evaluate import calibration, operating_points, run_metrics


def _sample(scores: list[float], hits: list[bool]):
    """Build one image. `hits[i]` decides whether detection i sits on a real box."""
    boxes, gt_boxes, gt_labels = [], [], []
    for i, h in enumerate(hits):
        x = i * 100.0
        boxes.append([x, 0.0, x + 50, 50.0])
        if h:
            gt_boxes.append([x, 0.0, x + 50, 50.0])
            gt_labels.append(0)
    pred = {
        "boxes": torch.tensor(boxes),
        "scores": torch.tensor(scores),
        "labels": torch.zeros(len(scores), dtype=torch.long),
    }
    gt = {
        "boxes": torch.tensor(gt_boxes) if gt_boxes else torch.zeros(0, 4),
        "labels": torch.tensor(gt_labels, dtype=torch.long),
    }
    return pred, gt


# --- population is set by the data, not by the scores ----------------------------


def test_population_equals_ground_truth_count():
    pred, gt = _sample([0.9 - i * 0.01 for i in range(50)], [True] * 8 + [False] * 42)
    assert calibration([pred], [gt])["n_measured"] == 8


def test_rescaling_every_score_does_not_change_the_population():
    """The failure both earlier versions had: shift the distribution, change the count."""
    hits = [True] * 6 + [False] * 24
    high, gt = _sample([0.9 - i * 0.01 for i in range(30)], hits)
    low, _ = _sample([0.09 - i * 0.001 for i in range(30)], hits)
    assert calibration([high], [gt])["n_measured"] == calibration([low], [gt])["n_measured"] == 6


# --- the bin table reports what it should ----------------------------------------


def test_underconfidence_shows_as_negative_gaps():
    """What the baseline exhibits: low scores, high hit rate."""
    pred, gt = _sample([0.05] * 10, [True] * 10)
    result = calibration([pred], [gt])
    assert all(b["gap"] < 0 for b in result["bins"])


def test_overconfidence_shows_as_positive_gaps():
    """High scores on the wrong detections, so the measured top-K are misses."""
    pred, gt = _sample([0.95, 0.95, 0.3, 0.3], [False, False, True, True])
    result = calibration([pred], [gt])
    assert all(b["gap"] > 0 for b in result["bins"])


def test_score_percentiles_expose_compression():
    pred, gt = _sample([0.02 + i * 0.001 for i in range(40)], [True] * 5 + [False] * 35)
    pct = calibration([pred], [gt])["score_percentiles"]
    assert pct["p99"] < 0.1 and pct["max"] < 0.1


def test_no_detections_is_not_a_crash():
    pred = {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }
    gt = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
    result = calibration([pred], [gt])
    assert result["n_measured"] == 0


# --- the operating-point table ----------------------------------------------------


def test_precision_rises_and_recall_falls_with_threshold():
    """The trade-off the table exists to expose. Correct detections score high."""
    scores = [0.9, 0.8, 0.7, 0.05, 0.04, 0.03, 0.02, 0.01]
    pred, gt = _sample(scores, [True] * 3 + [False] * 5)
    rows = operating_points([pred], [gt], thresholds=(0.01, 0.1))
    loose, tight = rows[0], rows[1]
    assert tight["precision"] > loose["precision"]
    assert tight["recall"] <= loose["recall"]
    assert tight["detections_per_image"] < loose["detections_per_image"]


def test_a_threshold_above_every_score_yields_nothing():
    """The situation the real model is in: p99 is 0.18, so 0.5 returns nothing."""
    pred, gt = _sample([0.1, 0.08, 0.05], [True, False, False])
    row = operating_points([pred], [gt], thresholds=(0.5,))[0]
    assert row["detections_per_image"] == 0.0
    assert row["precision"] == 0.0 and row["recall"] == 0.0


def test_recall_is_bounded_by_ground_truth_not_detections():
    pred, gt = _sample([0.9] * 2, [True] * 2)
    row = operating_points([pred], [gt], thresholds=(0.01,))[0]
    assert row["recall"] == 1.0 and row["precision"] == 1.0


# --- what gets logged must be what evaluate() returns -----------------------------


def test_run_metrics_takes_its_keys_from_evaluate():
    """Regression: the calibration headline was renamed and the logging call kept the old
    key, so every run raised KeyError after the metrics and before the model artifacts --
    the failure was invisible until a model turned out not to be in the registry."""
    result = {
        "map": 0.2,
        "map_50": 0.3,
        "ap_per_class": {"car": 0.4},
        "calibration": calibration([], []),
        "operating_points": [],
    }
    flat = run_metrics("val", result)
    assert flat["val.map"] == 0.2
    assert flat["val.map_50"] == 0.3
    assert flat["val.ece_same_predictions_only"] == 0.0
    assert "val.ap_per_class" not in flat
