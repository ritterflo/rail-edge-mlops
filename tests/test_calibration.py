"""Tests for confidence calibration.

The metric this replaces was dominated by threshold placement rather than calibration:
three identical runs produced 66,707 / 68,452 / 113,386 detections above a fixed 0.05
score, and ECE varied by 0.135 as a result. Top-K per image makes the measured
population identical across runs, and these tests pin that property.
"""

from __future__ import annotations

import torch

from rail_edge_mlops.evaluate import calibration


def _image(scores: list[float], hit: list[bool]):
    """One image's predictions plus ground truth, where `hit[i]` decides whether
    detection i overlaps a real box."""
    boxes, gt_boxes, gt_labels = [], [], []
    for i, h in enumerate(hit):
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


def test_top_k_bounds_the_measured_population():
    """20 detections measured from 100, regardless of where the scores sit."""
    scores = [0.9 - i * 0.005 for i in range(100)]
    pred, gt = _image(scores, [True] * 100)
    result = calibration([pred], [gt], top_k=20)
    assert result["n_detections"] == 20
    assert result["n_detections_above_threshold"] == 100


def test_identical_ranking_at_different_scales_measures_the_same_population():
    """The failure the old metric had: shifting every score changed how many
    detections were counted, and therefore the ECE."""
    hit = [True] * 40
    high, _ = _image([0.9 - i * 0.01 for i in range(40)], hit)
    low, gt = _image([0.2 - i * 0.001 for i in range(40)], hit)
    a = calibration([high], [gt], top_k=20)
    b = calibration([low], [gt], top_k=20)
    assert a["n_detections"] == b["n_detections"] == 20


def test_perfect_calibration_scores_near_zero():
    """Half the detections at 0.5 confidence, half of them correct."""
    scores = [0.55] * 20
    pred, gt = _image(scores, [True] * 10 + [False] * 10)
    result = calibration([pred], [gt], top_k=20)
    assert result["ece"] < 0.10


def test_underconfidence_shows_as_negative_gaps():
    """What the baseline actually exhibits: low scores, high hit rate."""
    pred, gt = _image([0.05] * 20, [True] * 20)
    result = calibration([pred], [gt], top_k=20)
    assert all(b["gap"] < 0 for b in result["bins"])
    assert result["ece"] > 0.5


def test_overconfidence_shows_as_positive_gaps():
    pred, gt = _image([0.95] * 20, [False] * 20)
    result = calibration([pred], [gt], top_k=20)
    assert all(b["gap"] > 0 for b in result["bins"])


def test_score_percentiles_expose_compression():
    """A distribution squashed near zero should be visible as a number, not inferred."""
    pred, gt = _image([0.02 + i * 0.001 for i in range(40)], [True] * 40)
    result = calibration([pred], [gt], top_k=20)
    assert result["score_percentiles"]["p99"] < 0.1
    assert result["score_percentiles"]["max"] < 0.1


def test_no_detections_is_not_a_crash():
    pred = {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }
    gt = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
    result = calibration([pred], [gt], top_k=20)
    assert result["n_detections"] == 0 and result["ece"] == 0.0
