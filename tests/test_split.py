"""Tests for the split policy.

These run against synthetic provenance rather than the real dataset, so they stay hermetic:
CI must not need `dvc pull` to tell you whether the split logic is sound.
"""

from __future__ import annotations

import pytest

from rail_edge_mlops.data.categories import CLASS_TO_ID, DETECTION_CLASSES, NUIMAGES_TO_DETECTION
from rail_edge_mlops.data.split import build_split


def _provenance(n_logs: int = 10, imgs_per_log: int = 5, test_location: str = "boston-seaport"):
    """Half the logs in the test location, half elsewhere."""
    prov, image_id = {}, 1
    for log_index in range(n_logs):
        loc = test_location if log_index % 2 == 0 else "singapore-onenorth"
        for _ in range(imgs_per_log):
            prov[str(image_id)] = {"log_token": f"log{log_index:02d}", "location": loc}
            image_id += 1
    return prov


def test_no_log_spans_two_splits():
    """The leakage invariant: frames from one drive must never straddle a split."""
    splits, _, _ = build_split({}, _provenance(), "boston-seaport", 0.2, 42)
    seen = [t for logs in splits.values() for t in logs]
    assert len(seen) == len(set(seen)), "a log leaked across splits"


def test_test_set_is_exactly_the_held_out_location():
    prov = _provenance()
    splits, _, log_location = build_split({}, prov, "boston-seaport", 0.2, 42)
    assert {log_location[t] for t in splits["test"]} == {"boston-seaport"}
    for other in ("train", "val"):
        assert "boston-seaport" not in {log_location[t] for t in splits[other]}


def test_split_is_deterministic():
    prov = _provenance()
    a, _, _ = build_split({}, prov, "boston-seaport", 0.2, 42)
    b, _, _ = build_split({}, prov, "boston-seaport", 0.2, 42)
    assert a == b


def test_a_different_seed_gives_a_different_partition():
    """Guards against the seed being silently ignored."""
    prov = _provenance(n_logs=40)
    a, _, _ = build_split({}, prov, "boston-seaport", 0.2, 1)
    b, _, _ = build_split({}, prov, "boston-seaport", 0.2, 2)
    assert a["val"] != b["val"]


def test_every_split_is_non_empty():
    splits, _, _ = build_split({}, _provenance(), "boston-seaport", 0.2, 42)
    for name, logs in splits.items():
        assert logs, f"{name} split is empty"


@pytest.mark.parametrize("source,target", sorted(NUIMAGES_TO_DETECTION.items()))
def test_every_mapping_target_is_a_known_class(source: str, target: str):
    assert target in DETECTION_CLASSES


def test_category_ids_are_stable_and_one_based():
    assert CLASS_TO_ID == {n: i + 1 for i, n in enumerate(DETECTION_CLASSES)}
    assert len(DETECTION_CLASSES) == 10
